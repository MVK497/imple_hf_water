from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .ccsd import CCSDResult, run_ccsd
from .geometry import (
    MoleculeSpec,
    angle_degrees,
    angle_radians,
    bond_length,
    convert_coords,
    convert_length_value,
    dihedral_degrees,
    dihedral_radians,
    format_atom_string,
    parse_atom_string,
    set_angle_rigid,
    set_bond_length_rigid,
    set_dihedral_rigid,
    wrap_angle_radians,
)
from .mp2 import MP2Result, run_mp2
from .optimize import OptimizationResult, optimize_geometry
from .rks import RKSResult, run_rks
from .rhf import RHFResult, build_molecule, run_rhf
from .ump2 import UMP2Result, run_ump2
from .uks import UKSResult, run_uks
from .uhf import UHFResult, run_uhf


SinglePointResult = RHFResult | UHFResult | RKSResult | UKSResult | MP2Result | UMP2Result | CCSDResult


@dataclass
class ScanPoint:
    index: int
    target_value: float
    actual_value: float
    energy: float
    converged: bool
    spec: MoleculeSpec
    result: SinglePointResult | RHFResult | UHFResult | None
    optimization_result: OptimizationResult | None = None


@dataclass
class ScanResult:
    mode: str
    coordinate_type: str
    method: str
    xc: str | None
    atoms: tuple[int, ...]
    value_unit: str
    points: list[ScanPoint]
    best_point: ScanPoint


def coordinate_arity(coordinate_type: str) -> int:
    return {"bond": 2, "angle": 3, "dihedral": 4}[coordinate_type]


def coordinate_display_unit(coordinate_type: str, spec_unit: str) -> str:
    if coordinate_type == "bond":
        return spec_unit
    return "deg"


def coordinate_value_display(coords: np.ndarray, coordinate_type: str, atoms: tuple[int, ...]) -> float:
    if coordinate_type == "bond":
        return bond_length(coords, atoms)  # type: ignore[arg-type]
    if coordinate_type == "angle":
        return angle_degrees(coords, atoms)  # type: ignore[arg-type]
    if coordinate_type == "dihedral":
        return dihedral_degrees(coords, atoms)  # type: ignore[arg-type]
    raise ValueError(f"Unsupported coordinate type '{coordinate_type}'.")


def coordinate_value_internal(coords_bohr: np.ndarray, coordinate_type: str, atoms: tuple[int, ...]) -> float:
    if coordinate_type == "bond":
        return bond_length(coords_bohr, atoms)  # type: ignore[arg-type]
    if coordinate_type == "angle":
        return angle_radians(coords_bohr, atoms)  # type: ignore[arg-type]
    if coordinate_type == "dihedral":
        return dihedral_radians(coords_bohr, atoms)  # type: ignore[arg-type]
    raise ValueError(f"Unsupported coordinate type '{coordinate_type}'.")


def set_coordinate_rigid(
    coords: np.ndarray,
    coordinate_type: str,
    atoms: tuple[int, ...],
    target_value: float,
) -> np.ndarray:
    if coordinate_type == "bond":
        return set_bond_length_rigid(coords, atoms, target_value)  # type: ignore[arg-type]
    if coordinate_type == "angle":
        return set_angle_rigid(coords, atoms, target_value)  # type: ignore[arg-type]
    if coordinate_type == "dihedral":
        return set_dihedral_rigid(coords, atoms, target_value)  # type: ignore[arg-type]
    raise ValueError(f"Unsupported coordinate type '{coordinate_type}'.")


def target_value_to_internal(target_value: float, coordinate_type: str, spec_unit: str) -> float:
    if coordinate_type == "bond":
        return convert_length_value(target_value, spec_unit, "Bohr")
    if coordinate_type in {"angle", "dihedral"}:
        return float(np.deg2rad(target_value))
    raise ValueError(f"Unsupported coordinate type '{coordinate_type}'.")


def displacement_unit_for_coordinate(coordinate_type: str) -> str:
    if coordinate_type == "bond":
        return "length"
    return "angular"


def periodic_difference(value: float, target: float, coordinate_type: str) -> float:
    delta = value - target
    if coordinate_type == "dihedral":
        return wrap_angle_radians(delta)
    return delta


def numerical_gradient(
    energy_fn: Callable[[np.ndarray], float],
    coords: np.ndarray,
    step: float = 1.0e-5,
) -> np.ndarray:
    gradient = np.zeros_like(coords)
    for atom_index in range(coords.shape[0]):
        for axis in range(3):
            plus = np.array(coords, copy=True)
            minus = np.array(coords, copy=True)
            plus[atom_index, axis] += step
            minus[atom_index, axis] -= step
            gradient[atom_index, axis] = (energy_fn(plus) - energy_fn(minus)) / (2.0 * step)
    return gradient


def build_coordinate_penalty(
    coordinate_type: str,
    atoms: tuple[int, ...],
    target_internal: float,
    penalty_k: float,
) -> Callable[[np.ndarray], tuple[float, np.ndarray]]:
    def penalty_energy(coords_bohr: np.ndarray) -> float:
        current_internal = coordinate_value_internal(coords_bohr, coordinate_type, atoms)
        delta = periodic_difference(current_internal, target_internal, coordinate_type)
        return 0.5 * penalty_k * delta * delta

    def penalty(coords_bohr: np.ndarray) -> tuple[float, np.ndarray]:
        energy = float(penalty_energy(coords_bohr))
        gradient = numerical_gradient(penalty_energy, coords_bohr)
        return energy, gradient

    return penalty


def evaluate_single_point(
    spec: MoleculeSpec,
    method: str,
    xc: str,
    max_iter: int,
    e_tol: float,
    d_tol: float,
    use_diis: bool,
    diis_space: int,
) -> tuple[float, SinglePointResult]:
    mol = build_molecule(spec)
    if method == "rhf":
        result = run_rhf(mol, max_iter=max_iter, e_tol=e_tol, d_tol=d_tol, use_diis=use_diis, diis_space=diis_space)
        return result.energy, result
    if method == "rks":
        result = run_rks(
            mol,
            xc=xc,
            max_iter=max_iter,
            e_tol=e_tol,
            d_tol=d_tol,
            use_diis=use_diis,
            diis_space=diis_space,
        )
        return result.energy, result
    if method == "mp2":
        rhf_result = run_rhf(mol, max_iter=max_iter, e_tol=e_tol, d_tol=d_tol, use_diis=use_diis, diis_space=diis_space)
        result = run_mp2(mol, rhf_result)
        return result.total_energy, result
    if method == "ccsd":
        rhf_result = run_rhf(mol, max_iter=max_iter, e_tol=e_tol, d_tol=d_tol, use_diis=use_diis, diis_space=diis_space)
        result = run_ccsd(mol, rhf_result)
        return result.total_energy, result
    if method == "uhf":
        result = run_uhf(mol, max_iter=max_iter, e_tol=e_tol, d_tol=d_tol, use_diis=use_diis, diis_space=diis_space)
        return result.energy, result
    if method == "uks":
        result = run_uks(
            mol,
            xc=xc,
            max_iter=max_iter,
            e_tol=e_tol,
            d_tol=d_tol,
            use_diis=use_diis,
            diis_space=diis_space,
        )
        return result.energy, result
    if method == "ump2":
        uhf_result = run_uhf(mol, max_iter=max_iter, e_tol=e_tol, d_tol=d_tol, use_diis=use_diis, diis_space=diis_space)
        result = run_ump2(mol, uhf_result)
        return result.total_energy, result
    raise ValueError(f"Unsupported scan method '{method}'.")


def rigid_scan(
    spec: MoleculeSpec,
    method: str,
    xc: str,
    coordinate_type: str,
    atoms: tuple[int, ...],
    start_value: float,
    stop_value: float,
    num_points: int,
    max_iter: int,
    e_tol: float,
    d_tol: float,
    use_diis: bool,
    diis_space: int,
) -> ScanResult:
    symbols, coords = parse_atom_string(spec.atom)
    values = np.linspace(start_value, stop_value, num_points)
    points: list[ScanPoint] = []

    for index, target_value in enumerate(values, start=1):
        scan_coords = set_coordinate_rigid(coords, coordinate_type, atoms, float(target_value))
        scan_spec = MoleculeSpec(
            atom=format_atom_string(symbols, scan_coords),
            basis=spec.basis,
            charge=spec.charge,
            spin=spec.spin,
            unit=spec.unit,
            title=spec.title,
        )
        actual = coordinate_value_display(scan_coords, coordinate_type, atoms)
        try:
            energy, result = evaluate_single_point(
                scan_spec,
                method,
                xc,
                max_iter,
                e_tol,
                d_tol,
                use_diis,
                diis_space,
            )
            converged = True
        except RuntimeError:
            energy = float("inf")
            result = None
            converged = False
        points.append(
            ScanPoint(
                index=index,
                target_value=float(target_value),
                actual_value=actual,
                energy=float(energy),
                converged=converged,
                spec=scan_spec,
                result=result,
            )
        )

    best_point = min(points, key=lambda point: point.energy)
    return ScanResult(
        mode="rigid",
        coordinate_type=coordinate_type,
        method=method,
        xc=xc if method in {"rks", "uks"} else None,
        atoms=atoms,
        value_unit=coordinate_display_unit(coordinate_type, spec.unit),
        points=points,
        best_point=best_point,
    )


def anchor_translation_bohr(
    coordinate_type: str,
    atoms: tuple[int, ...],
    reference_coords_bohr: np.ndarray,
    optimized_coords_bohr: np.ndarray,
) -> np.ndarray:
    if coordinate_type == "bond":
        anchor_index = atoms[0]
    elif coordinate_type == "angle":
        anchor_index = atoms[1]
    elif coordinate_type == "dihedral":
        anchor_index = atoms[1]
    else:
        raise ValueError(f"Unsupported coordinate type '{coordinate_type}'.")
    return reference_coords_bohr[anchor_index] - optimized_coords_bohr[anchor_index]


def relaxed_scan(
    spec: MoleculeSpec,
    method: str,
    xc: str,
    coordinate_type: str,
    atoms: tuple[int, ...],
    start_value: float,
    stop_value: float,
    num_points: int,
    penalty_k: float,
    max_opt_steps: int,
    grad_tol: float,
    opt_energy_tol: float,
    max_step_size: float,
    max_iter: int,
    scf_e_tol: float,
    scf_d_tol: float,
    use_diis: bool,
    diis_space: int,
) -> ScanResult:
    if method not in {"rhf", "uhf", "rks", "uks"}:
        raise ValueError("Relaxed scans currently support RHF, UHF, RKS, and UKS.")

    symbols, coords_input = parse_atom_string(spec.atom)
    coords_bohr = convert_coords(coords_input, spec.unit, "Bohr")
    display_values = np.linspace(start_value, stop_value, num_points)
    points: list[ScanPoint] = []
    working_coords_bohr = np.array(coords_bohr, copy=True)

    for index, target_display in enumerate(display_values, start=1):
        if coordinate_type == "bond":
            target_for_rigid = convert_length_value(float(target_display), spec.unit, "Bohr")
        else:
            target_for_rigid = float(target_display)

        start_coords_bohr = set_coordinate_rigid(
            working_coords_bohr,
            coordinate_type,
            atoms,
            target_for_rigid,
        )
        start_spec = MoleculeSpec(
            atom=format_atom_string(symbols, start_coords_bohr),
            basis=spec.basis,
            charge=spec.charge,
            spin=spec.spin,
            unit="Bohr",
            title=spec.title,
        )
        target_internal = target_value_to_internal(float(target_display), coordinate_type, spec.unit)
        try:
            opt_result = optimize_geometry(
                start_spec,
                method=method,
                xc=xc,
                max_opt_steps=max_opt_steps,
                grad_tol=grad_tol,
                energy_tol=opt_energy_tol,
                max_step_size=max_step_size,
                max_iter=max_iter,
                scf_e_tol=scf_e_tol,
                scf_d_tol=scf_d_tol,
                use_diis=use_diis,
                diis_space=diis_space,
                penalty_function=build_coordinate_penalty(coordinate_type, atoms, target_internal, penalty_k),
            )

            optimized_symbols, optimized_coords_out = parse_atom_string(opt_result.optimized_spec.atom)
            optimized_coords_bohr = convert_coords(optimized_coords_out, opt_result.optimized_spec.unit, "Bohr")
            translation_bohr = anchor_translation_bohr(
                coordinate_type,
                atoms,
                coords_bohr,
                optimized_coords_bohr,
            )
            optimized_coords_bohr = optimized_coords_bohr + translation_bohr
            optimized_coords_out = convert_coords(optimized_coords_bohr, "Bohr", spec.unit)
            working_coords_bohr = optimized_coords_bohr
            actual_display = coordinate_value_display(optimized_coords_out, coordinate_type, atoms)
            true_energy = float(opt_result.final_result.energy)
            point_spec = MoleculeSpec(
                atom=format_atom_string(optimized_symbols, optimized_coords_out),
                basis=spec.basis,
                charge=spec.charge,
                spin=spec.spin,
                unit=spec.unit,
                title=spec.title,
            )
            point_result = opt_result.final_result
            point_converged = opt_result.converged
        except RuntimeError:
            fallback_coords_out = convert_coords(start_coords_bohr, "Bohr", spec.unit)
            actual_display = coordinate_value_display(fallback_coords_out, coordinate_type, atoms)
            true_energy = float("inf")
            point_spec = MoleculeSpec(
                atom=format_atom_string(symbols, fallback_coords_out),
                basis=spec.basis,
                charge=spec.charge,
                spin=spec.spin,
                unit=spec.unit,
                title=spec.title,
            )
            point_result = None
            point_converged = False
            opt_result = None

        points.append(
            ScanPoint(
                index=index,
                target_value=float(target_display),
                actual_value=actual_display,
                energy=true_energy,
                converged=point_converged,
                spec=point_spec,
                result=point_result,
                optimization_result=opt_result,
            )
        )

    best_point = min(points, key=lambda point: point.energy)
    return ScanResult(
        mode="relaxed",
        coordinate_type=coordinate_type,
        method=method,
        xc=xc if method in {"rks", "uks"} else None,
        atoms=atoms,
        value_unit=coordinate_display_unit(coordinate_type, spec.unit),
        points=points,
        best_point=best_point,
    )


def write_scan_csv(scan_result: ScanResult, path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "index",
                "coordinate_type",
                "target_value",
                "actual_value",
                "value_unit",
                "energy_hartree",
                "converged",
            ]
        )
        for point in scan_result.points:
            writer.writerow(
                [
                    point.index,
                    scan_result.coordinate_type,
                    f"{point.target_value:.10f}",
                    f"{point.actual_value:.10f}",
                    scan_result.value_unit,
                    f"{point.energy:.12f}",
                    point.converged,
                ]
            )
