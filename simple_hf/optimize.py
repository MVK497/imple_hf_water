from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .geometry import MoleculeSpec, convert_coords, format_atom_string, parse_atom_string
from .rks import RKSResult, build_rks_reference_mf, run_rks
from .rhf import RHFResult, build_molecule, build_rhf_reference_mf, run_rhf
from .uks import UKSResult, build_uks_reference_mf, run_uks
from .uhf import UHFResult, build_uhf_reference_mf, run_uhf


@dataclass
class OptimizationStep:
    step: int
    energy: float
    max_gradient: float
    rms_gradient: float
    step_length: float


@dataclass
class OptimizationResult:
    method: str
    xc: str | None
    initial_energy: float
    final_energy: float
    converged: bool
    iterations: int
    optimized_spec: MoleculeSpec
    final_gradient: np.ndarray
    history: list[OptimizationStep]
    final_result: RHFResult | UHFResult | RKSResult | UKSResult


PenaltyFunction = Callable[[np.ndarray], tuple[float, np.ndarray]]


def evaluate_energy_and_gradient(
    spec: MoleculeSpec,
    method: str,
    xc: str,
    max_iter: int,
    e_tol: float,
    d_tol: float,
    use_diis: bool,
    diis_space: int,
    penalty_function: PenaltyFunction | None = None,
) -> tuple[float, np.ndarray, RHFResult | UHFResult | RKSResult | UKSResult]:
    mol = build_molecule(spec)
    if method == "rhf":
        result = run_rhf(
            mol,
            max_iter=max_iter,
            e_tol=e_tol,
            d_tol=d_tol,
            use_diis=use_diis,
            diis_space=diis_space,
        )
        mf, _ = build_rhf_reference_mf(mol, result)
    elif method == "rks":
        result = run_rks(
            mol,
            xc=xc,
            max_iter=max_iter,
            e_tol=e_tol,
            d_tol=d_tol,
            use_diis=use_diis,
            diis_space=diis_space,
        )
        mf, _ = build_rks_reference_mf(mol, result)
    elif method == "uhf":
        result = run_uhf(
            mol,
            max_iter=max_iter,
            e_tol=e_tol,
            d_tol=d_tol,
            use_diis=use_diis,
            diis_space=diis_space,
        )
        mf, _ = build_uhf_reference_mf(mol, result)
    elif method == "uks":
        result = run_uks(
            mol,
            xc=xc,
            max_iter=max_iter,
            e_tol=e_tol,
            d_tol=d_tol,
            use_diis=use_diis,
            diis_space=diis_space,
        )
        mf, _ = build_uks_reference_mf(mol, result)
    else:
        raise ValueError(f"Unsupported optimization method '{method}'.")

    grad_solver = mf.nuc_grad_method()
    grad_solver.verbose = 0
    gradient = grad_solver.kernel()
    total_energy = float(result.energy)
    total_gradient = np.array(gradient, copy=True)

    if penalty_function is not None:
        _, coords_bohr = parse_atom_string(spec.atom)
        penalty_energy, penalty_gradient = penalty_function(coords_bohr)
        total_energy += float(penalty_energy)
        total_gradient = total_gradient + np.array(penalty_gradient, copy=False)

    return total_energy, total_gradient, result


def make_spec_with_bohr_coords(
    symbols: list[str],
    coords_bohr: np.ndarray,
    template: MoleculeSpec,
) -> MoleculeSpec:
    return MoleculeSpec(
        atom=format_atom_string(symbols, coords_bohr),
        basis=template.basis,
        charge=template.charge,
        spin=template.spin,
        unit="Bohr",
        title=template.title,
    )


def format_spec_in_original_unit(spec_bohr: MoleculeSpec, output_unit: str) -> MoleculeSpec:
    symbols, coords_bohr = parse_atom_string(spec_bohr.atom)
    coords_out = convert_coords(coords_bohr, "Bohr", output_unit)
    return MoleculeSpec(
        atom=format_atom_string(symbols, coords_out),
        basis=spec_bohr.basis,
        charge=spec_bohr.charge,
        spin=spec_bohr.spin,
        unit=output_unit,
        title=spec_bohr.title,
    )


def optimize_geometry(
    spec: MoleculeSpec,
    method: str,
    xc: str = "b3lyp",
    max_opt_steps: int = 30,
    grad_tol: float = 1.0e-4,
    energy_tol: float = 1.0e-8,
    max_step_size: float = 0.2,
    max_iter: int = 100,
    scf_e_tol: float = 1.0e-10,
    scf_d_tol: float = 1.0e-8,
    use_diis: bool = True,
    diis_space: int = 6,
    penalty_function: PenaltyFunction | None = None,
) -> OptimizationResult:
    if method not in {"rhf", "uhf", "rks", "uks"}:
        raise ValueError("Geometry optimization currently supports RHF, UHF, RKS, and UKS.")

    symbols, coords_input = parse_atom_string(spec.atom)
    coords_bohr = convert_coords(coords_input, spec.unit, "Bohr")
    current_spec = make_spec_with_bohr_coords(symbols, coords_bohr, spec)
    current_energy, current_gradient, current_result = evaluate_energy_and_gradient(
        current_spec,
        method,
        xc,
        max_iter,
        scf_e_tol,
        scf_d_tol,
        use_diis,
        diis_space,
        penalty_function,
    )

    initial_energy = current_energy
    history: list[OptimizationStep] = []
    dimension = current_gradient.size
    inverse_hessian = np.eye(dimension)
    previous_energy = None
    converged = False

    for step in range(1, max_opt_steps + 1):
        gradient_flat = current_gradient.reshape(-1)
        max_gradient = float(np.max(np.abs(gradient_flat)))
        rms_gradient = float(np.sqrt(np.mean(gradient_flat**2)))

        if previous_energy is not None and abs(current_energy - previous_energy) < energy_tol and max_gradient < grad_tol:
            converged = True
            break

        if previous_energy is None and max_gradient < grad_tol:
            history.append(
                OptimizationStep(
                    step=step,
                    energy=current_energy,
                    max_gradient=max_gradient,
                    rms_gradient=rms_gradient,
                    step_length=0.0,
                )
            )
            converged = True
            break

        direction = -inverse_hessian @ gradient_flat
        if float(np.dot(direction, gradient_flat)) >= 0.0:
            direction = -gradient_flat.copy()
            inverse_hessian = np.eye(dimension)

        direction_norm = float(np.linalg.norm(direction))
        if direction_norm > max_step_size:
            direction *= max_step_size / direction_norm

        trial_alpha = 1.0
        accepted = False
        c1 = 1.0e-4
        best_trial: tuple[
            float,
            np.ndarray,
            RHFResult | UHFResult | RKSResult | UKSResult,
            np.ndarray,
        ] | None = None

        for _ in range(12):
            trial_coords_bohr = coords_bohr + trial_alpha * direction.reshape(-1, 3)
            trial_spec = make_spec_with_bohr_coords(symbols, trial_coords_bohr, spec)
            try:
                trial_energy, trial_gradient, trial_result = evaluate_energy_and_gradient(
                    trial_spec,
                    method,
                    xc,
                    max_iter,
                    scf_e_tol,
                    scf_d_tol,
                    use_diis,
                    diis_space,
                    penalty_function,
                )
            except RuntimeError:
                trial_alpha *= 0.5
                continue
            if trial_energy <= current_energy + c1 * trial_alpha * float(np.dot(gradient_flat, direction)):
                best_trial = (trial_energy, trial_gradient, trial_result, trial_coords_bohr)
                accepted = True
                break
            trial_alpha *= 0.5

        if not accepted or best_trial is None:
            history.append(
                OptimizationStep(
                    step=step,
                    energy=current_energy,
                    max_gradient=max_gradient,
                    rms_gradient=rms_gradient,
                    step_length=0.0,
                )
            )
            break

        trial_energy, trial_gradient, trial_result, trial_coords_bohr = best_trial
        step_vector = (trial_coords_bohr - coords_bohr).reshape(-1)
        new_gradient_flat = trial_gradient.reshape(-1)
        y_vector = new_gradient_flat - gradient_flat
        ys = float(np.dot(y_vector, step_vector))

        if ys > 1.0e-10:
            identity = np.eye(dimension)
            rho = 1.0 / ys
            outer_sy = np.outer(step_vector, y_vector)
            outer_ys = np.outer(y_vector, step_vector)
            inverse_hessian = (
                (identity - rho * outer_sy)
                @ inverse_hessian
                @ (identity - rho * outer_ys)
                + rho * np.outer(step_vector, step_vector)
            )
        else:
            inverse_hessian = np.eye(dimension)

        history.append(
            OptimizationStep(
                step=step,
                energy=trial_energy,
                max_gradient=float(np.max(np.abs(new_gradient_flat))),
                rms_gradient=float(np.sqrt(np.mean(new_gradient_flat**2))),
                step_length=float(np.linalg.norm(step_vector)),
            )
        )

        previous_energy = current_energy
        coords_bohr = trial_coords_bohr
        current_spec = make_spec_with_bohr_coords(symbols, coords_bohr, spec)
        current_energy = trial_energy
        current_gradient = trial_gradient
        current_result = trial_result

    optimized_spec = format_spec_in_original_unit(current_spec, spec.unit)
    return OptimizationResult(
        method=method,
        xc=xc if method in {"rks", "uks"} else None,
        initial_energy=initial_energy,
        final_energy=current_energy,
        converged=converged,
        iterations=len(history),
        optimized_spec=optimized_spec,
        final_gradient=np.array(current_gradient, copy=True),
        history=history,
        final_result=current_result,
    )
