from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyscf.hessian import thermo

from .geometry import MoleculeSpec
from .rks import RKSResult, build_rks_reference_mf, run_rks
from .rhf import RHFResult, build_molecule, build_rhf_reference_mf, run_rhf
from .uks import UKSResult, build_uks_reference_mf, run_uks
from .uhf import UHFResult, build_uhf_reference_mf, run_uhf


ReferenceResult = RHFResult | UHFResult | RKSResult | UKSResult


@dataclass
class FrequencyResult:
    method: str
    xc: str | None
    energy: float
    hessian: np.ndarray
    frequencies_cm1: np.ndarray
    frequencies_au: np.ndarray
    normal_modes: np.ndarray
    reduced_masses: np.ndarray
    force_constants_au: np.ndarray
    force_constants_dyne: np.ndarray
    vib_temperatures: np.ndarray
    num_imaginary: int
    freq_error: int
    reference_result: ReferenceResult


def run_frequency(
    spec: MoleculeSpec,
    method: str,
    xc: str = "b3lyp",
    max_iter: int = 100,
    e_tol: float = 1.0e-10,
    d_tol: float = 1.0e-8,
    use_diis: bool = True,
    diis_space: int = 6,
) -> FrequencyResult:
    if method not in {"rhf", "uhf", "rks", "uks"}:
        raise ValueError("Frequency analysis currently supports RHF, UHF, RKS, and UKS.")

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
    else:
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

    hessian_solver = mf.Hessian()
    hessian_solver.verbose = 0
    hessian = np.array(hessian_solver.kernel(), copy=True)
    harmonic_data = thermo.harmonic_analysis(mol, hessian, imaginary_freq=True)

    frequencies_cm1 = np.array(harmonic_data["freq_wavenumber"], copy=True)
    num_imaginary = int(np.count_nonzero(np.abs(frequencies_cm1.imag) > 1.0e-8))

    return FrequencyResult(
        method=method,
        xc=xc if method in {"rks", "uks"} else None,
        energy=float(result.energy),
        hessian=hessian,
        frequencies_cm1=frequencies_cm1,
        frequencies_au=np.array(harmonic_data["freq_au"], copy=True),
        normal_modes=np.array(harmonic_data["norm_mode"], copy=True),
        reduced_masses=np.array(harmonic_data["reduced_mass"], copy=True),
        force_constants_au=np.array(harmonic_data["force_const_au"], copy=True),
        force_constants_dyne=np.array(harmonic_data["force_const_dyne"], copy=True),
        vib_temperatures=np.array(harmonic_data["vib_temperature"], copy=True),
        num_imaginary=num_imaginary,
        freq_error=int(harmonic_data["freq_error"]),
        reference_result=result,
    )
