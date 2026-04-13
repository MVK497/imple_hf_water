from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyscf import cc, gto, scf

from .rhf import RHFResult


@dataclass
class CCSDResult:
    rhf_energy: float
    ccsd_correlation_energy: float
    total_energy: float
    orbital_energies: np.ndarray
    nocc: int
    nvir: int
    t1: np.ndarray
    t2: np.ndarray
    t1_norm: float
    t2_norm: float
    converged: bool


def build_reference_mf(mol: gto.Mole, rhf_result: RHFResult) -> tuple[scf.RHF, np.ndarray]:
    mf = scf.RHF(mol)
    nocc = mol.nelectron // 2
    mo_occ = np.zeros(mol.nao_nr())
    mo_occ[:nocc] = 2.0
    mf.mo_coeff = rhf_result.coefficients
    mf.mo_energy = rhf_result.orbital_energies
    mf.mo_occ = mo_occ
    mf.e_tot = rhf_result.energy
    mf.converged = True
    return mf, mo_occ


def run_ccsd(mol: gto.Mole, rhf_result: RHFResult) -> CCSDResult:
    mf, mo_occ = build_reference_mf(mol, rhf_result)
    ccsd_solver = cc.CCSD(mf, mo_coeff=rhf_result.coefficients, mo_occ=mo_occ)
    ccsd_solver.verbose = 0
    ccsd_solver.conv_tol = 1.0e-10
    correlation_energy, t1, t2 = ccsd_solver.kernel()

    nocc = mol.nelectron // 2
    nvir = rhf_result.coefficients.shape[1] - nocc
    total_energy = rhf_result.energy + correlation_energy

    return CCSDResult(
        rhf_energy=float(rhf_result.energy),
        ccsd_correlation_energy=float(correlation_energy),
        total_energy=float(total_energy),
        orbital_energies=rhf_result.orbital_energies.copy(),
        nocc=nocc,
        nvir=nvir,
        t1=t1.copy(),
        t2=t2.copy(),
        t1_norm=float(np.linalg.norm(t1)),
        t2_norm=float(np.linalg.norm(t2)),
        converged=bool(ccsd_solver.converged),
    )
