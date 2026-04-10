from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyscf import gto

from .uhf import UHFResult


@dataclass
class UMP2Result:
    uhf_energy: float
    correlation_energy_aa: float
    correlation_energy_ab: float
    correlation_energy_bb: float
    ump2_correlation_energy: float
    total_energy: float
    orbital_energies_alpha: np.ndarray
    orbital_energies_beta: np.ndarray
    nocc_alpha: int
    nocc_beta: int
    nvir_alpha: int
    nvir_beta: int


def transform_ao_eri_to_ovov(
    ao_eri: np.ndarray,
    occ_left: np.ndarray,
    vir_left: np.ndarray,
    occ_right: np.ndarray,
    vir_right: np.ndarray,
) -> np.ndarray:
    return np.einsum(
        "pqrs,pi,qa,rj,sb->iajb",
        ao_eri,
        occ_left,
        vir_left,
        occ_right,
        vir_right,
        optimize=True,
    )


def same_spin_mp2_energy(eri_ovov: np.ndarray, eps_occ: np.ndarray, eps_vir: np.ndarray) -> float:
    if eps_occ.size == 0 or eps_vir.size == 0:
        return 0.0

    exchange_ovov = eri_ovov.transpose(0, 3, 2, 1)
    antisymmetrized = eri_ovov - exchange_ovov
    denominators = (
        eps_occ[:, None, None, None]
        + eps_occ[None, None, :, None]
        - eps_vir[None, :, None, None]
        - eps_vir[None, None, None, :]
    )
    return 0.25 * np.sum(antisymmetrized * antisymmetrized / denominators)


def opposite_spin_mp2_energy(
    eri_ovov: np.ndarray,
    eps_occ_left: np.ndarray,
    eps_occ_right: np.ndarray,
    eps_vir_left: np.ndarray,
    eps_vir_right: np.ndarray,
) -> float:
    if eps_occ_left.size == 0 or eps_occ_right.size == 0 or eps_vir_left.size == 0 or eps_vir_right.size == 0:
        return 0.0

    denominators = (
        eps_occ_left[:, None, None, None]
        + eps_occ_right[None, None, :, None]
        - eps_vir_left[None, :, None, None]
        - eps_vir_right[None, None, None, :]
    )
    return np.sum(eri_ovov * eri_ovov / denominators)


def run_ump2(mol: gto.Mole, uhf_result: UHFResult) -> UMP2Result:
    occ_alpha = uhf_result.coefficients_alpha[:, : uhf_result.nalpha]
    occ_beta = uhf_result.coefficients_beta[:, : uhf_result.nbeta]
    vir_alpha = uhf_result.coefficients_alpha[:, uhf_result.nalpha :]
    vir_beta = uhf_result.coefficients_beta[:, uhf_result.nbeta :]

    eps_occ_alpha = uhf_result.orbital_energies_alpha[: uhf_result.nalpha]
    eps_occ_beta = uhf_result.orbital_energies_beta[: uhf_result.nbeta]
    eps_vir_alpha = uhf_result.orbital_energies_alpha[uhf_result.nalpha :]
    eps_vir_beta = uhf_result.orbital_energies_beta[uhf_result.nbeta :]

    ao_eri = mol.intor("int2e")
    eri_aa = transform_ao_eri_to_ovov(ao_eri, occ_alpha, vir_alpha, occ_alpha, vir_alpha)
    eri_ab = transform_ao_eri_to_ovov(ao_eri, occ_alpha, vir_alpha, occ_beta, vir_beta)
    eri_bb = transform_ao_eri_to_ovov(ao_eri, occ_beta, vir_beta, occ_beta, vir_beta)

    corr_aa = same_spin_mp2_energy(eri_aa, eps_occ_alpha, eps_vir_alpha)
    corr_ab = opposite_spin_mp2_energy(
        eri_ab,
        eps_occ_alpha,
        eps_occ_beta,
        eps_vir_alpha,
        eps_vir_beta,
    )
    corr_bb = same_spin_mp2_energy(eri_bb, eps_occ_beta, eps_vir_beta)
    total_corr = corr_aa + corr_ab + corr_bb

    return UMP2Result(
        uhf_energy=float(uhf_result.energy),
        correlation_energy_aa=float(corr_aa),
        correlation_energy_ab=float(corr_ab),
        correlation_energy_bb=float(corr_bb),
        ump2_correlation_energy=float(total_corr),
        total_energy=float(uhf_result.energy + total_corr),
        orbital_energies_alpha=uhf_result.orbital_energies_alpha.copy(),
        orbital_energies_beta=uhf_result.orbital_energies_beta.copy(),
        nocc_alpha=uhf_result.nalpha,
        nocc_beta=uhf_result.nbeta,
        nvir_alpha=vir_alpha.shape[1],
        nvir_beta=vir_beta.shape[1],
    )
