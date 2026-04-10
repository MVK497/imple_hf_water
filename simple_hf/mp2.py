from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyscf import gto

from .rhf import RHFResult


@dataclass
class MP2Result:
    rhf_energy: float
    mp2_correlation_energy: float
    total_energy: float
    orbital_energies: np.ndarray
    nocc: int
    nvir: int


def transform_ao_eri_to_ovov(
    ao_eri: np.ndarray,
    occ_coeff: np.ndarray,
    vir_coeff: np.ndarray,
) -> np.ndarray:
    return np.einsum(
        "pqrs,pi,qa,rj,sb->iajb",
        ao_eri,
        occ_coeff,
        vir_coeff,
        occ_coeff,
        vir_coeff,
        optimize=True,
    )


def run_mp2(mol: gto.Mole, rhf_result: RHFResult) -> MP2Result:
    nocc = mol.nelectron // 2
    nmo = rhf_result.coefficients.shape[1]
    nvir = nmo - nocc
    if nvir <= 0:
        raise ValueError("MP2 requires at least one virtual orbital.")

    occ_coeff = rhf_result.coefficients[:, :nocc]
    vir_coeff = rhf_result.coefficients[:, nocc:]
    eps_occ = rhf_result.orbital_energies[:nocc]
    eps_vir = rhf_result.orbital_energies[nocc:]

    ao_eri = mol.intor("int2e")
    eri_ovov = transform_ao_eri_to_ovov(ao_eri, occ_coeff, vir_coeff)
    exchange_ovov = eri_ovov.transpose(0, 3, 2, 1)

    denominators = (
        eps_occ[:, None, None, None]
        + eps_occ[None, None, :, None]
        - eps_vir[None, :, None, None]
        - eps_vir[None, None, None, :]
    )

    correlation_energy = np.sum(
        eri_ovov * (2.0 * eri_ovov - exchange_ovov) / denominators
    )
    total_energy = rhf_result.energy + correlation_energy

    return MP2Result(
        rhf_energy=float(rhf_result.energy),
        mp2_correlation_energy=float(correlation_energy),
        total_energy=float(total_energy),
        orbital_energies=rhf_result.orbital_energies.copy(),
        nocc=nocc,
        nvir=nvir,
    )
