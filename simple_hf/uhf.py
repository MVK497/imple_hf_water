from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyscf import gto, scf

from .rhf import (
    DIISHelper,
    compute_diis_error,
    diagonalize_fock,
    symmetric_orthogonalization,
)


@dataclass
class UHFResult:
    energy: float
    electronic_energy: float
    nuclear_repulsion: float
    orbital_energies_alpha: np.ndarray
    orbital_energies_beta: np.ndarray
    coefficients_alpha: np.ndarray
    coefficients_beta: np.ndarray
    density_alpha: np.ndarray
    density_beta: np.ndarray
    iterations: int
    history: list[float]
    nalpha: int
    nbeta: int
    s2: float
    expected_s2: float
    spin_contamination: float


def build_spin_density(coefficients: np.ndarray, nocc: int) -> np.ndarray:
    if nocc == 0:
        return np.zeros((coefficients.shape[0], coefficients.shape[0]))
    occupied = coefficients[:, :nocc]
    return occupied @ occupied.T


def build_uhf_fock(
    h_core: np.ndarray,
    eri: np.ndarray,
    density_alpha: np.ndarray,
    density_beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    density_total = density_alpha + density_beta
    coulomb = np.einsum("ls,mnls->mn", density_total, eri, optimize=True)
    exchange_alpha = np.einsum("ls,mlns->mn", density_alpha, eri, optimize=True)
    exchange_beta = np.einsum("ls,mlns->mn", density_beta, eri, optimize=True)
    fock_alpha = h_core + coulomb - exchange_alpha
    fock_beta = h_core + coulomb - exchange_beta
    return fock_alpha, fock_beta


def combine_spin_blocks(matrix_alpha: np.ndarray, matrix_beta: np.ndarray) -> np.ndarray:
    zeros = np.zeros_like(matrix_alpha)
    return np.block([[matrix_alpha, zeros], [zeros, matrix_beta]])


def split_spin_blocks(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nbf = matrix.shape[0] // 2
    return matrix[:nbf, :nbf], matrix[nbf:, nbf:]


def _validate_uhf_inputs(mol: gto.Mole) -> tuple[int, int]:
    nalpha, nbeta = mol.nelec
    if nalpha < 0 or nbeta < 0:
        raise ValueError("Invalid alpha/beta electron counts for UHF.")
    return nalpha, nbeta


def compute_uhf_s2(
    overlap: np.ndarray,
    coeff_alpha: np.ndarray,
    coeff_beta: np.ndarray,
    nalpha: int,
    nbeta: int,
) -> tuple[float, float, float]:
    occ_alpha = coeff_alpha[:, :nalpha]
    occ_beta = coeff_beta[:, :nbeta]
    if nalpha == 0 or nbeta == 0:
        overlap_occ = 0.0
    else:
        spin_overlap = occ_alpha.T @ overlap @ occ_beta
        overlap_occ = float(np.sum(np.abs(spin_overlap) ** 2))

    sz = 0.5 * (nalpha - nbeta)
    s2 = sz * (sz + 1.0) + nbeta - overlap_occ
    expected_s2 = 0.5 * mol_spin_from_counts(nalpha, nbeta) * (0.5 * mol_spin_from_counts(nalpha, nbeta) + 1.0)
    return s2, expected_s2, s2 - expected_s2


def mol_spin_from_counts(nalpha: int, nbeta: int) -> int:
    return nalpha - nbeta


def run_uhf(
    mol: gto.Mole,
    max_iter: int = 100,
    e_tol: float = 1.0e-10,
    d_tol: float = 1.0e-8,
    use_diis: bool = True,
    diis_space: int = 6,
) -> UHFResult:
    nalpha, nbeta = _validate_uhf_inputs(mol)

    s = mol.intor("int1e_ovlp")
    t = mol.intor("int1e_kin")
    v = mol.intor("int1e_nuc")
    eri = mol.intor("int2e")
    h_core = t + v
    e_nuc = mol.energy_nuc()

    x = symmetric_orthogonalization(s)
    diis_helper = DIISHelper(max_vectors=diis_space)

    orbital_energies_alpha, coeff_alpha = diagonalize_fock(h_core, x)
    orbital_energies_beta, coeff_beta = diagonalize_fock(h_core, x)
    density_alpha = build_spin_density(coeff_alpha, nalpha)
    density_beta = build_spin_density(coeff_beta, nbeta)

    previous_energy = None
    history: list[float] = []

    for iteration in range(1, max_iter + 1):
        fock_alpha, fock_beta = build_uhf_fock(h_core, eri, density_alpha, density_beta)
        if use_diis:
            error_alpha = compute_diis_error(fock_alpha, density_alpha, s, x)
            error_beta = compute_diis_error(fock_beta, density_beta, s, x)
            diis_helper.push(
                combine_spin_blocks(fock_alpha, fock_beta),
                combine_spin_blocks(error_alpha, error_beta),
            )
            fock_alpha_to_diagonalize, fock_beta_to_diagonalize = split_spin_blocks(
                diis_helper.extrapolate()
            )
        else:
            fock_alpha_to_diagonalize = fock_alpha
            fock_beta_to_diagonalize = fock_beta

        orbital_energies_alpha, coeff_alpha = diagonalize_fock(fock_alpha_to_diagonalize, x)
        orbital_energies_beta, coeff_beta = diagonalize_fock(fock_beta_to_diagonalize, x)

        new_density_alpha = build_spin_density(coeff_alpha, nalpha)
        new_density_beta = build_spin_density(coeff_beta, nbeta)
        new_fock_alpha, new_fock_beta = build_uhf_fock(
            h_core,
            eri,
            new_density_alpha,
            new_density_beta,
        )

        electronic_energy = 0.5 * (
            np.sum((new_density_alpha + new_density_beta) * h_core)
            + np.sum(new_density_alpha * new_fock_alpha)
            + np.sum(new_density_beta * new_fock_beta)
        )
        total_energy = electronic_energy + e_nuc
        history.append(total_energy)

        density_change = np.sqrt(
            np.linalg.norm(new_density_alpha - density_alpha) ** 2
            + np.linalg.norm(new_density_beta - density_beta) ** 2
        )
        if previous_energy is not None and abs(total_energy - previous_energy) < e_tol and density_change < d_tol:
            orbital_energies_alpha, coeff_alpha = diagonalize_fock(new_fock_alpha, x)
            orbital_energies_beta, coeff_beta = diagonalize_fock(new_fock_beta, x)
            s2, expected_s2, spin_contamination = compute_uhf_s2(
                s,
                coeff_alpha,
                coeff_beta,
                nalpha,
                nbeta,
            )
            return UHFResult(
                energy=float(total_energy),
                electronic_energy=float(electronic_energy),
                nuclear_repulsion=float(e_nuc),
                orbital_energies_alpha=orbital_energies_alpha,
                orbital_energies_beta=orbital_energies_beta,
                coefficients_alpha=coeff_alpha,
                coefficients_beta=coeff_beta,
                density_alpha=new_density_alpha,
                density_beta=new_density_beta,
                iterations=iteration,
                history=history,
                nalpha=nalpha,
                nbeta=nbeta,
                s2=float(s2),
                expected_s2=float(expected_s2),
                spin_contamination=float(spin_contamination),
            )

        density_alpha = new_density_alpha
        density_beta = new_density_beta
        previous_energy = total_energy

    raise RuntimeError("UHF did not converge within the iteration limit.")


def build_uhf_reference_mf(mol: gto.Mole, uhf_result: UHFResult) -> tuple[scf.UHF, tuple[np.ndarray, np.ndarray]]:
    mf = scf.UHF(mol)
    mo_occ_alpha = np.zeros(mol.nao_nr())
    mo_occ_beta = np.zeros(mol.nao_nr())
    mo_occ_alpha[: uhf_result.nalpha] = 1.0
    mo_occ_beta[: uhf_result.nbeta] = 1.0
    mf.mo_coeff = (uhf_result.coefficients_alpha, uhf_result.coefficients_beta)
    mf.mo_energy = (uhf_result.orbital_energies_alpha, uhf_result.orbital_energies_beta)
    mf.mo_occ = (mo_occ_alpha, mo_occ_beta)
    mf.e_tot = uhf_result.energy
    mf.converged = True
    return mf, (mo_occ_alpha, mo_occ_beta)
