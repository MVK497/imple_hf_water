from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyscf import gto

from .geometry import MoleculeSpec


@dataclass
class RHFResult:
    energy: float
    electronic_energy: float
    nuclear_repulsion: float
    orbital_energies: np.ndarray
    coefficients: np.ndarray
    density: np.ndarray
    iterations: int
    history: list[float]


def build_molecule(spec: MoleculeSpec) -> gto.Mole:
    mol = gto.Mole()
    mol.atom = spec.atom
    mol.unit = spec.unit
    mol.basis = spec.basis
    mol.charge = spec.charge
    mol.spin = spec.spin
    mol.build()
    return mol


def symmetric_orthogonalization(overlap: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(overlap)
    return eigenvectors @ np.diag(eigenvalues ** -0.5) @ eigenvectors.T


def build_density(coefficients: np.ndarray, nocc: int) -> np.ndarray:
    occupied = coefficients[:, :nocc]
    return 2.0 * occupied @ occupied.T


def _validate_closed_shell_rhf(mol: gto.Mole) -> None:
    if mol.spin != 0:
        raise ValueError(
            "This teaching example implements closed-shell RHF only. "
            "Please use a system with spin = 0."
        )
    if mol.nelectron % 2 != 0:
        raise ValueError(
            "Closed-shell RHF requires an even number of electrons after charge is applied."
        )


def run_rhf(
    mol: gto.Mole,
    max_iter: int = 100,
    e_tol: float = 1.0e-10,
    d_tol: float = 1.0e-8,
) -> RHFResult:
    _validate_closed_shell_rhf(mol)

    s = mol.intor("int1e_ovlp")
    t = mol.intor("int1e_kin")
    v = mol.intor("int1e_nuc")
    eri = mol.intor("int2e")
    h_core = t + v
    e_nuc = mol.energy_nuc()

    nocc = mol.nelectron // 2
    x = symmetric_orthogonalization(s)

    fock_prime = x.T @ h_core @ x
    _, coeff_prime = np.linalg.eigh(fock_prime)
    coeff = x @ coeff_prime
    density = build_density(coeff, nocc)

    previous_energy = None
    history: list[float] = []

    for iteration in range(1, max_iter + 1):
        coulomb = np.einsum("ls,mnls->mn", density, eri, optimize=True)
        exchange = np.einsum("ls,mlns->mn", density, eri, optimize=True)
        fock = h_core + coulomb - 0.5 * exchange

        fock_prime = x.T @ fock @ x
        orbital_energies, coeff_prime = np.linalg.eigh(fock_prime)
        coeff = x @ coeff_prime
        new_density = build_density(coeff, nocc)

        e_elec = 0.5 * np.sum(new_density * (h_core + fock))
        e_total = e_elec + e_nuc
        history.append(e_total)

        density_change = np.linalg.norm(new_density - density)
        if previous_energy is not None and abs(e_total - previous_energy) < e_tol and density_change < d_tol:
            return RHFResult(
                energy=float(e_total),
                electronic_energy=float(e_elec),
                nuclear_repulsion=float(e_nuc),
                orbital_energies=orbital_energies,
                coefficients=coeff,
                density=new_density,
                iterations=iteration,
                history=history,
            )

        density = new_density
        previous_energy = e_total

    raise RuntimeError("SCF did not converge within the iteration limit.")
