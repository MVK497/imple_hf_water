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


@dataclass
class DIISHelper:
    max_vectors: int = 6

    def __post_init__(self) -> None:
        self.fock_matrices: list[np.ndarray] = []
        self.error_matrices: list[np.ndarray] = []

    def push(self, fock: np.ndarray, error: np.ndarray) -> None:
        self.fock_matrices.append(fock.copy())
        self.error_matrices.append(error.copy())
        if len(self.fock_matrices) > self.max_vectors:
            self.fock_matrices.pop(0)
            self.error_matrices.pop(0)

    def extrapolate(self) -> np.ndarray:
        count = len(self.fock_matrices)
        if count < 2:
            return self.fock_matrices[-1]

        b_matrix = np.empty((count + 1, count + 1))
        b_matrix[:-1, :-1] = np.array(
            [
                [np.vdot(err_i, err_j).real for err_j in self.error_matrices]
                for err_i in self.error_matrices
            ]
        )
        b_matrix[-1, :-1] = -1.0
        b_matrix[:-1, -1] = -1.0
        b_matrix[-1, -1] = 0.0

        rhs = np.zeros(count + 1)
        rhs[-1] = -1.0

        try:
            coefficients = np.linalg.solve(b_matrix, rhs)[:-1]
        except np.linalg.LinAlgError:
            return self.fock_matrices[-1]

        return sum(coeff * fock for coeff, fock in zip(coefficients, self.fock_matrices, strict=True))


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


def build_fock(h_core: np.ndarray, eri: np.ndarray, density: np.ndarray) -> np.ndarray:
    coulomb = np.einsum("ls,mnls->mn", density, eri, optimize=True)
    exchange = np.einsum("ls,mlns->mn", density, eri, optimize=True)
    return h_core + coulomb - 0.5 * exchange


def compute_diis_error(fock: np.ndarray, density: np.ndarray, overlap: np.ndarray, x: np.ndarray) -> np.ndarray:
    commutator = fock @ density @ overlap - overlap @ density @ fock
    return x.T @ commutator @ x


def diagonalize_fock(fock: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fock_prime = x.T @ fock @ x
    orbital_energies, coeff_prime = np.linalg.eigh(fock_prime)
    return orbital_energies, x @ coeff_prime


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
    use_diis: bool = True,
    diis_space: int = 6,
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
    diis_helper = DIISHelper(max_vectors=diis_space)

    _, coeff = diagonalize_fock(h_core, x)
    density = build_density(coeff, nocc)

    previous_energy = None
    history: list[float] = []

    for iteration in range(1, max_iter + 1):
        fock = build_fock(h_core, eri, density)
        if use_diis:
            diis_error = compute_diis_error(fock, density, s, x)
            diis_helper.push(fock, diis_error)
            fock_to_diagonalize = diis_helper.extrapolate()
        else:
            fock_to_diagonalize = fock

        orbital_energies, coeff = diagonalize_fock(fock_to_diagonalize, x)
        new_density = build_density(coeff, nocc)
        new_fock = build_fock(h_core, eri, new_density)

        e_elec = 0.5 * np.sum(new_density * (h_core + new_fock))
        e_total = e_elec + e_nuc
        history.append(e_total)

        density_change = np.linalg.norm(new_density - density)
        if previous_energy is not None and abs(e_total - previous_energy) < e_tol and density_change < d_tol:
            orbital_energies, coeff = diagonalize_fock(new_fock, x)
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
