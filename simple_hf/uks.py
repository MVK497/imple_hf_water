from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyscf import dft, gto


@dataclass
class UKSResult:
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
    xc: str


def run_uks(
    mol: gto.Mole,
    xc: str = "b3lyp",
    max_iter: int = 100,
    e_tol: float = 1.0e-10,
    d_tol: float = 1.0e-8,
    use_diis: bool = True,
    diis_space: int = 6,
) -> UKSResult:
    mf = dft.UKS(mol)
    mf.xc = xc
    mf.max_cycle = max_iter
    mf.conv_tol = e_tol
    mf.conv_tol_grad = max(d_tol, float(np.sqrt(e_tol)))
    mf.verbose = 0

    if use_diis:
        mf.diis_space = diis_space
    else:
        mf.DIIS = None
        mf.diis = None

    history: list[float] = []

    def callback(envs: dict[str, object]) -> None:
        energy = envs.get("e_tot")
        if energy is not None:
            history.append(float(energy))

    mf.callback = callback
    total_energy = float(mf.kernel())
    if not mf.converged:
        raise RuntimeError("UKS did not converge within the iteration limit.")

    density_alpha, density_beta = mf.make_rdm1()
    orbital_energies_alpha, orbital_energies_beta = mf.mo_energy
    coefficients_alpha, coefficients_beta = mf.mo_coeff
    nalpha, nbeta = mol.nelec
    s2, _ = mf.spin_square()
    expected_s2 = 0.5 * mol.spin * (0.5 * mol.spin + 1.0)
    e_nuc = float(mol.energy_nuc())
    e_elec = total_energy - e_nuc

    return UKSResult(
        energy=total_energy,
        electronic_energy=float(e_elec),
        nuclear_repulsion=e_nuc,
        orbital_energies_alpha=np.array(orbital_energies_alpha, copy=True),
        orbital_energies_beta=np.array(orbital_energies_beta, copy=True),
        coefficients_alpha=np.array(coefficients_alpha, copy=True),
        coefficients_beta=np.array(coefficients_beta, copy=True),
        density_alpha=np.array(density_alpha, copy=True),
        density_beta=np.array(density_beta, copy=True),
        iterations=len(history),
        history=history,
        nalpha=nalpha,
        nbeta=nbeta,
        s2=float(s2),
        expected_s2=float(expected_s2),
        spin_contamination=float(s2 - expected_s2),
        xc=xc,
    )


def build_uks_reference_mf(
    mol: gto.Mole,
    uks_result: UKSResult,
) -> tuple[dft.UKS, tuple[np.ndarray, np.ndarray]]:
    mf = dft.UKS(mol)
    mf.xc = uks_result.xc
    mo_occ_alpha = np.zeros(mol.nao_nr())
    mo_occ_beta = np.zeros(mol.nao_nr())
    mo_occ_alpha[: uks_result.nalpha] = 1.0
    mo_occ_beta[: uks_result.nbeta] = 1.0
    mf.mo_coeff = (uks_result.coefficients_alpha, uks_result.coefficients_beta)
    mf.mo_energy = (uks_result.orbital_energies_alpha, uks_result.orbital_energies_beta)
    mf.mo_occ = (mo_occ_alpha, mo_occ_beta)
    mf.e_tot = uks_result.energy
    mf.converged = True
    return mf, (mo_occ_alpha, mo_occ_beta)
