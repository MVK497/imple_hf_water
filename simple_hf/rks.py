from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyscf import dft, gto


@dataclass
class RKSResult:
    energy: float
    electronic_energy: float
    nuclear_repulsion: float
    orbital_energies: np.ndarray
    coefficients: np.ndarray
    density: np.ndarray
    iterations: int
    history: list[float]
    xc: str


def _validate_closed_shell_rks(mol: gto.Mole) -> None:
    if mol.spin != 0:
        raise ValueError(
            "This teaching example implements closed-shell RKS only. "
            "Please use spin = 0 or switch to UKS."
        )
    if mol.nelectron % 2 != 0:
        raise ValueError(
            "Closed-shell RKS requires an even number of electrons after charge is applied."
        )


def run_rks(
    mol: gto.Mole,
    xc: str = "b3lyp",
    max_iter: int = 100,
    e_tol: float = 1.0e-10,
    d_tol: float = 1.0e-8,
    use_diis: bool = True,
    diis_space: int = 6,
) -> RKSResult:
    _validate_closed_shell_rks(mol)

    mf = dft.RKS(mol)
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
        raise RuntimeError("RKS did not converge within the iteration limit.")

    density = np.array(mf.make_rdm1(), copy=True)
    orbital_energies = np.array(mf.mo_energy, copy=True)
    coefficients = np.array(mf.mo_coeff, copy=True)
    e_nuc = float(mol.energy_nuc())
    e_elec = total_energy - e_nuc

    return RKSResult(
        energy=total_energy,
        electronic_energy=float(e_elec),
        nuclear_repulsion=e_nuc,
        orbital_energies=orbital_energies,
        coefficients=coefficients,
        density=density,
        iterations=len(history),
        history=history,
        xc=xc,
    )


def build_rks_reference_mf(mol: gto.Mole, rks_result: RKSResult) -> tuple[dft.RKS, np.ndarray]:
    mf = dft.RKS(mol)
    mf.xc = rks_result.xc
    nocc = mol.nelectron // 2
    mo_occ = np.zeros(mol.nao_nr())
    mo_occ[:nocc] = 2.0
    mf.mo_coeff = rks_result.coefficients
    mf.mo_energy = rks_result.orbital_energies
    mf.mo_occ = mo_occ
    mf.e_tot = rks_result.energy
    mf.converged = True
    return mf, mo_occ
