from __future__ import annotations

import argparse
from pathlib import Path

from .geometry import (
    MoleculeSpec,
    default_water_spec,
    normalize_basis_name,
    parse_inline_geometry,
    read_xyz_geometry,
)
from .mp2 import MP2Result, run_mp2
from .rhf import RHFResult, build_molecule, run_rhf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimal teaching example for closed-shell RHF using PySCF integrals."
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--xyz",
        type=str,
        help="Path to an XYZ geometry file.",
    )
    input_group.add_argument(
        "--geometry",
        type=str,
        help="Inline geometry such as 'O 0 0 0; H 0 -0.757 0.586; H 0 0.757 0.586'.",
    )

    parser.add_argument(
        "--basis",
        type=str,
        default="sto-3g",
        help="Basis set name. Supported: sto-3g, 6-31G(d), 6-31G*.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="rhf",
        choices=["rhf", "mp2"],
        help="Electronic structure method to run.",
    )
    parser.add_argument("--charge", type=int, default=0, help="Total molecular charge.")
    parser.add_argument(
        "--spin",
        type=int,
        default=0,
        help="2S = N(alpha) - N(beta). This example requires spin = 0.",
    )
    parser.add_argument(
        "--unit",
        type=str,
        default="Angstrom",
        choices=["Angstrom", "Bohr"],
        help="Coordinate unit for --geometry input. XYZ input is assumed to use Angstrom.",
    )
    parser.add_argument("--max-iter", type=int, default=100, help="Maximum SCF iterations.")
    parser.add_argument("--energy-tol", type=float, default=1.0e-10, help="Energy convergence threshold.")
    parser.add_argument("--density-tol", type=float, default=1.0e-8, help="Density convergence threshold.")
    parser.add_argument(
        "--no-diis",
        action="store_true",
        help="Disable DIIS and use plain SCF iteration.",
    )
    parser.add_argument(
        "--diis-space",
        type=int,
        default=6,
        help="Number of Fock/error matrices kept in the DIIS subspace.",
    )
    parser.add_argument(
        "--show-history",
        action="store_true",
        help="Print the SCF total energy at each iteration.",
    )
    return parser


def build_spec_from_args(args: argparse.Namespace) -> MoleculeSpec:
    basis = normalize_basis_name(args.basis)

    if args.xyz:
        atom, title = read_xyz_geometry(args.xyz)
        return MoleculeSpec(
            atom=atom,
            basis=basis,
            charge=args.charge,
            spin=args.spin,
            unit="Angstrom",
            title=title or Path(args.xyz).stem,
        )

    if args.geometry:
        return MoleculeSpec(
            atom=parse_inline_geometry(args.geometry),
            basis=basis,
            charge=args.charge,
            spin=args.spin,
            unit=args.unit,
            title="Inline geometry",
        )

    default = default_water_spec(basis=basis)
    return MoleculeSpec(
        atom=default.atom,
        basis=basis,
        charge=args.charge,
        spin=args.spin,
        unit=default.unit,
        title=default.title,
    )


def print_rhf_result(
    spec: MoleculeSpec,
    result: RHFResult,
    nao: int,
    nelectron: int,
    show_history: bool,
) -> None:
    print("Restricted Hartree-Fock")
    print(f"System: {spec.title}")
    print(f"Basis set: {spec.basis}")
    print(f"Unit: {spec.unit}")
    print(f"Charge: {spec.charge}")
    print(f"Spin: {spec.spin}")
    print(f"Basis functions: {nao}")
    print(f"Electrons: {nelectron}")
    print(f"SCF iterations: {result.iterations}")
    print(f"Nuclear repulsion energy: {result.nuclear_repulsion:.12f} Eh")
    print(f"Electronic energy:        {result.electronic_energy:.12f} Eh")
    print(f"Total RHF energy:         {result.energy:.12f} Eh")
    print()
    print("Orbital energies (Eh):")
    for index, energy in enumerate(result.orbital_energies, start=1):
        print(f"  MO {index:>2d}: {energy: .12f}")

    if show_history:
        print()
        print("SCF history (total energy in Eh):")
        for iteration, energy in enumerate(result.history, start=1):
            print(f"  Iter {iteration:>2d}: {energy:.12f}")


def print_mp2_result(
    spec: MoleculeSpec,
    rhf_result: RHFResult,
    mp2_result: MP2Result,
    nao: int,
    nelectron: int,
    show_history: bool,
) -> None:
    print("Restricted MP2")
    print(f"System: {spec.title}")
    print(f"Basis set: {spec.basis}")
    print(f"Unit: {spec.unit}")
    print(f"Charge: {spec.charge}")
    print(f"Spin: {spec.spin}")
    print(f"Basis functions: {nao}")
    print(f"Electrons: {nelectron}")
    print(f"Occupied orbitals: {mp2_result.nocc}")
    print(f"Virtual orbitals: {mp2_result.nvir}")
    print(f"RHF iterations: {rhf_result.iterations}")
    print(f"Nuclear repulsion energy: {rhf_result.nuclear_repulsion:.12f} Eh")
    print(f"RHF energy:               {mp2_result.rhf_energy:.12f} Eh")
    print(f"MP2 correlation energy:   {mp2_result.mp2_correlation_energy:.12f} Eh")
    print(f"Total MP2 energy:         {mp2_result.total_energy:.12f} Eh")
    print()
    print("Orbital energies (Eh):")
    for index, energy in enumerate(mp2_result.orbital_energies, start=1):
        print(f"  MO {index:>2d}: {energy: .12f}")

    if show_history:
        print()
        print("RHF SCF history (total energy in Eh):")
        for iteration, energy in enumerate(rhf_result.history, start=1):
            print(f"  Iter {iteration:>2d}: {energy:.12f}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.diis_space < 2:
        parser.error("--diis-space must be at least 2.")
    spec = build_spec_from_args(args)
    mol = build_molecule(spec)
    result = run_rhf(
        mol,
        max_iter=args.max_iter,
        e_tol=args.energy_tol,
        d_tol=args.density_tol,
        use_diis=not args.no_diis,
        diis_space=args.diis_space,
    )
    if args.method == "rhf":
        print_rhf_result(spec, result, mol.nao_nr(), mol.nelectron, args.show_history)
        return

    mp2_result = run_mp2(mol, result)
    print_mp2_result(spec, result, mp2_result, mol.nao_nr(), mol.nelectron, args.show_history)
