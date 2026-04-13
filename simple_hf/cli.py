from __future__ import annotations

import argparse
from pathlib import Path

from .ccsd import CCSDResult, run_ccsd
from .geometry import (
    MoleculeSpec,
    default_water_spec,
    normalize_basis_name,
    parse_inline_geometry,
    read_xyz_geometry,
)
from .mp2 import MP2Result, run_mp2
from .optimize import OptimizationResult, optimize_geometry
from .rhf import RHFResult, build_molecule, run_rhf
from .scan import (
    ScanResult,
    coordinate_arity,
    relaxed_scan,
    rigid_scan,
    write_scan_csv,
)
from .ump2 import UMP2Result, run_ump2
from .uhf import UHFResult, run_uhf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimal teaching example for RHF, UHF, MP2, UMP2, CCSD, geometry optimization, and angle scans."
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
        choices=["rhf", "uhf", "mp2", "ump2", "ccsd"],
        help="Electronic structure method to run.",
    )
    parser.add_argument("--charge", type=int, default=0, help="Total molecular charge.")
    parser.add_argument(
        "--spin",
        type=int,
        default=0,
        help="2S = N(alpha) - N(beta). RHF/MP2 require spin = 0; UHF supports open-shell cases.",
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
        "--optimize",
        action="store_true",
        help="Run a geometry optimization. Currently supported for RHF and UHF.",
    )
    parser.add_argument(
        "--scan",
        type=str,
        choices=["rigid", "relaxed"],
        help="Run an internal-coordinate scan. 'rigid' keeps other coordinates fixed; 'relaxed' optimizes all others.",
    )
    parser.add_argument(
        "--scan-coordinate",
        type=str,
        default="angle",
        choices=["bond", "angle", "dihedral"],
        help="Internal coordinate type for scanning.",
    )
    parser.add_argument(
        "--scan-atoms",
        type=str,
        help="1-based atom indices for the scan coordinate. bond: '1,2'; angle: '2,1,3'; dihedral: '1,2,3,4'.",
    )
    parser.add_argument("--scan-start", type=float, help="Scan start value. Degrees for angle/dihedral, geometry unit for bond.")
    parser.add_argument("--scan-stop", type=float, help="Scan stop value. Degrees for angle/dihedral, geometry unit for bond.")
    parser.add_argument("--scan-points", type=int, default=7, help="Number of scan points.")
    parser.add_argument(
        "--scan-output",
        type=str,
        help="Optional CSV file path for scan results.",
    )
    parser.add_argument(
        "--constraint-k",
        type=float,
        default=50.0,
        help="Harmonic penalty strength for relaxed angle scans in Eh/rad^2.",
    )
    parser.add_argument(
        "--opt-max-steps",
        type=int,
        default=30,
        help="Maximum geometry optimization steps.",
    )
    parser.add_argument(
        "--grad-tol",
        type=float,
        default=1.0e-4,
        help="Maximum Cartesian gradient threshold in Eh/Bohr.",
    )
    parser.add_argument(
        "--opt-energy-tol",
        type=float,
        default=1.0e-8,
        help="Optimization energy convergence threshold in Eh.",
    )
    parser.add_argument(
        "--max-step-size",
        type=float,
        default=0.2,
        help="Maximum geometry step size in Bohr.",
    )
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


def parse_scan_atoms(text: str) -> tuple[int, ...]:
    try:
        values = [int(token.strip()) for token in text.split(",")]
    except ValueError as exc:
        raise ValueError("--scan-atoms must look like '1,2' or '2,1,3' or '1,2,3,4'.") from exc
    if min(values) < 1:
        raise ValueError("--scan-atoms uses 1-based indices, so all values must be >= 1.")
    return tuple(value - 1 for value in values)


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


def print_ccsd_result(
    spec: MoleculeSpec,
    rhf_result: RHFResult,
    ccsd_result: CCSDResult,
    nao: int,
    nelectron: int,
    show_history: bool,
) -> None:
    print("Restricted CCSD")
    print(f"System: {spec.title}")
    print(f"Basis set: {spec.basis}")
    print(f"Unit: {spec.unit}")
    print(f"Charge: {spec.charge}")
    print(f"Spin: {spec.spin}")
    print(f"Basis functions: {nao}")
    print(f"Electrons: {nelectron}")
    print(f"Occupied orbitals: {ccsd_result.nocc}")
    print(f"Virtual orbitals: {ccsd_result.nvir}")
    print(f"RHF iterations: {rhf_result.iterations}")
    print(f"CCSD converged:           {ccsd_result.converged}")
    print(f"Nuclear repulsion energy: {rhf_result.nuclear_repulsion:.12f} Eh")
    print(f"RHF energy:               {ccsd_result.rhf_energy:.12f} Eh")
    print(f"CCSD correlation energy:  {ccsd_result.ccsd_correlation_energy:.12f} Eh")
    print(f"Total CCSD energy:        {ccsd_result.total_energy:.12f} Eh")
    print(f"||t1||:                   {ccsd_result.t1_norm:.12f}")
    print(f"||t2||:                   {ccsd_result.t2_norm:.12f}")
    print()
    print("Orbital energies (Eh):")
    for index, energy in enumerate(ccsd_result.orbital_energies, start=1):
        print(f"  MO {index:>2d}: {energy: .12f}")

    if show_history:
        print()
        print("RHF SCF history (total energy in Eh):")
        for iteration, energy in enumerate(rhf_result.history, start=1):
            print(f"  Iter {iteration:>2d}: {energy:.12f}")


def print_uhf_result(
    spec: MoleculeSpec,
    result: UHFResult,
    nao: int,
    nelectron: int,
    show_history: bool,
) -> None:
    print("Unrestricted Hartree-Fock")
    print(f"System: {spec.title}")
    print(f"Basis set: {spec.basis}")
    print(f"Unit: {spec.unit}")
    print(f"Charge: {spec.charge}")
    print(f"Spin: {spec.spin}")
    print(f"Basis functions: {nao}")
    print(f"Electrons: {nelectron}")
    print(f"Alpha electrons: {result.nalpha}")
    print(f"Beta electrons:  {result.nbeta}")
    print(f"SCF iterations: {result.iterations}")
    print(f"Nuclear repulsion energy: {result.nuclear_repulsion:.12f} Eh")
    print(f"Electronic energy:        {result.electronic_energy:.12f} Eh")
    print(f"Total UHF energy:         {result.energy:.12f} Eh")
    print(f"<S^2>:                    {result.s2:.12f}")
    print(f"Expected <S^2>:           {result.expected_s2:.12f}")
    print(f"Spin contamination:       {result.spin_contamination:.12f}")
    print()
    print("Alpha orbital energies (Eh):")
    for index, energy in enumerate(result.orbital_energies_alpha, start=1):
        print(f"  aMO {index:>2d}: {energy: .12f}")
    print()
    print("Beta orbital energies (Eh):")
    for index, energy in enumerate(result.orbital_energies_beta, start=1):
        print(f"  bMO {index:>2d}: {energy: .12f}")

    if show_history:
        print()
        print("UHF SCF history (total energy in Eh):")
        for iteration, energy in enumerate(result.history, start=1):
            print(f"  Iter {iteration:>2d}: {energy:.12f}")


def print_ump2_result(
    spec: MoleculeSpec,
    uhf_result: UHFResult,
    ump2_result: UMP2Result,
    nao: int,
    nelectron: int,
    show_history: bool,
) -> None:
    print("Unrestricted MP2")
    print(f"System: {spec.title}")
    print(f"Basis set: {spec.basis}")
    print(f"Unit: {spec.unit}")
    print(f"Charge: {spec.charge}")
    print(f"Spin: {spec.spin}")
    print(f"Basis functions: {nao}")
    print(f"Electrons: {nelectron}")
    print(f"Alpha electrons: {ump2_result.nocc_alpha}")
    print(f"Beta electrons:  {ump2_result.nocc_beta}")
    print(f"Alpha virtual orbitals: {ump2_result.nvir_alpha}")
    print(f"Beta virtual orbitals:  {ump2_result.nvir_beta}")
    print(f"UHF iterations: {uhf_result.iterations}")
    print(f"Nuclear repulsion energy: {uhf_result.nuclear_repulsion:.12f} Eh")
    print(f"UHF energy:               {ump2_result.uhf_energy:.12f} Eh")
    print(f"Reference <S^2>:          {uhf_result.s2:.12f}")
    print(f"Expected <S^2>:           {uhf_result.expected_s2:.12f}")
    print(f"Spin contamination:       {uhf_result.spin_contamination:.12f}")
    print(f"UMP2 aa correlation:      {ump2_result.correlation_energy_aa:.12f} Eh")
    print(f"UMP2 ab correlation:      {ump2_result.correlation_energy_ab:.12f} Eh")
    print(f"UMP2 bb correlation:      {ump2_result.correlation_energy_bb:.12f} Eh")
    print(f"UMP2 correlation energy:  {ump2_result.ump2_correlation_energy:.12f} Eh")
    print(f"Total UMP2 energy:        {ump2_result.total_energy:.12f} Eh")
    print()
    print("Alpha orbital energies (Eh):")
    for index, energy in enumerate(ump2_result.orbital_energies_alpha, start=1):
        print(f"  aMO {index:>2d}: {energy: .12f}")
    print()
    print("Beta orbital energies (Eh):")
    for index, energy in enumerate(ump2_result.orbital_energies_beta, start=1):
        print(f"  bMO {index:>2d}: {energy: .12f}")

    if show_history:
        print()
        print("UHF SCF history (total energy in Eh):")
        for iteration, energy in enumerate(uhf_result.history, start=1):
            print(f"  Iter {iteration:>2d}: {energy:.12f}")


def print_optimization_result(
    spec: MoleculeSpec,
    result: OptimizationResult,
    show_history: bool,
) -> None:
    print(f"{result.method.upper()} Geometry Optimization")
    print(f"System: {spec.title}")
    print(f"Basis set: {spec.basis}")
    print(f"Unit: {spec.unit}")
    print(f"Charge: {spec.charge}")
    print(f"Spin: {spec.spin}")
    print(f"Converged: {result.converged}")
    print(f"Optimization steps: {result.iterations}")
    print(f"Initial energy: {result.initial_energy:.12f} Eh")
    print(f"Final energy:   {result.final_energy:.12f} Eh")
    print(f"Final max |grad|: {float(abs(result.final_gradient).max()):.12f} Eh/Bohr")
    if isinstance(result.final_result, UHFResult):
        print(f"Final <S^2>: {result.final_result.s2:.12f}")
        print(f"Final spin contamination: {result.final_result.spin_contamination:.12f}")
    print()
    print("Optimized geometry:")
    print(result.optimized_spec.atom)

    if show_history and result.history:
        print()
        print("Optimization history:")
        for step in result.history:
            print(
                f"  Step {step.step:>2d}: E = {step.energy:.12f}  "
                f"max|g| = {step.max_gradient:.6e}  "
                f"rms|g| = {step.rms_gradient:.6e}  "
                f"|dx| = {step.step_length:.6e}"
            )


def print_scan_result(scan_result: ScanResult) -> None:
    print(f"{scan_result.mode.capitalize()} {scan_result.coordinate_type.capitalize()} Scan")
    print(f"Method: {scan_result.method}")
    atoms_label = "-".join(str(atom + 1) for atom in scan_result.atoms)
    print(f"Coordinate definition: {atoms_label}")
    print(f"Coordinate unit: {scan_result.value_unit}")
    print()
    print("Scan points:")
    for point in scan_result.points:
        status = "ok" if point.converged else "not converged"
        print(
            f"  Pt {point.index:>2d}: "
            f"target = {point.target_value:10.4f} {scan_result.value_unit}  "
            f"actual = {point.actual_value:10.4f} {scan_result.value_unit}  "
            f"E = {point.energy:.12f} Eh  "
            f"{status}"
        )
    print()
    print(
        f"Best point: target = {scan_result.best_point.target_value:.6f} {scan_result.value_unit}, "
        f"actual = {scan_result.best_point.actual_value:.6f} {scan_result.value_unit}, "
        f"E = {scan_result.best_point.energy:.12f} Eh"
    )
    print()
    print("Best geometry:")
    print(scan_result.best_point.spec.atom)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.diis_space < 2:
        parser.error("--diis-space must be at least 2.")
    if args.scan_points is not None and args.scan_points < 2:
        parser.error("--scan-points must be at least 2.")
    if args.optimize and args.scan:
        parser.error("--optimize and --scan cannot be used together.")
    if args.optimize and args.method not in {"rhf", "uhf"}:
        parser.error("--optimize currently supports only --method rhf or --method uhf.")
    spec = build_spec_from_args(args)
    if args.scan:
        if args.scan_atoms is None or args.scan_start is None or args.scan_stop is None:
            parser.error("--scan requires --scan-atoms, --scan-start, and --scan-stop.")
        scan_atoms = parse_scan_atoms(args.scan_atoms)
        expected_arity = coordinate_arity(args.scan_coordinate)
        if len(scan_atoms) != expected_arity:
            parser.error(f"--scan-coordinate {args.scan_coordinate} requires exactly {expected_arity} atom indices.")
        natom = len(spec.atom.splitlines())
        if max(scan_atoms) >= natom:
            parser.error(f"--scan-atoms refers to atom index > {natom}.")
        if args.scan == "rigid":
            scan_result = rigid_scan(
                spec,
                method=args.method,
                coordinate_type=args.scan_coordinate,
                atoms=scan_atoms,
                start_value=args.scan_start,
                stop_value=args.scan_stop,
                num_points=args.scan_points,
                max_iter=args.max_iter,
                e_tol=args.energy_tol,
                d_tol=args.density_tol,
                use_diis=not args.no_diis,
                diis_space=args.diis_space,
            )
        else:
            if args.method not in {"rhf", "uhf"}:
                parser.error("--scan relaxed currently supports only --method rhf or --method uhf.")
            scan_result = relaxed_scan(
                spec,
                method=args.method,
                coordinate_type=args.scan_coordinate,
                atoms=scan_atoms,
                start_value=args.scan_start,
                stop_value=args.scan_stop,
                num_points=args.scan_points,
                penalty_k=args.constraint_k,
                max_opt_steps=args.opt_max_steps,
                grad_tol=args.grad_tol,
                opt_energy_tol=args.opt_energy_tol,
                max_step_size=args.max_step_size,
                max_iter=args.max_iter,
                scf_e_tol=args.energy_tol,
                scf_d_tol=args.density_tol,
                use_diis=not args.no_diis,
                diis_space=args.diis_space,
            )
        if args.scan_output:
            write_scan_csv(scan_result, args.scan_output)
        print_scan_result(scan_result)
        return

    if args.optimize:
        opt_result = optimize_geometry(
            spec,
            method=args.method,
            max_opt_steps=args.opt_max_steps,
            grad_tol=args.grad_tol,
            energy_tol=args.opt_energy_tol,
            max_step_size=args.max_step_size,
            max_iter=args.max_iter,
            scf_e_tol=args.energy_tol,
            scf_d_tol=args.density_tol,
            use_diis=not args.no_diis,
            diis_space=args.diis_space,
        )
        print_optimization_result(spec, opt_result, args.show_history)
        return

    mol = build_molecule(spec)
    rhf_result = run_rhf(
        mol,
        max_iter=args.max_iter,
        e_tol=args.energy_tol,
        d_tol=args.density_tol,
        use_diis=not args.no_diis,
        diis_space=args.diis_space,
    ) if args.method in {"rhf", "mp2", "ccsd"} else None
    if args.method == "rhf":
        assert rhf_result is not None
        print_rhf_result(spec, rhf_result, mol.nao_nr(), mol.nelectron, args.show_history)
        return

    if args.method == "mp2":
        assert rhf_result is not None
        mp2_result = run_mp2(mol, rhf_result)
        print_mp2_result(spec, rhf_result, mp2_result, mol.nao_nr(), mol.nelectron, args.show_history)
        return

    if args.method == "ccsd":
        assert rhf_result is not None
        ccsd_result = run_ccsd(mol, rhf_result)
        print_ccsd_result(spec, rhf_result, ccsd_result, mol.nao_nr(), mol.nelectron, args.show_history)
        return

    uhf_result = run_uhf(
        mol,
        max_iter=args.max_iter,
        e_tol=args.energy_tol,
        d_tol=args.density_tol,
        use_diis=not args.no_diis,
        diis_space=args.diis_space,
    )
    if args.method == "uhf":
        print_uhf_result(spec, uhf_result, mol.nao_nr(), mol.nelectron, args.show_history)
        return

    ump2_result = run_ump2(mol, uhf_result)
    print_ump2_result(spec, uhf_result, ump2_result, mol.nao_nr(), mol.nelectron, args.show_history)
