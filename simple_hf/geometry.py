from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


BOHR_TO_ANGSTROM = 0.529177210903
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM


@dataclass(frozen=True)
class MoleculeSpec:
    atom: str
    basis: str = "sto-3g"
    charge: int = 0
    spin: int = 0
    unit: str = "Angstrom"
    title: str = "Molecule"


SUPPORTED_BASIS_ALIASES = {
    "sto-3g": "sto-3g",
    "6-31g(d)": "6-31g(d)",
    "6-31g*": "6-31g(d)",
}


def normalize_basis_name(name: str) -> str:
    normalized = name.strip().lower()
    if normalized not in SUPPORTED_BASIS_ALIASES:
        supported = ", ".join(sorted(SUPPORTED_BASIS_ALIASES))
        raise ValueError(
            f"Unsupported basis '{name}'. Supported options are: {supported}."
        )
    return SUPPORTED_BASIS_ALIASES[normalized]


def convert_coords(coords: np.ndarray, from_unit: str, to_unit: str) -> np.ndarray:
    if from_unit == to_unit:
        return np.array(coords, copy=True)
    if from_unit == "Angstrom" and to_unit == "Bohr":
        return np.array(coords, copy=True) * ANGSTROM_TO_BOHR
    if from_unit == "Bohr" and to_unit == "Angstrom":
        return np.array(coords, copy=True) * BOHR_TO_ANGSTROM
    raise ValueError(f"Unsupported unit conversion from {from_unit} to {to_unit}.")


def _normalize_geometry_lines(lines: list[str]) -> str:
    cleaned = [line.strip() for line in lines if line.strip()]
    if not cleaned:
        raise ValueError("No atomic coordinates were provided.")

    for line in cleaned:
        tokens = line.split()
        if len(tokens) < 4:
            raise ValueError(
                f"Invalid geometry line: '{line}'. Expected format like 'O 0.0 0.0 0.0'."
            )
    return "\n".join(cleaned)


def parse_inline_geometry(text: str) -> str:
    raw_lines = []
    for chunk in text.replace(";", "\n").splitlines():
        raw_lines.append(chunk)
    return _normalize_geometry_lines(raw_lines)


def parse_atom_string(atom: str) -> tuple[list[str], np.ndarray]:
    lines = _normalize_geometry_lines(atom.splitlines()).splitlines()
    symbols: list[str] = []
    coords = []
    for line in lines:
        symbol, x, y, z = line.split()[:4]
        symbols.append(symbol)
        coords.append([float(x), float(y), float(z)])
    return symbols, np.array(coords, dtype=float)


def format_atom_string(symbols: list[str], coords: np.ndarray) -> str:
    lines = [
        f"{symbol:<2s} {xyz[0]: .10f} {xyz[1]: .10f} {xyz[2]: .10f}"
        for symbol, xyz in zip(symbols, coords, strict=True)
    ]
    return "\n".join(lines)


def read_xyz_geometry(path: str) -> tuple[str, str]:
    xyz_path = Path(path)
    content = xyz_path.read_text(encoding="utf-8").splitlines()
    if len(content) < 3:
        raise ValueError("XYZ file is too short.")

    try:
        natom = int(content[0].strip())
    except ValueError as exc:
        raise ValueError("First line of XYZ file must be the atom count.") from exc

    title = content[1].strip() or xyz_path.stem
    coord_lines = content[2 : 2 + natom]
    if len(coord_lines) != natom:
        raise ValueError("XYZ file does not contain the declared number of atoms.")

    return _normalize_geometry_lines(coord_lines), title


def default_water_spec(basis: str = "sto-3g") -> MoleculeSpec:
    atom = """
    O  0.000000   0.000000   0.000000
    H  0.000000  -0.757160   0.586260
    H  0.000000   0.757160   0.586260
    """
    return MoleculeSpec(
        atom=_normalize_geometry_lines(atom.splitlines()),
        basis=normalize_basis_name(basis),
        charge=0,
        spin=0,
        unit="Angstrom",
        title="H2O",
    )
