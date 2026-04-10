from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
