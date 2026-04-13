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


def convert_length_value(value: float, from_unit: str, to_unit: str) -> float:
    coords = np.array([[value, 0.0, 0.0]])
    converted = convert_coords(coords, from_unit, to_unit)
    return float(converted[0, 0])


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


def unit_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-14:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vector / norm


def angle_radians(coords: np.ndarray, atoms: tuple[int, int, int]) -> float:
    i, j, k = atoms
    vec_ji = coords[i] - coords[j]
    vec_jk = coords[k] - coords[j]
    u_ji = unit_vector(vec_ji)
    u_jk = unit_vector(vec_jk)
    cosine = float(np.clip(np.dot(u_ji, u_jk), -1.0, 1.0))
    return float(np.arccos(cosine))


def angle_degrees(coords: np.ndarray, atoms: tuple[int, int, int]) -> float:
    return float(np.degrees(angle_radians(coords, atoms)))


def bond_length(coords: np.ndarray, atoms: tuple[int, int]) -> float:
    i, j = atoms
    return float(np.linalg.norm(coords[j] - coords[i]))


def choose_perpendicular(vector: np.ndarray) -> np.ndarray:
    candidates = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    unit = unit_vector(vector)
    for candidate in candidates:
        trial = np.cross(unit, candidate)
        if np.linalg.norm(trial) > 1.0e-10:
            return unit_vector(trial)
    raise ValueError("Could not construct a perpendicular direction.")


def set_angle_rigid(
    coords: np.ndarray,
    atoms: tuple[int, int, int],
    target_angle_deg: float,
) -> np.ndarray:
    i, j, k = atoms
    if not 0.0 < target_angle_deg < 180.0:
        raise ValueError("Target angle must be between 0 and 180 degrees.")

    new_coords = np.array(coords, copy=True)
    center = coords[j]
    ref_vec = coords[i] - center
    move_vec = coords[k] - center

    ref_unit = unit_vector(ref_vec)
    move_norm = float(np.linalg.norm(move_vec))
    move_parallel = np.dot(move_vec, ref_unit) * ref_unit
    move_perp = move_vec - move_parallel
    if np.linalg.norm(move_perp) < 1.0e-10:
        plane_unit = choose_perpendicular(ref_unit)
    else:
        plane_unit = unit_vector(move_perp)

    angle_rad = np.deg2rad(target_angle_deg)
    new_move_vec = move_norm * (np.cos(angle_rad) * ref_unit + np.sin(angle_rad) * plane_unit)
    new_coords[k] = center + new_move_vec
    return new_coords


def set_bond_length_rigid(
    coords: np.ndarray,
    atoms: tuple[int, int],
    target_distance: float,
) -> np.ndarray:
    i, j = atoms
    if target_distance <= 0.0:
        raise ValueError("Target bond length must be positive.")

    new_coords = np.array(coords, copy=True)
    anchor = coords[i]
    move_vec = coords[j] - anchor
    move_unit = unit_vector(move_vec)
    new_coords[j] = anchor + target_distance * move_unit
    return new_coords


def wrap_angle_radians(value: float) -> float:
    return float((value + np.pi) % (2.0 * np.pi) - np.pi)


def dihedral_radians(coords: np.ndarray, atoms: tuple[int, int, int, int]) -> float:
    i, j, k, l = atoms
    p0 = coords[i]
    p1 = coords[j]
    p2 = coords[k]
    p3 = coords[l]

    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2

    b1_unit = unit_vector(b1)
    v = b0 - np.dot(b0, b1_unit) * b1_unit
    w = b2 - np.dot(b2, b1_unit) * b1_unit
    v_unit = unit_vector(v)
    w_unit = unit_vector(w)

    x = float(np.dot(v_unit, w_unit))
    y = float(np.dot(np.cross(b1_unit, v_unit), w_unit))
    return float(np.arctan2(y, x))


def dihedral_degrees(coords: np.ndarray, atoms: tuple[int, int, int, int]) -> float:
    return float(np.degrees(dihedral_radians(coords, atoms)))


def rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis_unit = unit_vector(axis)
    x, y, z = axis_unit
    cos_a = float(np.cos(angle_rad))
    sin_a = float(np.sin(angle_rad))
    one_minus_cos = 1.0 - cos_a
    return np.array(
        [
            [
                cos_a + x * x * one_minus_cos,
                x * y * one_minus_cos - z * sin_a,
                x * z * one_minus_cos + y * sin_a,
            ],
            [
                y * x * one_minus_cos + z * sin_a,
                cos_a + y * y * one_minus_cos,
                y * z * one_minus_cos - x * sin_a,
            ],
            [
                z * x * one_minus_cos - y * sin_a,
                z * y * one_minus_cos + x * sin_a,
                cos_a + z * z * one_minus_cos,
            ],
        ]
    )


def set_dihedral_rigid(
    coords: np.ndarray,
    atoms: tuple[int, int, int, int],
    target_dihedral_deg: float,
) -> np.ndarray:
    i, j, k, l = atoms
    new_coords = np.array(coords, copy=True)
    current_rad = dihedral_radians(coords, atoms)
    target_rad = float(np.deg2rad(target_dihedral_deg))
    delta_rad = wrap_angle_radians(target_rad - current_rad)

    axis_point = coords[k]
    axis_vec = coords[k] - coords[j]
    rot = rotation_matrix(axis_vec, delta_rad)
    rel = coords[l] - axis_point
    new_coords[l] = axis_point + rot @ rel
    return new_coords


def angle_gradient(coords: np.ndarray, atoms: tuple[int, int, int]) -> np.ndarray:
    i, j, k = atoms
    vec_ji = coords[i] - coords[j]
    vec_jk = coords[k] - coords[j]
    norm_ji = float(np.linalg.norm(vec_ji))
    norm_jk = float(np.linalg.norm(vec_jk))
    u_ji = unit_vector(vec_ji)
    u_jk = unit_vector(vec_jk)
    cosine = float(np.clip(np.dot(u_ji, u_jk), -1.0, 1.0))
    sine = float(np.sqrt(max(1.0 - cosine * cosine, 1.0e-16)))

    dtheta_dji = -(u_jk - cosine * u_ji) / (norm_ji * sine)
    dtheta_djk = -(u_ji - cosine * u_jk) / (norm_jk * sine)

    gradient = np.zeros_like(coords)
    gradient[i] = dtheta_dji
    gradient[k] = dtheta_djk
    gradient[j] = -(dtheta_dji + dtheta_djk)
    return gradient


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
