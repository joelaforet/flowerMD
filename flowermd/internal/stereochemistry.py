"""Tetrahedral stereochemistry audit and a native HOOMD restraint that keeps it.

Soft DPD repulsion lets atoms pass through each other, so a tetrahedral
center can invert during initialization and a downstream minimizer will not
put it back. This module records the handedness of every potential
stereocenter before dynamics, checks it afterwards, and optionally adds a
restraint that keeps it.

Handedness of a center with topology-ordered substituents a, b, c, d is the
sign of ``det(a-d, b-d, c-d)``; the audit divides by the three edge lengths
so the number is scale free. A reflection flips the sign, proper rotations
and translations keep it. The reference is captured once from the input
geometry and never reassigned from output coordinates.

The restraint is a native `hoomd.md.dihedral.Periodic` on a, b, c, d with
``U = k/2 [1 - cos(phi - phi0)]`` (n=1, d=-1), where phi0 is the signed
dihedral of the reference geometry in HOOMD's own convention. HOOMD's
harmonic improper is mirror-even and cannot select a handedness. The
restraint is a finite bias, so the final audit stays mandatory.

Everything here works in Angstrom, matching the all-atom DPD frame.
Requires RDKit for reference capture (imported lazily).
"""

import hashlib
import json
from dataclasses import asdict, dataclass, replace

import numpy as np

STEREO_TYPE_PREFIX = "phantomwalk_stereo_"


@dataclass(frozen=True)
class StereoCenter:
    """One tetrahedral center: its atom, four substituents, and reference."""

    center_index: int
    neighbors: tuple
    expected_sign: int
    reference_volume: float
    phi0: float
    stereo_label: str = "unspecified"
    type_name: str = ""


@dataclass(frozen=True)
class StereoReference:
    """All tetrahedral centers of a system with their reference handedness."""

    n_atoms: int
    centers: tuple

    def to_dict(self):
        """Plain-dictionary form for records."""
        return asdict(self)

    @property
    def sha256(self):
        """Digest of the reference, for provenance."""
        text = json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)
        return hashlib.sha256(text.encode()).hexdigest()

    @property
    def type_names(self):
        """Dihedral type names used by the restraint, in first-seen order."""
        return tuple(dict.fromkeys(c.type_name for c in self.centers))


class StereoIntegrityError(RuntimeError):
    """Raised when coordinates do not hold the intended stereochemistry.

    The audit report is available as ``.report``.
    """

    def __init__(self, report):
        self.report = report
        super().__init__(
            "Tetrahedral stereochemistry audit failed: "
            f"{report.get('inverted_count', 0)} inverted, "
            f"{report.get('near_planar_count', 0)} near-planar, "
            f"{report.get('nonfinite_count', 0)} nonfinite of "
            f"{report.get('n_centers', 0)} centers. The coordinates must not "
            "be taken as the intended chemistry. See .report for details."
        )


def _minimum_image(vectors, box_lengths):
    if box_lengths is None:
        return np.asarray(vectors, dtype=float)
    box = np.asarray(box_lengths, dtype=float)
    return vectors - box * np.rint(vectors / box)


def normalized_volume(points):
    """Signed volume of the substituent tetrahedron, scaled by its edges."""
    points = np.asarray(points, dtype=float)
    if points.shape != (4, 3) or not np.isfinite(points).all():
        return float("nan")
    edges = points[:3] - points[3]
    denominator = float(np.prod(np.linalg.norm(edges, axis=1)))
    if denominator <= np.finfo(float).tiny:
        return 0.0
    return float(np.linalg.det(edges) / denominator)


def signed_dihedral(points):
    """Dihedral of four points in HOOMD's Periodic convention, on [-pi, pi]."""
    points = np.asarray(points, dtype=float)
    u = points[0] - points[1]
    v = points[2] - points[1]
    w = points[3] - points[2]
    aa, bb = np.cross(u, -v), np.cross(w, -v)
    x = float(np.dot(aa, bb))
    y = float(np.linalg.norm(v) * np.dot(aa, w))
    if x * x + y * y <= np.finfo(float).tiny:
        raise ValueError("Undefined dihedral: collinear or coincident points.")
    return float(np.arctan2(y, x))


def _assign_types(reference):
    phases = sorted(
        {round(c.phi0 % (2 * np.pi), 12) for c in reference.centers}
    )
    names = {p: f"{STEREO_TYPE_PREFIX}{i}" for i, p in enumerate(phases)}
    centers = tuple(
        replace(
            c,
            phi0=round(c.phi0 % (2 * np.pi), 12),
            type_name=names[round(c.phi0 % (2 * np.pi), 12)],
        )
        for c in reference.centers
    )
    return replace(reference, centers=centers)


def capture_stereochemistry(compound, planar_tolerance=0.05):
    """Record the handedness of every potential tetrahedral stereocenter.

    RDKit's `FindPotentialStereo` picks the centers; the reference sign and
    phase come from the compound's current geometry. Where the molecule
    carries a specified CIP label, it must agree with the geometry.

    Parameters
    ----------
    compound : mbuild.Compound, required
        All-atom compound with explicit hydrogens (nm coordinates).
    planar_tolerance : float, default 0.05
        Centers whose normalized volume is below this are rejected as
        near-planar references.

    Returns
    -------
    StereoReference

    """
    from rdkit import Chem

    from flowermd.internal.all_atom_parameters import (
        compound_to_rdkit,
        molecular_compounds,
    )

    if not 0 < planar_tolerance < 1:
        raise ValueError("planar_tolerance must be between 0 and 1.")
    xyz = np.asarray(compound.xyz, dtype=float) * 10.0
    particles = list(compound.particles())
    global_index = {p: i for i, p in enumerate(particles)}
    centers = []
    for chain in molecular_compounds(compound):
        local = list(chain.particles())
        indices = [global_index[p] for p in local]
        mol = compound_to_rdkit(chain)
        Chem.AssignStereochemistry(mol, cleanIt=False, force=True)
        candidates = [
            int(info.centeredOn)
            for info in Chem.FindPotentialStereo(
                mol, cleanIt=False, flagPossible=True
            )
            if info.type == Chem.StereoType.Atom_Tetrahedral
        ]
        if not candidates:
            continue
        # Perceive the geometry's own labels to check any specified ones.
        observed = Chem.Mol(mol)
        observed.RemoveAllConformers()
        conformer = Chem.Conformer(len(indices))
        conformer.Set3D(True)
        for i, point in enumerate(xyz[indices]):
            conformer.SetAtomPosition(i, [float(x) for x in point])
        observed.AddConformer(conformer, assignId=True)
        Chem.AssignAtomChiralTagsFromStructure(
            observed, replaceExistingTags=True
        )
        Chem.AssignStereochemistry(observed, cleanIt=True, force=True)
        for center in candidates:
            atom = mol.GetAtomWithIdx(center)
            neighbors = tuple(sorted(n.GetIdx() for n in atom.GetNeighbors()))
            if len(neighbors) != 4:
                raise ValueError(
                    "Tetrahedral audit needs four explicit neighbors on atom "
                    f"{indices[center]}; add hydrogens."
                )
            label = (
                atom.GetProp("_CIPCode")
                if atom.HasProp("_CIPCode")
                else "unspecified"
            )
            seen = observed.GetAtomWithIdx(center)
            seen_label = (
                seen.GetProp("_CIPCode")
                if seen.HasProp("_CIPCode")
                else "unspecified"
            )
            if label != "unspecified" and label != seen_label:
                raise StereoIntegrityError(
                    {
                        "stage": "capture",
                        "center_index": indices[center],
                        "reason": "specified label disagrees with geometry",
                        "intended_label": label,
                        "observed_label": seen_label,
                        "n_centers": 1,
                        "inverted_count": 1,
                    }
                )
            points = xyz[[indices[n] for n in neighbors]] - xyz[indices[center]]
            volume = normalized_volume(points)
            if not np.isfinite(volume) or abs(volume) < planar_tolerance:
                raise StereoIntegrityError(
                    {
                        "stage": "capture",
                        "center_index": indices[center],
                        "reason": "nonfinite or near-planar reference",
                        "normalized_volume": volume,
                        "n_centers": 1,
                        "near_planar_count": 1,
                    }
                )
            centers.append(
                StereoCenter(
                    center_index=indices[center],
                    neighbors=tuple(indices[n] for n in neighbors),
                    expected_sign=1 if volume > 0 else -1,
                    reference_volume=volume,
                    phi0=signed_dihedral(points) % (2 * np.pi),
                    stereo_label=label,
                )
            )
    return _assign_types(StereoReference(len(xyz), tuple(centers)))


def audit_stereochemistry(
    reference, positions, box_lengths=None, planar_tolerance=0.05
):
    """Compare current positions (Angstrom) against a `StereoReference`.

    Returns a dictionary with ``passed``, counts, and one entry per center
    with status ``preserved``, ``inverted``, ``near_planar`` or ``nonfinite``.
    `box_lengths` enables minimum-image vectors for wrapped coordinates.
    """
    xyz = np.asarray(positions, dtype=float)
    if xyz.shape != (reference.n_atoms, 3):
        raise ValueError("Coordinate and reference atom counts differ.")
    details = []
    inverted = planar = nonfinite = 0
    for center in reference.centers:
        points = xyz[list(center.neighbors)] - xyz[center.center_index]
        volume = None
        if not np.isfinite(points).all():
            status = "nonfinite"
            nonfinite += 1
        else:
            volume = normalized_volume(_minimum_image(points, box_lengths))
            if not np.isfinite(volume):
                status, volume = "nonfinite", None
                nonfinite += 1
            elif abs(volume) < planar_tolerance:
                status = "near_planar"
                planar += 1
            elif volume * center.expected_sign < 0:
                status = "inverted"
                inverted += 1
            else:
                status = "preserved"
        details.append(
            {
                "center_index": center.center_index,
                "status": status,
                "normalized_volume": volume,
                "expected_sign": center.expected_sign,
            }
        )
    return {
        "passed": not (inverted or planar or nonfinite),
        "n_centers": len(reference.centers),
        "inverted_count": inverted,
        "near_planar_count": planar,
        "nonfinite_count": nonfinite,
        "planar_tolerance": planar_tolerance,
        "reference_sha256": reference.sha256,
        "centers": details,
    }


def require_stereochemistry(
    reference, positions, box_lengths=None, planar_tolerance=0.05
):
    """Audit and raise `StereoIntegrityError` if any center failed."""
    report = audit_stereochemistry(
        reference, positions, box_lengths, planar_tolerance
    )
    if not report["passed"]:
        raise StereoIntegrityError(report)
    return report


def append_stereochemistry_dihedrals(frame, reference):
    """Add one dihedral group per center to a `gsd.hoomd.Frame`.

    The a-d pair of every center is already a 1-3 pair (both bonded to the
    center), so the new dihedrals do not change neighbor-list exclusions
    when ``"angle"`` is excluded. Returns the number of groups added.
    """
    if not reference.centers:
        return 0
    types = list(frame.dihedrals.types or [])
    if any(t.startswith(STEREO_TYPE_PREFIX) for t in types):
        raise ValueError("Stereochemistry dihedrals were already appended.")
    n_old = int(frame.dihedrals.N or 0)
    groups = (
        np.asarray(frame.dihedrals.group, dtype=np.uint32).reshape(-1, 4)
        if n_old
        else np.empty((0, 4), dtype=np.uint32)
    )
    ids = (
        np.asarray(frame.dihedrals.typeid, dtype=np.uint32)
        if n_old
        else np.empty(0, dtype=np.uint32)
    )
    types.extend(reference.type_names)
    index = {name: i for i, name in enumerate(types)}
    frame.dihedrals.N = n_old + len(reference.centers)
    frame.dihedrals.types = types
    frame.dihedrals.group = np.concatenate(
        [
            groups,
            np.asarray(
                [c.neighbors for c in reference.centers], dtype=np.uint32
            ),
        ]
    )
    frame.dihedrals.typeid = np.concatenate(
        [
            ids,
            np.asarray(
                [index[c.type_name] for c in reference.centers], dtype=np.uint32
            ),
        ]
    )
    return len(reference.centers)


def build_stereochemistry_force(reference, k, dihedral_types=()):
    """Native `hoomd.md.dihedral.Periodic` restraint for every center.

    Other dihedral types in the frame get ``k=0`` entries so the force
    covers every type in the state, as HOOMD requires.
    """
    import hoomd

    if not np.isfinite(k) or k <= 0:
        raise ValueError("k must be finite and positive.")
    force = hoomd.md.dihedral.Periodic()
    for name in dihedral_types:
        force.params[name] = {"k": 0.0, "d": -1, "n": 1, "phi0": 0.0}
    for center in reference.centers:
        force.params[center.type_name] = {
            "k": float(k),
            "d": -1,
            "n": 1,
            "phi0": center.phi0,
        }
    return force
