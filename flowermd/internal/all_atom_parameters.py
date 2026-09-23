"""Numeric bonded and pair parameters for the all-atom DPD force field.

Everything here is in Angstrom, kcal/mol and amu. Interaction types are
named by their complete coefficient tuple, not by the atom types they join,
so two bonds between the same elements with different equilibrium lengths
stay distinct.
"""

from dataclasses import dataclass, field

import gsd.hoomd
import numpy as np


@dataclass
class AllAtomParameters:
    """HOOMD-ready parameter tables for one periodic all-atom system.

    Attributes
    ----------
    positions_a : numpy.ndarray, shape (N, 3)
        Particle positions in Angstrom (unwrapped).
    box_lengths_a : numpy.ndarray, shape (3,)
        Orthorhombic box lengths in Angstrom.
    masses_amu : numpy.ndarray, shape (N,)
    particle_types : list of str
        One type name per particle.
    particle_epsilons : dict
        Type name to Lennard-Jones well depth in kcal/mol, used to weight
        the DPD coefficients.
    particle_keys : dict
        Type name to the coefficient tuple that defines it.
    bonds, angles, dihedrals, impropers : list of tuple
        Particle index groups.
    bond_types, angle_types, dihedral_types, improper_types : list of str
        One type name per group.
    bond_params, angle_params, dihedral_params, improper_params : dict
        Type name to HOOMD parameter dictionary. Bonds carry ``k`` and
        ``r0``; angles ``k`` and ``t0``; dihedrals and impropers ``k``,
        ``n``, ``d`` and ``phi0``, in the conventions of
        `hoomd.md.bond.Harmonic`, `hoomd.md.angle.Harmonic` and
        `hoomd.md.dihedral.Periodic`.
    epsilon_ref : float
        Largest particle epsilon, the DPD weighting reference.
    source : str
        Which parameter source produced the tables.

    """

    positions_a: np.ndarray
    box_lengths_a: np.ndarray
    masses_amu: np.ndarray = field(default_factory=lambda: np.empty(0))
    particle_types: list = field(default_factory=list)
    particle_epsilons: dict = field(default_factory=dict)
    particle_keys: dict = field(default_factory=dict)
    bonds: list = field(default_factory=list)
    bond_types: list = field(default_factory=list)
    bond_params: dict = field(default_factory=dict)
    angles: list = field(default_factory=list)
    angle_types: list = field(default_factory=list)
    angle_params: dict = field(default_factory=dict)
    dihedrals: list = field(default_factory=list)
    dihedral_types: list = field(default_factory=list)
    dihedral_params: dict = field(default_factory=dict)
    impropers: list = field(default_factory=list)
    improper_types: list = field(default_factory=list)
    improper_params: dict = field(default_factory=dict)
    epsilon_ref: float = 0.0
    source: str = ""

    @property
    def n_particles(self):
        """Number of particles."""
        return len(self.positions_a)


def intern_type(table, prefix, key, values):
    """Return the name for `key` in `table`, adding it if new.

    `table` maps type name to ``{"_key": key, **values}``. Two interactions
    with the same coefficient tuple share one name.
    """
    for name, entry in table.items():
        if entry["_key"] == key:
            return name
    name = f"{prefix}{len(table)}"
    table[name] = {"_key": key, **values}
    return name


def strip_keys(table):
    """Return `table` without the internal ``_key`` entries."""
    return {
        name: {k: v for k, v in entry.items() if k != "_key"}
        for name, entry in table.items()
    }


def molecular_compounds(compound):
    """Return the disconnected top-level molecules of `compound`.

    Returns the children of `compound` when they partition its particles
    and no bond crosses between them, otherwise ``[compound]``.
    """
    children = list(compound.children)
    if len(children) <= 1:
        return [compound]
    owner = {p: child for child in children for p in child.particles()}
    if sum(child.n_particles for child in children) != compound.n_particles:
        return [compound]
    if any(owner.get(a) is not owner.get(b) for a, b in compound.bonds()):
        return [compound]
    return children


def compound_to_rdkit(compound):
    """Return a sanitized RDKit molecule for `compound`.

    mBuild leaves the order of bonds it created itself unspecified; those
    are treated as single bonds. Bond orders read from a SMILES string are
    kept. Requires RDKit.
    """
    from rdkit import Chem

    mol = compound.to_rdkit()
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.UNSPECIFIED:
            bond.SetBondType(Chem.BondType.SINGLE)
    Chem.SanitizeMol(mol)
    particles = list(compound.particles())
    if mol.GetNumAtoms() != len(particles):
        raise ValueError(
            "mBuild and RDKit disagree on the atom count: "
            f"{len(particles)} vs {mol.GetNumAtoms()}."
        )
    for index, particle in enumerate(particles):
        element = getattr(particle, "element", None)
        number = getattr(element, "atomic_number", None)
        if number is None:
            raise ValueError(f"Particle {particle!r} has no element.")
        if number != mol.GetAtomWithIdx(index).GetAtomicNum():
            raise ValueError(
                f"mBuild and RDKit disagree on atom {index}: "
                f"{number} vs {mol.GetAtomWithIdx(index).GetAtomicNum()}."
            )
    return mol


def compound_box_lengths_a(compound):
    """Return the compound's periodic box lengths in Angstrom."""
    box = getattr(compound, "box", None)
    if box is None:
        raise ValueError("compound.box must define the periodic box.")
    return np.asarray(box.lengths, dtype=float) * 10.0


def merge_parameters(chunks, positions_a, box_lengths_a, source):
    """Concatenate per-molecule parameter tables into one system table.

    Type names are reconciled so that identical coefficient sets share a
    name across molecules and distinct sets never collide.
    """
    merged = AllAtomParameters(
        positions_a=positions_a, box_lengths_a=box_lengths_a, source=source
    )
    offset = 0
    tables = (
        ("bond_types", "bond_params"),
        ("angle_types", "angle_params"),
        ("dihedral_types", "dihedral_params"),
        ("improper_types", "improper_params"),
    )
    for chunk in chunks:
        for group_field in ("bonds", "angles", "dihedrals", "impropers"):
            getattr(merged, group_field).extend(
                tuple(i + offset for i in group)
                for group in getattr(chunk, group_field)
            )
        merged.masses_amu = np.concatenate(
            (merged.masses_amu, chunk.masses_amu)
        )
        rename = {}
        for old_name, key in chunk.particle_keys.items():
            match = next(
                (n for n, k in merged.particle_keys.items() if k == key), None
            )
            if match is None:
                match = old_name
                suffix = 1
                while match in merged.particle_keys:
                    match = f"{old_name}_{suffix}"
                    suffix += 1
                merged.particle_keys[match] = key
                merged.particle_epsilons[match] = chunk.particle_epsilons[
                    old_name
                ]
            rename[old_name] = match
        merged.particle_types.extend(
            rename[name] for name in chunk.particle_types
        )
        for types_field, params_field in tables:
            target = getattr(merged, params_field)
            rename = {}
            for old_name, values in getattr(chunk, params_field).items():
                match = next(
                    (n for n, v in target.items() if v == values), None
                )
                if match is None:
                    match = old_name
                    suffix = 1
                    while match in target:
                        match = f"{old_name}_{suffix}"
                        suffix += 1
                    target[match] = values
                rename[old_name] = match
            getattr(merged, types_field).extend(
                rename[name] for name in getattr(chunk, types_field)
            )
        merged.epsilon_ref = max(merged.epsilon_ref, chunk.epsilon_ref)
        offset += chunk.n_particles
    return merged


def _set_groups(container, groups, types):
    unique = list(dict.fromkeys(types))
    container.N = len(groups)
    container.types = unique
    if groups:
        container.typeid = np.asarray(
            [unique.index(t) for t in types], dtype=np.uint32
        )
        container.group = np.asarray(groups, dtype=np.uint32)


def to_gsd_frame(parameters):
    """Build a `gsd.hoomd.Frame` whose types match the parameter tables.

    Positions are wrapped into the box with matching image flags. Bond,
    angle, dihedral and improper types are the parameter-set names, so
    `AllAtomDPD` force parameters apply one to one.
    """
    lengths = np.asarray(parameters.box_lengths_a, dtype=float)
    unwrapped = np.asarray(parameters.positions_a, dtype=float)
    # Keep the image flags so unwrapping the state recovers the input
    # coordinates and molecules straddling the boundary stay whole.
    images = np.floor((unwrapped + lengths / 2) / lengths).astype(np.int32)
    positions = unwrapped - images * lengths
    frame = gsd.hoomd.Frame()
    frame.configuration.box = [*lengths, 0.0, 0.0, 0.0]
    frame.particles.N = len(positions)
    unique = list(dict.fromkeys(parameters.particle_types))
    frame.particles.types = unique
    frame.particles.typeid = np.asarray(
        [unique.index(t) for t in parameters.particle_types], dtype=np.uint32
    )
    frame.particles.position = positions
    frame.particles.image = images
    frame.particles.mass = np.asarray(parameters.masses_amu, dtype=float)
    frame.particles.charge = np.zeros(len(positions))
    _set_groups(frame.bonds, parameters.bonds, parameters.bond_types)
    _set_groups(frame.angles, parameters.angles, parameters.angle_types)
    _set_groups(
        frame.dihedrals, parameters.dihedrals, parameters.dihedral_types
    )
    _set_groups(
        frame.impropers, parameters.impropers, parameters.improper_types
    )
    return frame
