"""Internal UFF atom, bond and harmonic-angle surrogate parameters."""

import math
from itertools import combinations


def extract_uff_parameters(molecule):
    """Extract UFF parameters from an explicit-hydrogen RDKit molecule.

    A sanitized copy is used; the input graph, properties, stereochemistry and
    conformers are not modified. Hydrogens must be graph atoms, not implicit
    hydrogens or atom-local explicit hydrogen counts. Disconnected graphs are
    supported. Dummy atoms and unsupported bond orders or UFF assignments
    raise ValueError.

    Returns
    -------
    dict
        ``particle_types`` is a tuple in atom-index order;
        ``particle_type_params`` maps names to ``r_min_a``,
        ``epsilon_kcal_mol`` and ``mass_amu``; ``bonds`` is a tuple of sorted
        atom-index pairs in lexicographic order; ``bond_orders`` and
        ``bond_types`` are aligned tuples; ``bond_params`` maps names to
        ``k_kcal_mol_a2`` and ``r0_a``. Types are deduplicated by exact parameter
        tuples in first-appearance order, including isotope mass for atoms.
        Bond orders describe the supplied graph; UFF assignment on the private
        copy may perceive aromaticity in a Kekulized input representation.

        ``angles`` contains ``(first, center, third)`` tuples in center-index
        order with sorted neighbor pairs; ``angle_types`` is aligned and
        ``angle_params`` maps names to ``k_kcal_mol_rad2``, ``theta0_rad`` and
        ``uff_order``. Exact parameter triples define first-appearance types.

        ``r_min_a`` is the UFF minimum-energy distance in angstroms, not a
        Lennard-Jones sigma. Bond energy is ``0.5 * k * (r - r0)**2`` in
        kcal/mol for distances in angstroms. No unit scaling, force creation,
        coordinate extraction or torsion terms are performed.

        Angles describe the frozen AA-DPD harmonic surrogate, with energy
        ``0.5 * bonded_scale * k * (theta - theta0)**2``. The bonded scale is
        not applied here. ``uff_order`` records provenance only. SP2 small-ring
        targets are adjusted while retaining the getter force constant; this
        is not full-UFF small-ring curvature. Angle-bearing SP3D and SP3D2
        centers require geometry-specific targets and are unsupported.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdForceFieldHelpers as uff
    except ImportError as error:
        raise ImportError(
            "UFF extraction requires RDKit; add rdkit to your Pixi environment."
        ) from error

    if not isinstance(molecule, Chem.Mol) or molecule.GetNumAtoms() == 0:
        raise ValueError("molecule must be a nonempty RDKit Mol")
    mol = Chem.Mol(molecule)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            raise ValueError(
                f"dummy atom at index {atom.GetIdx()} is unsupported"
            )
    allowed_bonds = {
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE,
        Chem.BondType.AROMATIC,
    }
    input_orders = {}
    for bond in mol.GetBonds():
        if bond.GetBondType() not in allowed_bonds:
            raise ValueError(
                f"unsupported bond order at atoms "
                f"({bond.GetBeginAtomIdx()}, {bond.GetEndAtomIdx()})"
            )
        group = tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        input_orders[group] = bond.GetBondTypeAsDouble()
    try:
        Chem.SanitizeMol(mol)
    except Exception as error:
        raise ValueError(f"cannot sanitize molecule: {error}") from error
    for atom in mol.GetAtoms():
        if atom.GetNumImplicitHs() or atom.GetNumExplicitHs():
            raise ValueError(
                f"atom {atom.GetIdx()} requires explicit graph hydrogens"
            )
        if atom.GetDegree() >= 2 and atom.GetHybridization() in (
            Chem.HybridizationType.SP3D,
            Chem.HybridizationType.SP3D2,
        ):
            raise ValueError(
                f"angle center {atom.GetIdx()} with degree {atom.GetDegree()} "
                f"and hybridization {atom.GetHybridization()} requires "
                "unsupported geometry-specific targets"
            )
    if not uff.UFFHasAllMoleculeParams(mol):
        atoms = ", ".join(
            f"{atom.GetIdx()}:{atom.GetSymbol()}" for atom in mol.GetAtoms()
        )
        raise ValueError(f"incomplete UFF parameter assignment; atoms {atoms}")

    particle_types, particle_params, particle_keys = [], {}, {}
    for atom in mol.GetAtoms():
        index = atom.GetIdx()
        values = uff.GetUFFVdWParams(mol, index, index)
        if values is None:
            raise ValueError(
                f"UFF did not assign vdW parameters to atom {index}"
            )
        key = _positive_parameters((*values, atom.GetMass()), f"atom {index}")
        if key not in particle_keys:
            name = f"uff_vdw_{len(particle_keys)}"
            particle_keys[key] = name
            particle_params[name] = dict(
                zip(("r_min_a", "epsilon_kcal_mol", "mass_amu"), key)
            )
        particle_types.append(particle_keys[key])

    bonds, bond_orders, bond_types, bond_params, bond_keys = [], [], [], {}, {}
    for bond in sorted(
        mol.GetBonds(),
        key=lambda item: tuple(
            sorted((item.GetBeginAtomIdx(), item.GetEndAtomIdx()))
        ),
    ):
        group = tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        values = uff.GetUFFBondStretchParams(mol, *group)
        if values is None:
            raise ValueError(
                f"UFF did not assign bond parameters to atoms {group}"
            )
        key = _positive_parameters(values, f"bond {group}")
        if key not in bond_keys:
            name = f"uff_bond_{len(bond_keys)}"
            bond_keys[key] = name
            bond_params[name] = dict(zip(("k_kcal_mol_a2", "r0_a"), key))
        bonds.append(group)
        bond_orders.append(input_orders[group])
        bond_types.append(bond_keys[key])

    angles, angle_types, angle_params, angle_keys = [], [], {}, {}
    for atom in mol.GetAtoms():
        neighbors = sorted(
            neighbor.GetIdx() for neighbor in atom.GetNeighbors()
        )
        for first, third in combinations(neighbors, 2):
            group = (first, atom.GetIdx(), third)
            values = uff.GetUFFAngleBendParams(mol, *group)
            if values is None:
                raise ValueError(
                    f"UFF did not assign angle parameters to atoms {group}"
                )
            ka, theta_degrees = _positive_parameters(values, f"angle {group}")
            theta = math.radians(theta_degrees)
            if not 0 < theta <= math.pi:
                raise ValueError(
                    f"UFF target for angle {group} must be in (0, pi] radians"
                )
            order, adjusted = _angle_order(atom, first, third, mol, Chem)
            if adjusted is not None:
                theta = adjusted
            key = (ka, theta, order)
            if key not in angle_keys:
                name = f"uff_angle_{len(angle_keys)}"
                angle_keys[key] = name
                angle_params[name] = dict(
                    zip(("k_kcal_mol_rad2", "theta0_rad", "uff_order"), key)
                )
            angles.append(group)
            angle_types.append(angle_keys[key])
    return {
        "particle_types": tuple(particle_types),
        "particle_type_params": particle_params,
        "bonds": tuple(bonds),
        "bond_orders": tuple(bond_orders),
        "bond_types": tuple(bond_types),
        "bond_params": bond_params,
        "angles": tuple(angles),
        "angle_types": tuple(angle_types),
        "angle_params": angle_params,
    }


def _angle_order(atom, first, third, mol, chem):
    """Retain the frozen ring-membership precedence and UFF order metadata."""
    hybridization = atom.GetHybridization()
    if hybridization == chem.HybridizationType.SP:
        return 1, None
    if hybridization == chem.HybridizationType.SP2:
        rings = mol.GetRingInfo()
        for size, outside, inside in ((3, 150.0, 60.0), (4, 135.0, 90.0)):
            if rings.IsAtomInRingOfSize(atom.GetIdx(), size):
                first_inside = rings.IsAtomInRingOfSize(first, size)
                third_inside = rings.IsAtomInRingOfSize(third, size)
                if first_inside != third_inside:
                    return 0, math.radians(outside)
                if first_inside and third_inside:
                    return 0, math.radians(inside)
        return 3, None
    return 0, None


def _positive_parameters(values, location):
    parameters = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value <= 0 for value in parameters):
        raise ValueError(
            f"UFF parameters for {location} must be finite and positive"
        )
    return parameters
