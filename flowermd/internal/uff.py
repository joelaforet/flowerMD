"""Internal extraction of unscaled UFF atom and harmonic bond parameters."""

import math


def extract_uff_atoms_and_bonds(molecule):
    """Extract UFF parameters from an explicit-hydrogen RDKit molecule.

    A sanitized copy is used; the input graph, properties, stereochemistry and
    conformers are not modified. Hydrogens must be graph atoms, not implicit
    hydrogens or atom-local explicit hydrogen counts. Disconnected graphs are
    supported. Dummy atoms and unsupported bond orders or UFF assignments
    raise ValueError.

    Returns
    -------
    dict
        Six fields: ``particle_types`` is a tuple in atom-index order;
        ``particle_type_params`` maps names to ``r_min_a``,
        ``epsilon_kcal_mol`` and ``mass_amu``; ``bonds`` is a tuple of sorted
        atom-index pairs in lexicographic order; ``bond_orders`` and
        ``bond_types`` are aligned tuples; ``bond_params`` maps names to
        ``k_kcal_mol_a2`` and ``r0_a``. Types are deduplicated by exact parameter
        tuples in first-appearance order, including isotope mass for atoms.
        Bond orders describe the supplied graph; UFF assignment on the private
        copy may perceive aromaticity in a Kekulized input representation.

        ``r_min_a`` is the UFF minimum-energy distance in angstroms, not a
        Lennard-Jones sigma. Bond energy is ``0.5 * k * (r - r0)**2`` in
        kcal/mol for distances in angstroms. No unit scaling, force creation,
        coordinate extraction or higher-order bonded terms are performed.
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
    return {
        "particle_types": tuple(particle_types),
        "particle_type_params": particle_params,
        "bonds": tuple(bonds),
        "bond_orders": tuple(bond_orders),
        "bond_types": tuple(bond_types),
        "bond_params": bond_params,
    }


def _positive_parameters(values, location):
    parameters = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value <= 0 for value in parameters):
        raise ValueError(
            f"UFF parameters for {location} must be finite and positive"
        )
    return parameters
