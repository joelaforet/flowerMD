"""Internal UFF atom and bonded parameters for the frozen AA-DPD protocol."""

import math
from itertools import combinations


def extract_uff_parameters(molecule):
    """Extract UFF parameters from an explicit-hydrogen RDKit molecule.

    The function sanitizes a copy. It preserves the input graph, properties,
    stereochemistry and conformers. Hydrogens must be graph atoms, not implicit
    hydrogens or atom-local explicit hydrogen counts. Disconnected graphs are
    supported. Dummy atoms and unsupported bond orders or UFF assignments
    raise ValueError.

    Returns
    -------
    dict
        ``particle_types`` is a tuple in atom-index order.
        ``particle_type_params`` maps names to ``r_min_a``,
        ``epsilon_kcal_mol`` and ``mass_amu``. ``bonds`` is a tuple of sorted
        atom-index pairs in lexicographic order. ``bond_orders`` and
        ``bond_types`` are aligned tuples. ``bond_params`` maps names to
        ``k_kcal_mol_a2`` and ``r0_a``. Exact parameter tuples define types
        in first-appearance order, including isotope mass for atoms.
        Bond orders describe the supplied graph. UFF assignment on the copy
        may perceive aromaticity in a Kekulized input representation.

        ``angles`` contains ``(first, center, third)`` tuples in center-index
        order with sorted neighbor pairs. ``angle_types`` is aligned and
        ``angle_params`` maps names to ``k_kcal_mol_rad2``, ``theta0_rad`` and
        ``uff_order``. Exact parameter triples define first-appearance types.

        ``dihedrals`` contains distinct-index proper torsions, ordered by
        sorted central bonds and sorted outer neighbors. ``dihedral_types``
        is aligned. ``dihedral_params`` contains ``k_kcal_mol``, ``n``, ``d``
        and ``phi0_rad=0.0``. Each getter barrier is divided by the number of
        eligible torsions around its central bond, including zero-barrier
        terms. Exact tuples of barrier, periodicity and sign define the types.

        ``r_min_a`` is the UFF minimum-energy distance in angstroms, not a
        Lennard-Jones sigma. Bond energy is ``0.5 * k * (r - r0)**2`` in
        kcal/mol for distances in angstroms. The function does not scale units,
        create forces, extract coordinates or assign improper terms.

        Full UFF angle bending uses geometry-dependent trigonometric forms.
        The initializer instead retains the frozen harmonic model
        ``0.5 * k * (theta - theta0)**2``. Its force consumer applies
        bonded_scale once. This function does not execute the trigonometric
        energy or apply scaling. The supported OpenFF SMIRNOFF angle handler
        supplies a harmonic potential directly.

        RDKit's getter supplies the starting target and stiffness. SP2
        small-ring target overrides retain that stiffness, so they do not
        reproduce full UFF small-ring curvature. ``uff_order`` records
        provenance; it does not select an executed expression. See
        ``_angle_order`` for the target and order rules. Angle-bearing SP3D
        and SP3D2 centers are unsupported because this adapter lacks their
        geometry-specific targets.

        Proper torsion energy is ``0.5 * bonded_scale * k *
        (1 + d*cos(n*phi))``. No scaling or extra factor of two is applied.
        SP central atoms have no torsion terms. Other torsion-bearing centers
        must be SP2 or SP3.
    """
    return _extract_uff_parameters_with_molecule(molecule)[0]


def _extract_uff_parameters_with_molecule(molecule):
    """Return existing tables and the same sanitized copy for GMSO assignment."""
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

    dihedrals, dihedral_types, dihedral_params, dihedral_keys = [], [], {}, {}
    for second, third in bonds:
        groups = [
            (first, second, third, fourth)
            for first in sorted(
                n.GetIdx() for n in mol.GetAtomWithIdx(second).GetNeighbors()
            )
            for fourth in sorted(
                n.GetIdx() for n in mol.GetAtomWithIdx(third).GetNeighbors()
            )
            if len({first, second, third, fourth}) == 4
        ]
        if not groups:
            continue
        centers = [mol.GetAtomWithIdx(index) for index in (second, third)]
        if any(
            atom.GetHybridization() == Chem.HybridizationType.SP
            for atom in centers
        ):
            continue
        for atom in centers:
            if atom.GetHybridization() not in (
                Chem.HybridizationType.SP2,
                Chem.HybridizationType.SP3,
            ):
                raise ValueError(
                    f"unsupported torsion center {atom.GetIdx()} with "
                    f"hybridization {atom.GetHybridization()}"
                )
        for group in groups:
            barrier = uff.GetUFFTorsionParams(mol, *group)
            if barrier is None:
                raise ValueError(
                    f"UFF did not assign torsion parameters to atoms {group}"
                )
            barrier = float(barrier)
            if not math.isfinite(barrier) or barrier < 0:
                raise ValueError(
                    f"UFF barrier for torsion {group} must be finite and nonnegative"
                )
            barrier /= len(groups)
            order, sign = _torsion_form(mol, group, Chem)
            key = (barrier, order, sign)
            if key not in dihedral_keys:
                name = f"uff_torsion_{len(dihedral_keys)}"
                dihedral_keys[key] = name
                dihedral_params[name] = {
                    "k_kcal_mol": barrier,
                    "n": order,
                    "d": sign,
                    "phi0_rad": 0.0,
                }
            dihedrals.append(group)
            dihedral_types.append(dihedral_keys[key])
    parameters = {
        "particle_types": tuple(particle_types),
        "particle_type_params": particle_params,
        "bonds": tuple(bonds),
        "bond_orders": tuple(bond_orders),
        "bond_types": tuple(bond_types),
        "bond_params": bond_params,
        "angles": tuple(angles),
        "angle_types": tuple(angle_types),
        "angle_params": angle_params,
        "dihedrals": tuple(dihedrals),
        "dihedral_types": tuple(dihedral_types),
        "dihedral_params": dihedral_params,
    }
    return parameters, mol


def _torsion_form(mol, group, chem):
    """Return the frozen proper-torsion periodicity and cosine sign."""
    first, second, third, fourth = group
    left, right = (mol.GetAtomWithIdx(index) for index in (second, third))
    left_hybrid, right_hybrid = (
        left.GetHybridization(),
        right.GetHybridization(),
    )
    sp2, sp3 = chem.HybridizationType.SP2, chem.HybridizationType.SP3
    chalcogens = {8, 16, 34, 52, 84}
    single = mol.GetBondBetweenAtoms(second, third).GetBondTypeAsDouble() == 1.0
    if left_hybrid == right_hybrid == sp3:
        if (
            single
            and left.GetAtomicNum() in chalcogens
            and right.GetAtomicNum() in chalcogens
        ):
            return 2, 1
        return 3, 1
    if left_hybrid == right_hybrid == sp2:
        return 2, -1
    if single:
        tetrahedral, trigonal = (
            (left, right) if left_hybrid == sp3 else (right, left)
        )
        if (
            tetrahedral.GetAtomicNum() in chalcogens
            and trigonal.GetAtomicNum() not in chalcogens
        ):
            return 2, 1
        if any(
            mol.GetAtomWithIdx(index).GetHybridization() == sp2
            for index in (first, fourth)
        ):
            return 3, 1
    return 6, -1


def _angle_order(atom, first, third, mol, chem):
    """Return UFF order metadata and an optional frozen angle target.

    SP centers record order 1 and ordinary SP2 centers order 3, with no target
    override. For an SP2 center in a three-member ring, the target is 60
    degrees if both neighbors belong to three-member rings and 150 degrees
    if exactly one does. Four-member ring cases use 90 and 135 degrees.
    Three-member target overrides take precedence over four-member overrides.
    Each check tests an atom's ring-size membership, not whether the atoms
    share one specific ring. Ring overrides record order 0.

    Other accepted centers record order 0 and retain the getter target.
    Overrides leave the getter stiffness unchanged. The order records
    provenance only; all accepted angles use the frozen harmonic model.
    """
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
