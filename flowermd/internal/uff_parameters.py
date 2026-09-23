"""UFF bonded and pair parameters from RDKit, for the all-atom DPD force field.

RDKit's Universal Force Field typer assigns every atom in a sanitized
molecule, so the tables come straight from the chemical graph with no
force-field XML. Bonds are harmonic, angles are harmonic at UFF's
equilibrium geometry with UFF's local curvature, and torsions are the UFF
cosine form. UFF inversion (improper) terms are not exported: UFF uses a
Wilson out-of-plane coordinate that HOOMD's dihedral-based impropers do not
reproduce, so the UFF source has no impropers by design.

Requires RDKit (imported lazily).
"""

import itertools
import math

import numpy as np

from flowermd.internal.all_atom_parameters import (
    AllAtomParameters,
    compound_box_lengths_a,
    compound_to_rdkit,
    intern_type,
    merge_parameters,
    molecular_compounds,
    strip_keys,
)


def _angle_order(atom, first, third, mol):
    """Return RDKit's UFF angle order and any ring-adjusted theta0 (rad)."""
    from rdkit.Chem.rdchem import HybridizationType as H

    hybrid = atom.GetHybridization()
    if hybrid == H.SP:
        return 1, None
    if hybrid == H.SP2:
        rings = mol.GetRingInfo()
        center = atom.GetIdx()
        for size, outside, inside in ((3, 150.0, 60.0), (4, 135.0, 90.0)):
            if rings.IsAtomInRingOfSize(center, size):
                in_first = rings.IsAtomInRingOfSize(first, size)
                in_third = rings.IsAtomInRingOfSize(third, size)
                if in_first != in_third:
                    return 0, math.radians(outside)
                if in_first and in_third:
                    return 0, math.radians(inside)
        return 3, None
    if hybrid == H.SP3D2:
        return 4, None
    return 0, None


def _torsion_form(mol, group):
    """Return ``(periodicity, cos_term)`` following RDKit's TorsionAngle."""
    from rdkit.Chem.rdchem import HybridizationType as H

    i, j, k, ell = group
    aj, ak = mol.GetAtomWithIdx(j), mol.GetAtomWithIdx(k)
    hj, hk = aj.GetHybridization(), ak.GetHybridization()
    order, cos_term = 6, 1
    group6 = {8, 16, 34, 52, 84}
    bond_order = mol.GetBondBetweenAtoms(j, k).GetBondTypeAsDouble()
    if hj == H.SP3 and hk == H.SP3:
        order, cos_term = 3, -1
        if (
            bond_order == 1.0
            and aj.GetAtomicNum() in group6
            and ak.GetAtomicNum() in group6
        ):
            order, cos_term = 2, -1
    elif hj == H.SP2 and hk == H.SP2:
        order, cos_term = 2, 1
    elif bond_order == 1.0:
        sp3, sp2 = (aj, ak) if hj == H.SP3 else (ak, aj)
        if sp3.GetAtomicNum() in group6 and sp2.GetAtomicNum() not in group6:
            order, cos_term = 2, -1
        elif (
            mol.GetAtomWithIdx(i).GetHybridization() == H.SP2
            or mol.GetAtomWithIdx(ell).GetHybridization() == H.SP2
        ):
            order, cos_term = 3, -1
    return order, cos_term


def _parameterize_one(compound, box_lengths_a):
    from rdkit.Chem import rdForceFieldHelpers as uff

    mol = compound_to_rdkit(compound)
    if not uff.UFFHasAllMoleculeParams(mol):
        raise ValueError("UFF does not cover every atom in this compound.")
    n_atoms = mol.GetNumAtoms()
    masses = np.asarray(
        [mol.GetAtomWithIdx(i).GetMass() for i in range(n_atoms)]
    )
    result = AllAtomParameters(
        positions_a=np.asarray(compound.xyz, dtype=float) * 10.0,
        box_lengths_a=box_lengths_a,
        masses_amu=masses,
        source="uff",
    )

    particle_table = {}
    for i in range(n_atoms):
        values = uff.GetUFFVdWParams(mol, i, i)
        if values is None:
            raise ValueError(f"UFF assigned no vdW parameters to atom {i}.")
        sigma, epsilon = map(float, values)
        name = intern_type(
            particle_table,
            "uff_",
            (sigma, epsilon, float(masses[i])),
            {"epsilon": epsilon},
        )
        result.particle_types.append(name)
    result.particle_epsilons = {
        name: entry["epsilon"] for name, entry in particle_table.items()
    }
    result.particle_keys = {
        name: entry["_key"] for name, entry in particle_table.items()
    }
    result.epsilon_ref = max(result.particle_epsilons.values())

    bond_table = {}
    sorted_bonds = sorted(
        mol.GetBonds(),
        key=lambda b: tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))),
    )
    for bond in sorted_bonds:
        group = tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        values = uff.GetUFFBondStretchParams(mol, *group)
        if values is None:
            raise ValueError(f"UFF assigned no parameters to bond {group}.")
        k, r0 = map(float, values)
        # RDKit UFF: E = k/2 (r - r0)^2, same convention as HOOMD Harmonic.
        name = intern_type(bond_table, "uff_bond_", (k, r0), {"k": k, "r0": r0})
        result.bonds.append(group)
        result.bond_types.append(name)
    result.bond_params = strip_keys(bond_table)

    angle_table = {}
    for center in range(n_atoms):
        atom = mol.GetAtomWithIdx(center)
        neighbors = sorted(n.GetIdx() for n in atom.GetNeighbors())
        for first, third in itertools.combinations(neighbors, 2):
            group = (first, center, third)
            values = uff.GetUFFAngleBendParams(mol, *group)
            if values is None:
                raise ValueError(
                    f"UFF assigned no parameters to angle {group}."
                )
            k, theta0_degrees = map(float, values)
            theta0 = math.radians(theta0_degrees)
            order, adjusted = _angle_order(atom, first, third, mol)
            if adjusted is not None:
                theta0 = adjusted
            # Harmonic in theta at UFF's minimum with UFF's local curvature.
            name = intern_type(
                angle_table,
                "uff_angle_",
                (k, theta0, order),
                {"k": k, "t0": theta0},
            )
            result.angles.append(group)
            result.angle_types.append(name)
    result.angle_params = strip_keys(angle_table)

    torsion_table = {}
    for bond in sorted_bonds:
        j, k = sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        groups = []
        for i in sorted(
            n.GetIdx()
            for n in mol.GetAtomWithIdx(j).GetNeighbors()
            if n.GetIdx() != k
        ):
            for ell in sorted(
                n.GetIdx()
                for n in mol.GetAtomWithIdx(k).GetNeighbors()
                if n.GetIdx() not in (j, i)
            ):
                if uff.GetUFFTorsionParams(mol, i, j, k, ell) is not None:
                    groups.append((i, j, k, ell))
        # UFF shares one barrier across all torsions about a bond.
        for group in groups:
            barrier = float(uff.GetUFFTorsionParams(mol, *group)) / len(groups)
            order, cos_term = _torsion_form(mol, group)
            # UFF: E = V/2 (1 - cos(n phi0) cos(n phi)); HOOMD Periodic:
            # E = k/2 (1 + d cos(n phi - phi0)), so d = -cos_term, phi0 = 0.
            name = intern_type(
                torsion_table,
                "uff_torsion_",
                (barrier, order, cos_term),
                {"k": barrier, "n": order, "d": -cos_term, "phi0": 0.0},
            )
            result.dihedrals.append(group)
            result.dihedral_types.append(name)
    result.dihedral_params = strip_keys(torsion_table)
    return result


def parameterize_uff(compound):
    """Return `AllAtomParameters` for `compound` from RDKit's UFF typer.

    Disconnected molecules are parameterized once per distinct chemical
    graph and the tables are merged, so a melt of identical chains costs
    one RDKit pass.

    Parameters
    ----------
    compound : mbuild.Compound, required
        All-atom compound with elements on every particle and a periodic
        ``box``. Coordinates in nm (mBuild convention).

    """
    box_lengths_a = compound_box_lengths_a(compound)
    molecules = molecular_compounds(compound)
    if molecules == [compound]:
        return _parameterize_one(compound, box_lengths_a)
    cache = {}
    chunks = []
    for molecule in molecules:
        particles = list(molecule.particles())
        local = {p: i for i, p in enumerate(particles)}
        key = (
            tuple(p.element.atomic_number for p in particles),
            tuple(
                sorted(
                    tuple(sorted((local[a], local[b])))
                    for a, b in molecule.bonds()
                )
            ),
        )
        chunk = cache.get(key)
        if chunk is None:
            chunk = _parameterize_one(molecule, box_lengths_a)
            cache[key] = chunk
        chunks.append(chunk)
    return merge_parameters(
        chunks,
        positions_a=np.asarray(compound.xyz, dtype=float) * 10.0,
        box_lengths_a=box_lengths_a,
        source="uff",
    )
