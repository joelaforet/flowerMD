"""Assign the implemented UFF terms to existing GMSO topology objects."""

from collections.abc import Mapping
from copy import deepcopy
from numbers import Integral

import gmso
import unyt as u
from gmso.lib.potential_templates import PotentialTemplateLibrary

from flowermd.internal.uff import extract_uff_parameters


def assign_uff_parameters(topology, molecule, *, atom_map, include_impropers):
    """Return a typed topology copy and a report of the partial UFF assignment.

    Parameters
    ----------
    topology : gmso.Topology
        Untyped input, including the preidentified connections produced by
        flowerMD's ordinary mBuild-to-GMSO conversion. Positions, box, site
        names, labels and connection order are preserved on the copy.
    molecule : rdkit.Chem.Mol
        Authoritative explicit-hydrogen chemical graph. Input is not modified.
    atom_map : mapping of int to int
        Required bijection from RDKit atom indices to GMSO site indices.
        Elements and mapped bond edges must agree.
    include_impropers : bool
        Must explicitly be False in this partial implementation. True raises
        NotImplementedError; inferred improper connections remain untyped.

    Notes
    -----
    Assigns native GMSO atom, harmonic bond/angle and periodic proper types.
    Angles use the frozen harmonic surrogate, not the full UFF cosine form.
    Atom-type LJ sigma is converted from UFF's minimum-energy distance; this
    stores parameters and does not create LJ forces. No bonded scale is applied.
    Isotope masses overwrite copied site masses; explicit site charges remain.
    Charges are not parameterized. Sites without explicit charges inherit the
    native GMSO atom-type zero default; no formal-to-partial charge conversion
    is performed.

    Matching derived connections are assigned in place, accepting full
    reversal. Missing assigned groups are added. Unassigned derived groups
    (e.g. SP-centered proper torsions) are removed and reported, unless they
    carry restraints, which are rejected. Existing typed inputs are rejected.
    The report explicitly identifies deferred improper assignment; GMSO typing
    status must not be interpreted as full UFF coverage.
    """
    if not isinstance(include_impropers, bool):
        raise ValueError("include_impropers must be a bool")
    if include_impropers:
        raise NotImplementedError(
            "UFF improper assignment is not implemented; explicitly use include_impropers=False for partial coverage"
        )
    if not isinstance(topology, gmso.Topology):
        raise ValueError("topology must be a GMSO Topology")
    if (
        any(site.atom_type is not None for site in topology.sites)
        or any(
            connection.connection_type is not None
            for connection in topology.connections
        )
        or topology.pairpotential_types
    ):
        raise ValueError("UFF assignment requires an untyped topology")
    extracted = extract_uff_parameters(molecule)
    atom_count = len(extracted["particle_types"])
    if not isinstance(atom_map, Mapping) or len(atom_map) != atom_count:
        raise ValueError(
            "atom_map must be a complete atom-index to site-index bijection"
        )
    if any(
        isinstance(index, bool) or not isinstance(index, Integral)
        for index in (*atom_map.keys(), *atom_map.values())
    ):
        raise ValueError("atom_map indices must be integers, not booleans")
    if (
        set(atom_map) != set(range(atom_count))
        or set(atom_map.values()) != set(range(topology.n_sites))
        or topology.n_sites != atom_count
    ):
        raise ValueError(
            "atom_map must be a complete atom-index to site-index bijection"
        )
    source_sites = list(topology.sites)
    for atom_index, site_index in atom_map.items():
        element = source_sites[site_index].element
        if (
            element is None
            or element.atomic_number
            != molecule.GetAtomWithIdx(atom_index).GetAtomicNum()
        ):
            raise ValueError(
                f"element mismatch for RDKit atom {atom_index} and GMSO site {site_index}"
            )
    source_indices = {site: index for index, site in enumerate(source_sites)}
    edges = {
        _canonical(
            tuple(source_indices[site] for site in bond.connection_members)
        )
        for bond in topology.bonds
    }
    mapped_edges = {
        _canonical(tuple(atom_map[index] for index in group))
        for group in extracted["bonds"]
    }
    if edges != mapped_edges or len(edges) != topology.n_bonds:
        raise ValueError(
            "GMSO bond graph does not match the mapped RDKit graph"
        )
    for connection in (
        *topology.angles,
        *topology.dihedrals,
        *topology.impropers,
    ):
        try:
            group = tuple(
                source_indices[site] for site in connection.connection_members
            )
        except KeyError as error:
            raise ValueError(
                "derived connection refers to a site outside topology"
            ) from error
        pairs = (
            ((group[0], member) for member in group[1:])
            if isinstance(connection, gmso.Improper)
            else zip(group, group[1:])
        )
        if any(_canonical(pair) not in edges for pair in pairs):
            raise ValueError(
                f"invalid {type(connection).__name__} bond connectivity at sites {group}"
            )

    result = deepcopy(topology)
    sites = list(result.sites)
    indices = {site: index for index, site in enumerate(sites)}
    templates = PotentialTemplateLibrary()
    energy = u.kcal / u.mol
    atom_types = {
        name: gmso.AtomType.from_template(
            templates["LennardJonesPotential"],
            {
                "sigma": values["r_min_a"] / 2 ** (1 / 6) * u.angstrom,
                "epsilon": values["epsilon_kcal_mol"] * energy,
            },
            name=name,
            mass=values["mass_amu"] * u.amu,
        )
        for name, values in extracted["particle_type_params"].items()
    }
    for atom_index, name in enumerate(extracted["particle_types"]):
        site = sites[atom_map[atom_index]]
        site.atom_type = atom_types[name]
        site.mass = atom_types[name].mass.copy()
    bond_types = {
        name: gmso.BondType.from_template(
            templates["HarmonicBondPotential"],
            {
                "k": values["k_kcal_mol_a2"] * energy / u.angstrom**2,
                "r_eq": values["r0_a"] * u.angstrom,
            },
            name=name,
        )
        for name, values in extracted["bond_params"].items()
    }
    bond_assignments = {
        _canonical(tuple(atom_map[index] for index in group)): (
            bond_types[name],
            order,
        )
        for group, name, order in zip(
            extracted["bonds"],
            extracted["bond_types"],
            extracted["bond_orders"],
        )
    }
    for bond in result.bonds:
        key = _canonical(
            tuple(indices[site] for site in bond.connection_members)
        )
        bond.bond_type, bond.bond_order = bond_assignments[key]
    angle_types = {
        name: gmso.AngleType.from_template(
            templates["HarmonicAnglePotential"],
            {
                "k": values["k_kcal_mol_rad2"] * energy / u.radian**2,
                "theta_eq": values["theta0_rad"] * u.radian,
            },
            name=name,
            tags={"uff_order": values["uff_order"]},
        )
        for name, values in extracted["angle_params"].items()
    }
    dihedral_types = {
        name: gmso.DihedralType.from_template(
            templates["HOOMDPeriodicDihedralPotential"],
            {
                "k": values["k_kcal_mol"] * energy,
                "n": values["n"] * u.dimensionless,
                "d": values["d"] * u.dimensionless,
                "phi0": values["phi0_rad"] * u.radian,
            },
            name=name,
        )
        for name, values in extracted["dihedral_params"].items()
    }
    removed = {}
    for groups_key, types_key, potentials, connection_class in (
        ("angles", "angle_types", angle_types, gmso.Angle),
        ("dihedrals", "dihedral_types", dihedral_types, gmso.Dihedral),
    ):
        assigned = {
            _canonical(tuple(atom_map[index] for index in group)): (
                tuple(atom_map[index] for index in group),
                potentials[name],
            )
            for group, name in zip(extracted[groups_key], extracted[types_key])
        }
        removed[groups_key] = []
        for connection in tuple(getattr(result, groups_key)):
            group = tuple(
                indices[site] for site in connection.connection_members
            )
            key = _canonical(group)
            if key in assigned:
                _, connection.connection_type = assigned.pop(key)
            else:
                if getattr(connection, "restraint", None):
                    raise ValueError(
                        f"cannot remove unassigned {groups_key} {group} with a restraint"
                    )
                result.remove_connection(connection)
                removed[groups_key].append(group)
        for group, potential in assigned.values():
            connection = connection_class(
                connection_members=[sites[index] for index in group]
            )
            # GMSO may return an equivalent cached connection previously
            # removed through its public API; assign to the returned object.
            connection = result.add_connection(connection)
            connection.connection_type = potential
        removed[groups_key] = tuple(removed[groups_key])
    result.update_topology()
    from rdkit import rdBase

    report = {
        "source": "UFF",
        "rdkit_version": rdBase.rdkitVersion,
        "angle_model": "frozen harmonic surrogate",
        "include_impropers": False,
        "charges_parameterized": False,
        "absent_charge_default": "GMSO atom-type zero",
        "assigned_counts": {
            "atoms": result.n_sites,
            "bonds": result.n_bonds,
            "angles": result.n_angles,
            "proper_dihedrals": result.n_dihedrals,
        },
        "retained_untyped_impropers": result.n_impropers,
        "removed_unassigned_groups": removed,
    }
    return result, report


def _canonical(group):
    return min(tuple(group), tuple(reversed(group)))
