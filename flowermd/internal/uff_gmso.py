"""Assign the implemented UFF terms to existing GMSO topology objects."""

import math
from collections.abc import Mapping
from copy import deepcopy
from numbers import Integral

import gmso
import unyt as u
from gmso.lib.potential_templates import PotentialTemplateLibrary

from flowermd.internal.uff import _extract_uff_parameters_with_molecule


def assign_uff_parameters(
    topology, molecule, *, atom_map, include_impropers=True
):
    """Return a typed topology copy and a report of UFF assignment coverage.

    Parameters
    ----------
    topology : gmso.Topology
        Untyped input, including connections identified during flowerMD's
        mBuild-to-GMSO conversion. The copy preserves positions, box, site
        names, labels and the order of retained connections.
    molecule : rdkit.Chem.Mol
        Chemical graph with hydrogens as explicit atoms. The function preserves
        the input molecule.
    atom_map : mapping of int to int
        Required bijection from RDKit atom indices to GMSO site indices.
        Elements and mapped bond edges must agree.
    include_impropers : bool, default True
        Assign native UFF inversions. False leaves inferred impropers untyped
        and reports partial assignment.

    Notes
    -----
    The function assigns native GMSO atom types, harmonic bond and angle
    types, and periodic proper torsion types. Angles use the frozen harmonic
    model ``0.5*k*(theta-theta0)**2`` rather than full UFF's geometry-dependent
    trigonometric forms. RDKit supplies the starting angle targets and
    stiffnesses. Small-ring target overrides retain the getter stiffness and
    do not reproduce full UFF small-ring curvature. ``uff_order`` is provenance
    only. The force consumer applies bonded scaling once. See
    ``extract_uff_parameters`` for supported centers and target rules.

    The function converts UFF's minimum-energy distance to Lennard-Jones sigma
    and stores it on the atom type. It does not create Lennard-Jones forces
    or apply bonded scaling.

    Isotope masses overwrite copied site masses. Explicit site charges remain.
    The function does not assign charges. Sites without explicit charges use
    the native GMSO atom-type default of zero. Formal charges do not become
    partial charges.

    The function assigns matching derived connections in place. Angle and
    proper torsion matching accepts full reversal. Improper matching preserves
    the center and out atom roles and permits swapping only the two plane
    atoms. The function adds missing assigned groups. It removes and reports
    unassigned derived groups, such as proper torsions with SP centers.
    It rejects these groups if they carry restraints. It also rejects typed
    inputs.

    Inversions use the Wilson out-of-plane coordinate. GMSO members have the
    order center, plane atom, plane atom, out atom. The three ordered terms
    keep the getter's force constant, which already includes the division by
    three. Generic improper sorting or a harmonic-dihedral substitution
    changes this model. A compatible force backend is still required.
    This function does not construct forces.
    """
    if not isinstance(include_impropers, bool):
        raise ValueError("include_impropers must be a bool")
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
    extracted, sanitized = _extract_uff_parameters_with_molecule(molecule)
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
    if include_impropers:
        removed["impropers"] = _assign_inversions(result, sanitized, atom_map)
    result.update_topology()
    from rdkit import rdBase

    report = {
        "source": "UFF",
        "rdkit_version": rdBase.rdkitVersion,
        "angle_model": "frozen harmonic surrogate",
        "include_impropers": include_impropers,
        "improper_force_backend": "native UFF inversion backend required"
        if include_impropers
        else "not assigned",
        "charges_parameterized": False,
        "absent_charge_default": "GMSO atom-type zero",
        "assigned_counts": {
            "atoms": result.n_sites,
            "bonds": result.n_bonds,
            "angles": result.n_angles,
            "proper_dihedrals": result.n_dihedrals,
            "impropers": result.n_impropers if include_impropers else 0,
        },
        "retained_untyped_impropers": 0
        if include_impropers
        else result.n_impropers,
        "removed_unassigned_groups": removed,
    }
    return result, report


def _canonical(group):
    return min(tuple(group), tuple(reversed(group)))


def _improper_key(group):
    center, first, second, out = group
    return center, min(first, second), max(first, second), out


def _assign_inversions(topology, molecule, atom_map):
    """Assign native ordered inversion terms directly to GMSO classes."""
    from rdkit import Chem
    from rdkit.Chem import rdForceFieldHelpers as uff

    sites = list(topology.sites)
    indices = {site: index for index, site in enumerate(sites)}
    assignments, types = {}, {}
    for atom in molecule.GetAtoms():
        neighbors = sorted(
            neighbor.GetIdx() for neighbor in atom.GetNeighbors()
        )
        number = atom.GetAtomicNum()
        eligible = (
            number in (6, 7, 8)
            and atom.GetHybridization() == Chem.HybridizationType.SP2
        ) or number in (15, 33, 51, 83)
        if len(neighbors) != 3 or not eligible:
            continue
        center = atom.GetIdx()
        first, second, third = neighbors
        if number in (6, 7, 8):
            coefficients = (1.0, -1.0, 0.0)
        else:
            target = math.radians(
                {15: 84.4339, 33: 86.9735, 51: 87.7047, 83: 90.0}[number]
            )
            c1, c2 = -4 * math.cos(target), 1.0
            coefficients = (
                -(c1 * math.cos(target) + c2 * math.cos(2 * target)),
                c1,
                c2,
            )
        for group in (
            (first, center, second, third),
            (first, center, third, second),
            (second, center, third, first),
        ):
            k = uff.GetUFFInversionParams(molecule, *group)
            if k is None:
                raise ValueError(
                    f"UFF did not assign inversion parameters to atoms {group}"
                )
            k = float(k)
            if (
                not math.isfinite(k)
                or k < 0
                or not all(math.isfinite(value) for value in coefficients)
            ):
                raise ValueError(
                    f"UFF inversion parameters for atoms {group} must have finite nonnegative k and finite coefficients"
                )
            key = (k, *coefficients)
            if key not in types:
                types[key] = gmso.ImproperType(
                    name=f"uff_inversion_{len(types)}",
                    expression="k*(c0+c1*cos(omega)+c2*cos(2*omega))",
                    independent_variables={"omega"},
                    parameters={
                        "k": k * u.kcal / u.mol,
                        **{
                            name: value * u.dimensionless
                            for name, value in zip(
                                ("c0", "c1", "c2"), coefficients
                            )
                        },
                    },
                    tags={
                        "form": "uff_inversion",
                        "coordinate": "wilson_out_of_plane",
                        "member_convention": "center,plane1,plane2,out",
                    },
                )
            mapped = tuple(
                atom_map[index]
                for index in (center, group[0], group[2], group[3])
            )
            assignments[_improper_key(mapped)] = (mapped, types[key])
    removed = []
    for improper in tuple(topology.impropers):
        group = tuple(indices[site] for site in improper.connection_members)
        key = _improper_key(group)
        if key in assignments:
            _, improper.improper_type = assignments.pop(key)
        else:
            if getattr(improper, "restraint", None):
                raise ValueError(
                    f"cannot remove unassigned improper {group} with a restraint"
                )
            topology.remove_connection(improper)
            removed.append(group)
    for group, potential in assignments.values():
        improper = topology.add_connection(
            gmso.Improper(connection_members=[sites[index] for index in group])
        )
        improper.improper_type = potential
    return tuple(removed)
