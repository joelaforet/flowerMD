"""Assign SMIRNOFF valence parameters to native GMSO potentials."""

import hashlib
import importlib.metadata
from collections.abc import Mapping
from copy import deepcopy
from numbers import Integral
from pathlib import Path

import gmso
import numpy as np
import unyt as u
from gmso.lib.potential_templates import PotentialTemplateLibrary


def assign_sage_parameters(
    topology,
    molecule,
    *,
    atom_map,
    force_field="openff-2.3.0.offxml",
    assign_nonbonded=True,
):
    """Return a typed topology copy and an assignment report.

    ``molecule`` is the authoritative explicit-H RDKit graph. ``atom_map`` maps
    its atom indices bijectively to input GMSO site indices. Undefined stereo,
    isotopes, typed inputs and unsupported parameter forms raise errors.
    Site metadata, coordinates, box and explicit charges survive the copy.
    OpenFF supplies physical masses. No partial charges or forces are created.

    ``force_field`` accepts an OFFXML resource name, path or OpenFF ForceField.
    Public Toolkit labels supply harmonic bonds and angles and native periodic
    proper and improper Fourier arrays. Signed coefficients are divided by
    idivf exactly once. Improper members follow the exact center-first trefoil
    order and use that ordered periodic dihedral coordinate.
    Constraints remain flexible bonds and their matches appear in the report.

    False ``assign_nonbonded`` skips the vdW handler and parameter consumption.
    Its mass-bearing atom types have empty parameters and an explicit tag.
    True stores sigma and epsilon without creating Lennard-Jones forces.
    Derived connections are reconciled with assigned terms. Matching objects
    retain metadata; unassigned objects are removed unless restrained.
    This adapter does not provide Fourier-array or improper force execution.
    """
    from openff.toolkit import ForceField, Molecule
    from rdkit import Chem, rdBase

    if not isinstance(assign_nonbonded, bool):
        raise ValueError("assign_nonbonded must be a bool")
    if not isinstance(topology, gmso.Topology):
        raise ValueError("topology must be a GMSO Topology")
    if (
        any(s.atom_type is not None for s in topology.sites)
        or any(c.connection_type is not None for c in topology.connections)
        or topology.pairpotential_types
    ):
        raise ValueError("Sage assignment requires an untyped topology")
    if not isinstance(molecule, Chem.Mol) or molecule.GetNumAtoms() == 0:
        raise ValueError("molecule must be a nonempty RDKit molecule")
    mol = Chem.Mol(molecule)
    Chem.SanitizeMol(mol)
    if any(
        a.GetNumImplicitHs() or a.GetNumExplicitHs() for a in mol.GetAtoms()
    ):
        raise ValueError("molecule must contain hydrogens as explicit atoms")
    if any(a.GetIsotope() for a in mol.GetAtoms()):
        raise ValueError("OpenFF conversion does not preserve isotope masses")
    # User map numbers do not define this adapter's atom mapping. Removing
    # them also avoids false stereo centers caused by distinct mapped H atoms.
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    offmol = Molecule.from_rdkit(
        mol, allow_undefined_stereo=False, hydrogens_are_explicit=True
    )
    count = mol.GetNumAtoms()
    off_edges = {
        _canonical((b.atom1_index, b.atom2_index)) for b in offmol.bonds
    }
    mol_edges = {
        _canonical((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
        for b in mol.GetBonds()
    }
    if (
        offmol.n_atoms != count
        or off_edges != mol_edges
        or any(
            a.atomic_number != mol.GetAtomWithIdx(i).GetAtomicNum()
            or int(a.formal_charge.m_as("elementary_charge"))
            != mol.GetAtomWithIdx(i).GetFormalCharge()
            for i, a in enumerate(offmol.atoms)
        )
    ):
        raise ValueError("OpenFF conversion changed atom indices or graph")
    _validate_graph(topology, mol, atom_map, mol_edges)
    resource = (
        _resource_path(force_field)
        if isinstance(force_field, (str, Path))
        else None
    )
    if not isinstance(force_field, (str, Path, ForceField)):
        raise ValueError(
            "force_field must be an OFFXML resource or OpenFF ForceField"
        )
    ff = (
        ForceField(str(resource or force_field))
        if isinstance(force_field, (str, Path))
        else deepcopy(force_field)
    )
    # Remove vdW before labeling, so the unweighted path does not even match
    # its parameters. Charge handlers never run assignment during labeling.
    if not assign_nonbonded and "vdW" in ff.registered_parameter_handlers:
        ff.deregister_parameter_handler("vdW")
    forms = {
        "Bonds": {"harmonic", "(k/2)*(r-length)^2"},
        "Angles": {"harmonic"},
        "ProperTorsions": {"k*(1+cos(periodicity*theta-phase))"},
        "ImproperTorsions": {"k*(1+cos(periodicity*theta-phase))"},
        "vdW": {"Lennard-Jones-12-6"},
    }
    for handler, supported in forms.items():
        if (
            handler in ff.registered_parameter_handlers
            and ff[handler].potential not in supported
        ):
            raise ValueError(
                f"unsupported {handler} potential {ff[handler].potential}"
            )
    labels = ff.label_molecules(offmol.to_topology())[0]
    neighbors = {i: set() for i in range(count)}
    for a, b in mol_edges:
        neighbors[a].add(b)
        neighbors[b].add(a)
    angles = {
        _canonical((a, c, b))
        for c in neighbors
        for a in neighbors[c]
        for b in neighbors[c]
        if a != b
    }
    propers = {
        _canonical((a, b, c, d))
        for b, c in mol_edges
        for a in neighbors[b] - {c}
        for d in neighbors[c] - {b, a}
    }
    for name, expected in (
        ("Bonds", mol_edges),
        ("Angles", angles),
        ("ProperTorsions", propers),
    ):
        actual = [_canonical(group) for group in labels.get(name, {})]
        if set(actual) != expected or len(actual) != len(set(actual)):
            raise ValueError(f"missing or unexpected {name} assignments")
    if assign_nonbonded and set(labels.get("vdW", {})) != {
        (i,) for i in range(count)
    }:
        raise ValueError("missing vdW assignments")

    result = deepcopy(topology)
    sites = list(result.sites)
    templates = PotentialTemplateLibrary()
    atom_types = {}
    for i, atom in enumerate(offmol.atoms):
        mass = float(atom.mass.m_as("dalton"))
        parameter = labels["vdW"][(i,)] if assign_nonbonded else None
        cache_key = (parameter.smirks if parameter is not None else None, mass)
        if cache_key in atom_types:
            atom_type = atom_types[cache_key]
            sites[atom_map[i]].atom_type = atom_type
            sites[atom_map[i]].mass = atom_type.mass.copy()
            continue
        kwargs = dict(
            name=f"sage_atom_{len(atom_types)}",
            mass=_quantity(atom.mass, "dalton", u.amu),
            tags={"source": "Sage", "nonbonded_assigned": assign_nonbonded},
        )
        if assign_nonbonded:
            parameter = labels["vdW"][(i,)]
            sigma = _quantity(parameter.sigma, "angstrom", u.angstrom)
            epsilon = _quantity(
                parameter.epsilon, "kilocalorie_per_mole", u.kcal / u.mol
            )
            if sigma <= 0 or epsilon < 0:
                raise ValueError(
                    "sigma must be positive and epsilon nonnegative"
                )
            atom_type = gmso.AtomType.from_template(
                templates["LennardJonesPotential"],
                {"sigma": sigma, "epsilon": epsilon},
                **kwargs,
            )
        else:
            atom_type = gmso.AtomType(
                expression="0",
                independent_variables=set(),
                parameters={},
                **kwargs,
            )
        atom_types[cache_key] = atom_type
        sites[atom_map[i]].atom_type = atom_type
        sites[atom_map[i]].mass = atom_type.mass.copy()

    removed = {}
    for label, kind, cls, type_cls, template in (
        ("Bonds", "bonds", gmso.Bond, gmso.BondType, "HarmonicBondPotential"),
        (
            "Angles",
            "angles",
            gmso.Angle,
            gmso.AngleType,
            "HarmonicAnglePotential",
        ),
        (
            "ProperTorsions",
            "dihedrals",
            gmso.Dihedral,
            gmso.DihedralType,
            "PeriodicTorsionPotential",
        ),
        (
            "ImproperTorsions",
            "impropers",
            gmso.Improper,
            gmso.ImproperType,
            "PeriodicImproperPotential",
        ),
    ):
        assignments = {}
        potentials = {}
        for group, parameter in labels.get(label, {}).items():
            if any(
                getattr(parameter, attr, None)
                for attr in ("k_bondorder", "length_bondorder")
            ):
                raise ValueError(
                    f"unsupported fractional bond-order interpolation in {label}"
                )
            tags = {
                "source": "Sage",
                "parameter_id": parameter.id,
                "smirks": parameter.smirks,
            }
            groups = [group]
            if label == "Bonds":
                params = {
                    "k": _quantity(
                        parameter.k,
                        "kilocalorie_per_mole/angstrom**2",
                        u.kcal / u.mol / u.angstrom**2,
                    ),
                    "r_eq": _quantity(parameter.length, "angstrom", u.angstrom),
                }
            elif label == "Angles":
                params = {
                    "k": _quantity(
                        parameter.k,
                        "kilocalorie_per_mole/radian**2",
                        u.kcal / u.mol / u.radian**2,
                    ),
                    "theta_eq": _quantity(parameter.angle, "radian", u.radian),
                }
            else:
                params, divisors = _fourier(parameter, ff[label], label)
                tags["idivf"] = divisors
                tags["coordinate"] = "periodic_dihedral"
                if label == "ImproperTorsions":
                    a, c, b, d = group
                    if set((a, b, d)) - neighbors[c] or len(set(group)) != 4:
                        raise ValueError("invalid improper assignment graph")
                    groups = [(c, a, b, d), (c, b, d, a), (c, d, a, b)]
                    tags["member_convention"] = "center,outer1,outer2,outer3"
            if parameter.smirks not in potentials:
                potentials[parameter.smirks] = type_cls.from_template(
                    templates[template],
                    params,
                    name=f"sage_{label}_{parameter.id}",
                    tags=tags,
                )
            potential = potentials[parameter.smirks]
            for ordered in groups:
                mapped = tuple(atom_map[i] for i in ordered)
                key = (
                    _improper_key(mapped)
                    if kind == "impropers"
                    else _canonical(mapped)
                )
                if key in assignments:
                    raise ValueError(f"duplicate {label} assignment")
                assignments[key] = (mapped, potential)
        removed[kind] = _reconcile(result, sites, kind, cls, assignments)
    # Preserve the authoritative chemical bond orders without changing members.
    site_indices = {s: i for i, s in enumerate(sites)}
    orders = {
        _canonical(
            (atom_map[b.GetBeginAtomIdx()], atom_map[b.GetEndAtomIdx()])
        ): b.GetBondTypeAsDouble()
        for b in mol.GetBonds()
    }
    for bond in result.bonds:
        bond.bond_order = orders[
            _canonical(tuple(site_indices[s] for s in bond.connection_members))
        ]
    result.update_topology()
    return result, {
        "source": "Sage",
        "force_field": str(force_field)
        if isinstance(force_field, (str, Path))
        else "OpenFF ForceField object",
        "resource_sha256": hashlib.sha256(resource.read_bytes()).hexdigest()
        if resource
        else None,
        "resource_hash_scope": "raw OFFXML bytes"
        if resource
        else "no resolved OFFXML resource",
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("openff-toolkit", "gmso", "openff-units")
        },
        "rdkit_version": rdBase.rdkitVersion,
        "aromaticity_model": ff.aromaticity_model,
        "mass_source": "OpenFF atom masses",
        "assign_nonbonded": assign_nonbonded,
        "charges_parameterized": False,
        "absent_charge_default": "GMSO atom-type zero",
        "constraint_treatment": "flexible bonded potentials",
        "constraint_matches": tuple(
            tuple(atom_map[i] for i in group)
            for group in labels.get("Constraints", {})
        ),
        "assigned_counts": {
            "atoms": result.n_sites,
            "bonds": result.n_bonds,
            "angles": result.n_angles,
            "proper_dihedrals": result.n_dihedrals,
            "impropers": result.n_impropers,
        },
        "removed_unassigned_groups": removed,
        "force_backend": "Fourier-array and periodic-improper execution required",
    }


def _resource_path(source):
    path = Path(source)
    if path.is_file():
        return path.resolve()
    from openff.toolkit.typing.engines.smirnoff import (
        get_available_force_fields,
    )

    for candidate in get_available_force_fields(full_paths=True):
        if Path(candidate).name == str(source):
            return Path(candidate)
    return None


def _quantity(value, unit_name, unit):
    numeric = np.asarray(value.m_as(unit_name), dtype=float)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("Sage parameters must be finite")
    return u.unyt_array(numeric, unit)


def _fourier(parameter, handler, label):
    ks, phases, ns = parameter.k, parameter.phase, parameter.periodicity
    if not len(ks) or len(ks) != len(phases) or len(ks) != len(ns):
        raise ValueError("Fourier arrays must have equal nonzero lengths")
    divisors = parameter.idivf
    if divisors is None:
        if label == "ProperTorsions":
            raise ValueError("proper torsions require explicit idivf")
        default = handler.default_idivf
        divisors = [3.0 if default == "auto" else float(default)] * len(ks)
    if len(divisors) != len(ks):
        raise ValueError("idivf length must match Fourier arrays")
    if any(v is None for v in divisors):
        if label == "ProperTorsions":
            raise ValueError("proper torsions require explicit idivf")
        default = handler.default_idivf
        divisors = [
            v if v is not None else 3.0 if default == "auto" else float(default)
            for v in divisors
        ]
    divisors = [float(v) for v in divisors]
    if any(not np.isfinite(v) or v <= 0 for v in divisors):
        raise ValueError("idivf must be finite and positive")
    if any(not np.isfinite(n) or n < 1 or int(n) != n for n in ns):
        raise ValueError("periodicities must be positive integers")
    coefficients = np.array(
        [
            float(_quantity(k, "kilocalorie_per_mole", u.kcal / u.mol))
            for k in ks
        ]
    )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        normalized = coefficients / np.asarray(divisors)
    if not np.all(np.isfinite(normalized)) or np.any(
        (coefficients != 0) & (normalized == 0)
    ):
        raise ValueError(
            "normalized Fourier coefficients exceed floating-point range"
        )
    return {
        "k": u.unyt_array(normalized, u.kcal / u.mol),
        "n": u.unyt_array(ns, u.dimensionless),
        "phi_eq": u.unyt_array(
            [float(_quantity(p, "radian", u.radian)) for p in phases], u.radian
        ),
    }, divisors


def _canonical(group):
    return min(tuple(group), tuple(reversed(group)))


def _improper_key(group):
    center, a, b, d = group
    return center, min(a, b), max(a, b), d


def _validate_graph(topology, molecule, atom_map, mol_edges):
    count = molecule.GetNumAtoms()
    if (
        not isinstance(atom_map, Mapping)
        or len(atom_map) != count
        or any(
            isinstance(i, bool) or not isinstance(i, Integral)
            for i in (*atom_map.keys(), *atom_map.values())
        )
        or set(atom_map) != set(range(count))
        or set(atom_map.values()) != set(range(topology.n_sites))
        or topology.n_sites != count
    ):
        raise ValueError(
            "atom_map must be a complete integer atom-index to site-index bijection"
        )
    sites = list(topology.sites)
    for atom, site in atom_map.items():
        if (
            sites[site].element is None
            or sites[site].element.atomic_number
            != molecule.GetAtomWithIdx(atom).GetAtomicNum()
        ):
            raise ValueError("element mismatch in atom_map")
    indices = {s: i for i, s in enumerate(sites)}
    edges = {
        _canonical(tuple(indices[s] for s in b.connection_members))
        for b in topology.bonds
    }
    if (
        edges
        != {_canonical(tuple(atom_map[i] for i in edge)) for edge in mol_edges}
        or len(edges) != topology.n_bonds
    ):
        raise ValueError("GMSO bond graph does not match mapped RDKit graph")
    for connection in (
        *topology.angles,
        *topology.dihedrals,
        *topology.impropers,
    ):
        group = tuple(indices[s] for s in connection.connection_members)
        pairs = (
            [(group[0], i) for i in group[1:]]
            if isinstance(connection, gmso.Improper)
            else zip(group, group[1:])
        )
        if any(_canonical(pair) not in edges for pair in pairs):
            raise ValueError("invalid derived connection bond connectivity")


def _reconcile(topology, sites, kind, cls, assignments):
    indices = {s: i for i, s in enumerate(sites)}
    key_for = _improper_key if kind == "impropers" else _canonical
    removed = []
    for connection in tuple(getattr(topology, kind)):
        group = tuple(indices[s] for s in connection.connection_members)
        key = key_for(group)
        if key in assignments:
            ordered, connection.connection_type = assignments.pop(key)
            if kind == "impropers":
                connection.connection_members = [sites[i] for i in ordered]
        else:
            if getattr(connection, "restraint", None):
                raise ValueError(
                    f"cannot remove unassigned {kind} with a restraint"
                )
            topology.remove_connection(connection)
            removed.append(group)
    for ordered, potential in assignments.values():
        connection = topology.add_connection(
            cls(connection_members=[sites[i] for i in ordered])
        )
        connection.connection_members = [sites[i] for i in ordered]
        connection.connection_type = potential
    return tuple(removed)
