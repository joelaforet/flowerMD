"""Validate provider ownership and the preserved supplied-structure contract."""

from collections.abc import Mapping
from copy import deepcopy

import gmso
import numpy as np
import unyt as u

from flowermd.internal.openff_gmso import (
    _canonical,
    _improper_key,
    _validate_graph,
)


def own_bonded_assignment(result, original, molecule, atom_map):
    """Validate an assignment result and return independent owned copies."""
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise ValueError("provider assign must return topology and provenance")
    topology, report = result
    if not isinstance(topology, gmso.Topology):
        raise ValueError("provider must return a native GMSO Topology")
    if (
        not isinstance(report, Mapping)
        or not isinstance(report.get("source"), str)
        or not report["source"].strip()
    ):
        raise ValueError(
            "provider provenance requires a nonempty source string"
        )
    topology, report = deepcopy(topology), deepcopy(dict(report))
    edges = {
        tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        for bond in molecule.GetBonds()
    }
    _validate_graph(topology, molecule, atom_map, edges)
    if (
        topology.box is None
        or not np.array_equal(
            topology.box.lengths.to_value("nm"),
            original.box.lengths.to_value("nm"),
        )
        or not np.array_equal(
            topology.box.angles.to_value("degree"),
            original.box.angles.to_value("degree"),
        )
        or not np.array_equal(
            topology.positions.to_value("nm"), original.positions.to_value("nm")
        )
    ):
        raise ValueError("provider changed supplied coordinates or box")
    for site, before in zip(topology.sites, original.sites):
        excluded = {"atom_type_", "mass_", "position_"}
        if not _same_fields(site, before, excluded):
            raise ValueError(
                "provider changed site order, metadata or explicit charge"
            )
        mass = None if site.atom_type is None else site.atom_type.mass
        if not isinstance(mass, u.unyt_array) or mass.shape != ():
            raise ValueError(
                "provider atom types require scalar physical masses"
            )
        try:
            value = float(mass.to_value("amu"))
        except u.exceptions.UnitConversionError as error:
            raise ValueError(
                "provider atom types require physical masses in amu"
            ) from error
        if not np.isfinite(value) or value <= 0:
            raise ValueError(
                "provider atom types require positive finite masses"
            )
    indices = {site: i for i, site in enumerate(topology.sites)}
    original_indices = {site: i for i, site in enumerate(original.sites)}
    orders = {
        _canonical(
            (atom_map[b.GetBeginAtomIdx()], atom_map[b.GetEndAtomIdx()])
        ): b.GetBondTypeAsDouble()
        for b in molecule.GetBonds()
    }
    for bond in topology.bonds:
        if (
            bond.bond_order
            != orders[
                _canonical(tuple(indices[s] for s in bond.connection_members))
            ]
        ):
            raise ValueError(
                "provider bond orders do not match mapped RDKit chemistry"
            )
    for kind in ("bonds", "angles", "dihedrals", "impropers"):
        key_for = _improper_key if kind == "impropers" else _canonical
        assigned = {}
        for connection in getattr(topology, kind):
            key = key_for(
                tuple(indices[s] for s in connection.connection_members)
            )
            if key in assigned:
                raise ValueError(
                    "provider returned duplicate derived connection groups"
                )
            assigned[key] = connection
        for before in getattr(original, kind):
            key = key_for(
                tuple(original_indices[s] for s in before.connection_members)
            )
            after = assigned.get(key)
            if after is None:
                if getattr(before, "restraint", None):
                    raise ValueError("provider removed a restrained connection")
            elif not _same_fields(
                before,
                after,
                {
                    "connection_members_",
                    "bonds_",
                    "bond_order_",
                    "bond_type_",
                    "angle_type_",
                    "dihedral_type_",
                    "improper_type_",
                },
            ):
                raise ValueError(
                    "provider changed retained connection metadata"
                )
    return topology, report


def _same_fields(first, second, excluded):
    return type(first) is type(second) and all(
        _same_value(getattr(first, name), getattr(second, name))
        for name in type(first).model_fields
        if name not in excluded
    )


def _same_value(first, second):
    if isinstance(first, u.unyt_array) or isinstance(second, u.unyt_array):
        if not isinstance(first, u.unyt_array) or not isinstance(
            second, u.unyt_array
        ):
            return False
        try:
            return np.array_equal(first.value, second.to_value(first.units))
        except u.exceptions.UnitConversionError:
            return False
    if isinstance(first, Mapping):
        return (
            isinstance(second, Mapping)
            and first.keys() == second.keys()
            and all(
                _same_value(value, second[key]) for key, value in first.items()
            )
        )
    if hasattr(type(first), "model_fields"):
        return _same_fields(first, second, set())
    if isinstance(first, (list, tuple)):
        return (
            isinstance(second, type(first))
            and len(first) == len(second)
            and all(_same_value(a, b) for a, b in zip(first, second))
        )
    return np.array_equal(first, second)
