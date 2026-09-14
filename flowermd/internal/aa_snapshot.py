"""Build native GSD frames from assigned all-atom GMSO topologies."""

from collections.abc import Mapping

import gmso
import gsd.hoomd
import numpy as np
import sympy
import unyt as u
from gmso.lib.potential_templates import PotentialTemplateLibrary

_CATEGORIES = ("sites", "bonds", "angles", "dihedrals", "impropers")


def route_all_atom_connections(topology, *, type_labels):
    """Return physical type maps and ordered groups without changing inputs.

    Semantic maps must contain exactly the actual potential objects. Their
    names do not establish identity. This detects replaced objects, but not
    parameter edits on an unchanged object. Periodic impropers follow proper
    groups in the physical dihedral block, with their exact center-first order.
    Other improper forms remain in the improper block. Routing does not imply
    that a force backend supports the stored form.
    """
    if not isinstance(topology, gmso.Topology) or not topology.n_sites:
        raise ValueError("topology must be a nonempty GMSO Topology")
    if not isinstance(type_labels, Mapping) or set(type_labels) != set(
        _CATEGORIES
    ):
        raise ValueError(
            "type_labels must contain exactly the semantic categories"
        )
    maps, groups = {}, {}
    indices = {id(site): i for i, site in enumerate(topology.sites)}
    for kind in _CATEGORIES:
        items = tuple(getattr(topology, kind))
        labels = type_labels[kind]
        potentials = [
            item.atom_type if kind == "sites" else item.connection_type
            for item in items
        ]
        if not isinstance(labels, Mapping) or {id(p) for p in labels} != {
            id(p) for p in potentials
        }:
            raise ValueError(
                f"{kind} labels must match actual potential objects exactly"
            )
        if kind != "impropers" and None in labels:
            raise ValueError(f"{kind} require assigned potentials")
        if any(
            not isinstance(label, str) or not label or "\x00" in label
            for label in labels.values()
        ):
            raise ValueError(
                f"{kind} labels must be nonempty strings without null bytes"
            )
        maps[kind] = dict(labels)
        groups[kind] = items
        if kind == "sites":
            continue
        arity = {"bonds": 2, "angles": 3, "dihedrals": 4, "impropers": 4}[kind]
        for item in items:
            members = tuple(id(s) for s in item.connection_members)
            if (
                len(members) != arity
                or len(set(members)) != arity
                or any(s not in indices for s in members)
            ):
                raise ValueError(
                    f"{kind} have invalid arity, repeated members or foreign sites"
                )
    edges = {
        frozenset(id(s) for s in b.connection_members) for b in topology.bonds
    }
    template = PotentialTemplateLibrary()["PeriodicImproperPotential"]
    periodic = {
        p
        for p in maps["impropers"]
        if p is not None
        and p.independent_variables == template.independent_variables
        and sympy.simplify(p.expression - template.expression) == 0
    }
    routed = tuple(
        c for c in groups["impropers"] if c.connection_type in periodic
    )
    for connection in routed:
        center, *outer = (id(s) for s in connection.connection_members)
        if any(frozenset((center, s)) not in edges for s in outer):
            raise ValueError(
                "periodic impropers require a center-first star bond graph"
            )
    if any(p in maps["dihedrals"] for p in periodic):
        raise ValueError(
            "proper and improper potentials must have distinct objects"
        )
    maps["dihedrals"].update(
        (p, label) for p, label in maps["impropers"].items() if p in periodic
    )
    maps["impropers"] = {
        p: label for p, label in maps["impropers"].items() if p not in periodic
    }
    groups["dihedrals"] += routed
    groups["impropers"] = tuple(
        c for c in groups["impropers"] if c.connection_type not in periodic
    )
    for kind, labels in maps.items():
        if len(set(labels.values())) != len(labels):
            raise ValueError(f"colliding labels in physical {kind} block")
    return maps, groups


def _bare_array(value, shape, name):
    if (
        isinstance(value, u.unyt_array)
        or hasattr(value, "units")
        or hasattr(value, "unit")
    ):
        raise ValueError(f"{name} must be bare numeric values in nm")
    try:
        objects = np.asarray(value, dtype=object)
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must be a numeric array of shape {shape}"
        ) from error
    if (
        array.shape != shape
        or array.dtype.kind not in "iuf"
        or any(
            isinstance(x, (bool, np.bool_))
            or hasattr(x, "units")
            or hasattr(x, "unit")
            for x in objects.flat
        )
    ):
        raise ValueError(
            f"{name} must be a bare numeric array of shape {shape} without booleans"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(array, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite and within float64 range")
    return result


def create_all_atom_frame(
    topology, *, type_labels, positions_nm, box_lengths_nm
):
    """Return a native GSD frame using angstrom, amu and zero charge.

    Coordinates are bare origin-based nm values in site order. Box lengths are
    three bare positive nm values for an orthorhombic box. The frame stores
    float32 lengths L in angstrom before wrapping. Reconstruct origin-based
    angstrom coordinates as position + image*L + L/2, using float64 arithmetic.
    Reconstruction error is bounded by the float32 spacing toward zero of L plus
    8*eps64*abs(x), where x is the input coordinate in angstrom. The second
    term covers float64 arithmetic at large images. Extremely small boxes
    that cannot represent half-open wrapped bounds raise an error.
    Inputs and their arrays remain unchanged.

    This frame layout is preparatory. A force bundle must define all combined
    proper and periodic-improper labels before attaching to its dihedral block.
    The existing AllAtomDPD bundle does not yet fill those combined labels.
    """
    maps, groups = route_all_atom_connections(topology, type_labels=type_labels)
    count = topology.n_sites
    if count - 1 > np.iinfo(np.int32).max:
        raise ValueError("particle indices exceed the GSD int32 group range")
    if any(len(items) > np.iinfo(np.uint32).max for items in groups.values()):
        raise ValueError("group counts exceed the GSD uint32 range")
    xyz = _bare_array(positions_nm, (count, 3), "positions_nm")
    box = _bare_array(box_lengths_nm, (3,), "box_lengths_nm")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        lengths = (box * 10).astype(np.float32)
    if (
        np.any(box <= 0)
        or not np.all(np.isfinite(lengths))
        or np.any(lengths <= 0)
    ):
        raise ValueError(
            "box lengths must remain positive finite float32 angstrom values"
        )
    wide_lengths = lengths.astype(np.longdouble)
    with np.errstate(over="ignore", invalid="ignore"):
        coordinates = xyz.astype(np.longdouble) * 10
        images = np.floor(coordinates / wide_lengths)
        wrapped = (
            coordinates - images * wide_lengths - wide_lengths / 2
        ).astype(np.float32)
    high = wrapped.astype(np.longdouble) >= wide_lengths / 2
    wrapped = np.where(
        high, wrapped.astype(np.longdouble) - wide_lengths, wrapped
    ).astype(np.float32)
    images += high
    limit = np.iinfo(np.int32)
    if (
        not np.all(np.isfinite(images))
        or np.any(images < limit.min)
        or np.any(images > limit.max)
    ):
        raise ValueError("particle images exceed the int32 range")
    if (
        not np.all(np.isfinite(wrapped))
        or np.any(wrapped.astype(np.longdouble) < -wide_lengths / 2)
        or np.any(wrapped.astype(np.longdouble) >= wide_lengths / 2)
    ):
        raise ValueError(
            "wrapped positions cannot satisfy stored half-open box bounds"
        )
    reconstructed = (
        wrapped.astype(np.float64)
        + images.astype(np.float64) * lengths.astype(np.float64)
        + lengths.astype(np.float64) / 2
    )
    # Compute spacing in float64 so the largest finite float32 box does not
    # produce infinite spacing through a float32 nextafter overflow.
    spacing = lengths.astype(np.float64) - np.nextafter(
        lengths, np.float32(0)
    ).astype(np.float64)
    tolerance = spacing + 8 * np.finfo(np.float64).eps * np.abs(coordinates)
    if np.any(
        np.abs(reconstructed.astype(np.longdouble) - coordinates) > tolerance
    ):
        raise ValueError(
            "stored coordinates exceed the reconstruction tolerance"
        )
    masses = []
    for site in topology.sites:
        mass = site.mass
        if not isinstance(mass, u.unyt_array) or mass.shape != ():
            raise ValueError("sites require scalar physical masses in amu")
        try:
            value = float(mass.to_value("amu"))
        except (ValueError, u.exceptions.UnitConversionError) as error:
            raise ValueError("sites require physical masses in amu") from error
        with np.errstate(over="ignore", under="ignore"):
            stored = np.float32(value)
        if (
            not np.isfinite(value)
            or value <= 0
            or not np.isfinite(stored)
            or stored <= 0
        ):
            raise ValueError(
                "masses must remain positive finite float32 amu values"
            )
        masses.append(stored)
    frame = gsd.hoomd.Frame()
    frame.configuration.box = np.array([*lengths, 0, 0, 0], dtype=np.float32)
    frame.configuration.dimensions = 3
    frame.particles.N = count
    frame.particles.types = list(maps["sites"].values())
    site_type_ids = {p: i for i, p in enumerate(maps["sites"])}
    frame.particles.typeid = np.array(
        [site_type_ids[s.atom_type] for s in topology.sites], dtype=np.uint32
    )
    frame.particles.position = wrapped.copy()
    frame.particles.image = images.astype(np.int32)
    frame.particles.mass = np.array(masses, dtype=np.float32)
    frame.particles.charge = np.zeros(count, dtype=np.float32)
    indices = {id(site): i for i, site in enumerate(topology.sites)}
    for kind, arity in (
        ("bonds", 2),
        ("angles", 3),
        ("dihedrals", 4),
        ("impropers", 4),
    ):
        block = getattr(frame, kind)
        block.N = len(groups[kind])
        block.types = list(maps[kind].values())
        ids = {p: i for i, p in enumerate(maps[kind])}
        block.typeid = np.array(
            [ids[c.connection_type] for c in groups[kind]], dtype=np.uint32
        )
        block.group = np.array(
            [
                [indices[id(s)] for s in c.connection_members]
                for c in groups[kind]
            ],
            dtype=np.int32,
        ).reshape(-1, arity)
    frame.pairs.N = 0
    frame.pairs.types = []
    frame.pairs.typeid = np.empty(0, dtype=np.uint32)
    frame.pairs.group = np.empty((0, 2), dtype=np.int32)
    frame.constraints.N = 0
    frame.constraints.group = np.empty((0, 2), dtype=np.int32)
    frame.constraints.value = np.empty(0, dtype=np.float32)
    frame.validate()
    return frame
