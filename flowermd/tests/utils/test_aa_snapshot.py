"""Check all-atom GSD routing, units and periodic coordinate storage."""

from copy import deepcopy

import gmso
import gsd.hoomd
import hoomd
import numpy as np
import pytest
import unyt as u
from gmso.lib.potential_templates import PotentialTemplateLibrary

from flowermd.internal.aa_snapshot import (
    create_all_atom_frame,
    route_all_atom_connections,
)


def example():
    top = gmso.Topology()
    atoms = [gmso.AtomType(name="same", mass=m * u.amu) for m in (12, 1)]
    sites = [
        gmso.Atom(atom_type=atoms[i % 2], position=[i, 0, 0] * u.nm)
        for i in range(10)
    ]
    for site in sites:
        top.add_site(site)
    bond = gmso.BondType()
    angle = gmso.AngleType()
    proper = gmso.DihedralType()
    template = PotentialTemplateLibrary()["PeriodicImproperPotential"]
    periodic = [
        gmso.ImproperType.from_template(
            template,
            {
                "k": [1] * u.kcal / u.mol,
                "n": [2] * u.dimensionless,
                "phi_eq": [0.3] * u.rad,
            },
            name="same",
        )
        for _ in range(2)
    ]
    inversion = gmso.ImproperType(
        expression="k*(c0+c1*cos(omega)+c2*cos(2*omega))",
        independent_variables={"omega"},
        parameters={
            "k": 1 * u.kcal / u.mol,
            "c0": 1 * u.dimensionless,
            "c1": -1 * u.dimensionless,
            "c2": 0 * u.dimensionless,
        },
        tags={"form": "uff_inversion"},
    )
    expected = {
        "bonds": [
            (0, 1),
            (1, 2),
            (2, 3),
            (1, 4),
            (5, 6),
            (6, 7),
            (7, 8),
            (6, 9),
        ],
        "angles": [(0, 1, 2), (5, 6, 7)],
        "dihedrals": [(0, 1, 2, 3), (5, 6, 7, 8)],
        "impropers": [(1, 0, 2, 4), (6, 5, 7, 9), (1, 2, 4, 0), (6, 7, 9, 5)],
    }
    for kind, cls, potentials in (
        ("bonds", gmso.Bond, [bond] * 8),
        ("angles", gmso.Angle, [angle] * 2),
        ("dihedrals", gmso.Dihedral, [proper] * 2),
        (
            "impropers",
            gmso.Improper,
            [periodic[0], inversion, periodic[1], None],
        ),
    ):
        for group, potential in zip(expected[kind], potentials):
            top.add_connection(
                cls(
                    connection_members=[sites[i] for i in group],
                    **{
                        f"{kind[:-1] if kind != 'dihedrals' else 'dihedral'}_type": potential
                    },
                )
            )
    maps = {
        "sites": {atoms[1]: "H", atoms[0]: "C"},
        "bonds": {bond: "b"},
        "angles": {angle: "a"},
        "dihedrals": {proper: "p"},
        "impropers": {
            periodic[1]: "i1",
            None: "untyped",
            inversion: "uff",
            periodic[0]: "i0",
        },
    }
    return top, maps, expected


def frame(top, maps, positions=None, box=None):
    return create_all_atom_frame(
        top,
        type_labels=maps,
        positions_nm=np.zeros((top.n_sites, 3))
        if positions is None
        else positions,
        box_lengths_nm=[2, 3, 4] if box is None else box,
    )


def test_mixed_layout_exclusions_and_input_preservation(tmp_path):
    top, maps, expected = example()
    positions = np.arange(30).reshape(10, 3) / 3 - 2
    box = np.array([2.1, 3.2, 4.3])
    original_positions, original_box = positions.copy(), box.copy()
    original_top_positions = top.positions.copy()
    before = {kind: tuple(getattr(top, kind)) for kind in maps}
    map_before = {kind: list(values.items()) for kind, values in maps.items()}
    out = frame(top, maps, positions, box)
    assert out.particles.types == ["H", "C"]
    np.testing.assert_array_equal(out.particles.typeid, [1, 0] * 5)
    np.testing.assert_array_equal(out.particles.mass, [12, 1] * 5)
    np.testing.assert_array_equal(out.particles.charge, np.zeros(10))
    assert out.dihedrals.types == ["p", "i1", "i0"]
    assert out.impropers.types == ["untyped", "uff"]
    np.testing.assert_array_equal(out.dihedrals.typeid, [0, 0, 2, 1])
    np.testing.assert_array_equal(out.impropers.typeid, [1, 0])
    expected["dihedrals"] += [
        expected["impropers"][0],
        expected["impropers"][2],
    ]
    expected["impropers"] = [expected["impropers"][1], expected["impropers"][3]]
    for kind in expected:
        np.testing.assert_array_equal(getattr(out, kind).group, expected[kind])
    bonds = {frozenset(g) for g in expected["bonds"]}
    exclusions = bonds | {
        frozenset((g[0], g[-1]))
        for g in expected["angles"] + expected["dihedrals"][:2]
    }
    routed_exclusions = bonds | {
        frozenset((g[0], g[-1]))
        for g in expected["angles"] + expected["dihedrals"]
    }
    assert routed_exclusions == exclusions
    stored_box = out.configuration.box[:3].astype(float)
    rebuilt = (
        out.particles.position.astype(float)
        + out.particles.image * stored_box
        + stored_box / 2
    )
    np.testing.assert_allclose(
        rebuilt,
        positions * 10,
        rtol=0,
        atol=np.spacing(out.configuration.box[:3]).max(),
    )
    assert (out.particles.image < 0).any() and (out.particles.image > 0).any()
    np.testing.assert_array_equal(positions, original_positions)
    np.testing.assert_array_equal(box, original_box)
    np.testing.assert_array_equal(top.positions, original_top_positions)
    assert before == {kind: tuple(getattr(top, kind)) for kind in maps}
    assert map_before == {
        kind: list(values.items()) for kind, values in maps.items()
    }
    assert not np.shares_memory(positions, out.particles.position)
    assert not np.shares_memory(box, out.configuration.box)
    path = tmp_path / "frame.gsd"
    with gsd.hoomd.open(path, "w") as trajectory:
        trajectory.append(out)
    with gsd.hoomd.open(path, "r") as trajectory:
        loaded = trajectory[0]
    for field in ("box", "dimensions"):
        np.testing.assert_array_equal(
            getattr(loaded.configuration, field),
            getattr(out.configuration, field),
        )
    for field in (
        "N",
        "types",
        "typeid",
        "position",
        "image",
        "mass",
        "charge",
    ):
        np.testing.assert_array_equal(
            getattr(loaded.particles, field), getattr(out.particles, field)
        )
    for kind in expected:
        for field in ("N", "types", "typeid", "group"):
            np.testing.assert_array_equal(
                getattr(getattr(loaded, kind), field),
                getattr(getattr(out, kind), field),
            )
    for kind, fields in (
        ("pairs", ("N", "types", "typeid", "group")),
        ("constraints", ("N", "group", "value")),
    ):
        for field in fields:
            np.testing.assert_array_equal(
                getattr(getattr(loaded, kind), field),
                getattr(getattr(out, kind), field),
            )
    simulation = hoomd.Simulation(device=hoomd.device.CPU(), seed=3)
    simulation.create_state_from_gsd(str(path))
    snapshot = simulation.state.get_snapshot()
    np.testing.assert_array_equal(snapshot.dihedrals.group, out.dihedrals.group)
    np.testing.assert_array_equal(snapshot.impropers.group, out.impropers.group)


@pytest.mark.parametrize(
    "coordinate",
    [
        0,
        2,
        -2,
        np.nextafter(2.0, 0),
        np.nextafter(2.0, 3),
        np.nextafter(0.0, -1),
    ],
)
def test_faces_and_adjacent_values(coordinate):
    top, maps, _ = example()
    out = frame(top, maps, np.full((10, 3), coordinate), [2, 2, 2])
    wrapped = out.particles.position.astype(float)
    assert np.all(wrapped >= -10) and np.all(wrapped < 10)
    reconstructed = wrapped + out.particles.image.astype(float) * 20 + 10
    np.testing.assert_allclose(
        reconstructed, coordinate * 10, rtol=0, atol=np.spacing(np.float32(20))
    )


@pytest.mark.parametrize("image", [-(2**31), 2**31 - 1])
def test_int32_endpoints(image):
    top, maps, _ = example()
    out = frame(top, maps, np.full((10, 3), image), [1] * 3)
    np.testing.assert_array_equal(out.particles.image, image)
    stored_box = out.configuration.box[:3].astype(float)
    rebuilt = (
        out.particles.position.astype(float)
        + out.particles.image * stored_box
        + stored_box / 2
    )
    np.testing.assert_array_equal(rebuilt, image * 10)


@pytest.mark.parametrize("coordinate", [2**31 / 10, (-(2**31) - 1) / 10, 1e308])
def test_image_overflow(coordinate):
    top, maps, _ = example()
    with pytest.raises(ValueError, match="int32"):
        frame(top, maps, np.full((10, 3), coordinate), [0.1] * 3)


@pytest.mark.parametrize(
    "box",
    [
        [0, 1, 1],
        [-1, 1, 1],
        [1e308] * 3,
        [1e-300] * 3,
        [float("nan")] * 3,
        [1, 2],
        [True, 2, 3],
        [1, 2, 3] * u.nm,
    ],
)
def test_invalid_boxes(box):
    top, maps, _ = example()
    with pytest.raises(ValueError):
        frame(top, maps, box=box)


@pytest.mark.parametrize(
    "value", [0, -1, 1e-300, 1e300, float("nan"), float("inf")]
)
def test_invalid_masses(value):
    top, maps, _ = example()
    top.sites[0].mass = value * u.amu
    with pytest.raises(ValueError, match="mass"):
        frame(top, maps)


@pytest.mark.parametrize(
    "positions",
    [
        np.zeros((3, 10)),
        np.zeros((10, 3)) * u.nm,
        np.full((10, 3), True),
        np.full((10, 3), float("inf")),
        [[0, False, 0]] * 10,
    ],
)
def test_invalid_coordinates(positions):
    top, maps, _ = example()
    with pytest.raises(ValueError, match="positions_nm"):
        frame(top, maps, positions)


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "extra",
        "stale",
        "collision",
        "cross_collision",
        "null",
        "empty",
        "none",
    ],
)
def test_invalid_maps(change):
    top, maps, _ = example()
    if change == "missing":
        maps.pop("angles")
    elif change == "extra":
        maps["sites"][gmso.AtomType()] = "extra"
    elif change == "stale":
        p, label = maps["sites"].popitem()
        maps["sites"][deepcopy(p)] = label
    elif change == "collision":
        maps["sites"] = dict.fromkeys(maps["sites"], "same")
    elif change == "cross_collision":
        p = next(p for p, label in maps["impropers"].items() if label == "i0")
        maps["impropers"][p] = "p"
    elif change in ("null", "empty"):
        maps["bonds"][next(iter(maps["bonds"]))] = (
            "\x00" if change == "null" else ""
        )
    else:
        for site in top.sites:
            site.atom_type = None
        maps["sites"] = {None: "none"}
    with pytest.raises(ValueError):
        frame(top, maps)


def test_invalid_star_and_members():
    top, maps, _ = example()
    top.impropers[0].connection_members = [top.sites[i] for i in (0, 1, 2, 4)]
    with pytest.raises(ValueError, match="star"):
        frame(top, maps)
    top, maps, _ = example()
    # Bypass GMSO assignment validation to exercise defensive frame validation.
    top.bonds[0].__dict__["connection_members_"] = (top.sites[0], gmso.Atom())
    with pytest.raises(ValueError, match="foreign sites"):
        frame(top, maps)


def test_empty_blocks():
    top, maps, _ = example()
    for kind in ("bonds", "angles", "dihedrals", "impropers"):
        for connection in tuple(getattr(top, kind)):
            top.remove_connection(connection)
        maps[kind] = {}
    out = frame(top, maps)
    for kind, arity in (
        ("bonds", 2),
        ("angles", 3),
        ("dihedrals", 4),
        ("impropers", 4),
    ):
        block = getattr(out, kind)
        assert block.N == 0 and block.types == []
        assert block.group.shape == (0, arity)
        assert block.group.dtype == np.int32 and block.typeid.dtype == np.uint32
    physical, groups = route_all_atom_connections(top, type_labels=maps)
    physical["sites"].clear()
    assert maps["sites"] and groups["sites"] == tuple(top.sites)


@pytest.mark.parametrize("scale", [1e-42, 1e-5, 1, 1e30, 1e36])
def test_stored_box_controls_wrapping_across_float32_range(scale):
    top, maps, _ = example()
    box = np.array([1.23456789, 2.34567891, 3.45678912]) * scale
    lengths = (box * 10).astype(np.float32).astype(float)
    rng = np.random.default_rng(94)
    cells = rng.integers(-(2**30), 2**30, size=(10, 3))
    fractions = rng.uniform(0.2, 0.8, size=(10, 3))
    xyz = (cells + fractions) * lengths / 10
    out = frame(top, maps, xyz, box)
    np.testing.assert_array_equal(out.configuration.box[:3], lengths)
    np.testing.assert_array_equal(out.particles.image, cells)
    wrapped = out.particles.position.astype(float)
    assert np.all(wrapped >= -lengths / 2) and np.all(wrapped < lengths / 2)
    reconstructed = wrapped + cells * lengths + lengths / 2
    # A conservative independent bound combines stored position precision
    # with float64 multiplication and addition at extreme image counts.
    bound = (
        lengths * np.finfo(np.float32).eps
        + np.finfo(np.float64).eps * np.abs(xyz * 10) * 8
        + np.nextafter(np.float32(0), np.float32(1))
    )
    assert np.all(np.abs(reconstructed - xyz * 10) <= bound)


@pytest.mark.parametrize("members", [(0,), (0, 0), (0, 1, 2)])
def test_invalid_group_arity_and_repeated_members(members):
    top, maps, _ = example()
    top.bonds[0].__dict__["connection_members_"] = tuple(
        top.sites[i] for i in members
    )
    with pytest.raises(ValueError, match="arity, repeated members"):
        frame(top, maps)


def test_empty_topology_rejected():
    with pytest.raises(ValueError, match="nonempty"):
        create_all_atom_frame(
            gmso.Topology(),
            type_labels={},
            positions_nm=[],
            box_lengths_nm=[1, 1, 1],
        )


def test_rounding_correction_checks_final_image_range():
    top, maps, _ = example()
    # L=2 angstrom has a positive half face of 1. The residual below that
    # face rounds upward to 1, so wrapping must increment the image.
    coordinate = np.nextafter(0.2, 0)
    out = frame(top, maps, np.full((10, 3), coordinate), [0.2] * 3)
    np.testing.assert_array_equal(out.particles.image, 1)
    np.testing.assert_array_equal(out.particles.position, -1)
