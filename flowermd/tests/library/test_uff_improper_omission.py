"""Verify the supported UFF omission policy and independent OpenFF execution."""

from copy import deepcopy
from itertools import product

import hoomd
import numpy as np
import pytest
import unyt as u

from flowermd.internal.aa_snapshot import create_all_atom_frame
from flowermd.internal.uff_gmso import assign_uff_parameters
from flowermd.library import AllAtomDPD, UFFProvider
from flowermd.tests.library.test_aa_system import apply, create


def exclusions(frame):
    return {
        tuple(sorted((int(group[0]), int(group[-1]))))
        for kind in ("bonds", "angles", "dihedrals")
        for group in getattr(frame, kind).group
    }


@pytest.mark.parametrize(
    "bonded,weighting,enabled",
    product(["uff", UFFProvider()], [False, True], [False, True]),
)
def test_uff_provider_omits_candidates_without_overriding_flags(
    bonded, weighting, enabled
):
    system = create("C=O")
    raw = system._construction_topology
    indices = {s: i for i, s in enumerate(raw.sites)}
    expected_groups = tuple(
        tuple(indices[s] for s in c.connection_members) for c in raw.impropers
    )
    assert expected_groups
    native, _ = assign_uff_parameters(
        raw, system._molecule, atom_map=system._atom_map
    )
    apply(
        system,
        bonded=bonded,
        epsilon_weighting=weighting,
        include_impropers=enabled,
    )
    report = system.assignment_report
    assignment = report["assignment"]
    assert report["dpd"]["include_impropers"] is enabled
    assert assignment["include_impropers"] is False
    assert assignment["assigned_counts"]["impropers"] == 0
    assert assignment["retained_untyped_impropers"] == 0
    assert assignment["improper_force_backend"] == "unsupported"
    omission = assignment["improper_omission"]
    assert omission["inferred_candidate_groups"] == expected_groups
    assert omission["inferred_candidate_count"] == len(expected_groups)
    assert "intentionally omitted" in omission["reason"]
    assert (
        assignment["removed_unassigned_groups"]["impropers"] == expected_groups
    )
    assert report["execution"]["impropers"] == {
        "assigned_groups": 0,
        "executed_groups": 0,
        "force_objects": 0,
    }
    assert (
        system.gmso_system.n_impropers == system.hoomd_snapshot.impropers.N == 0
    )
    assert system.dpd_forcefield.forces_by_category["impropers"] == ()
    assert not any(
        isinstance(force, hoomd.md.force.Custom)
        for force in system.hoomd_forcefield
    )
    options = deepcopy(report["dpd"])
    options["include_impropers"] = False
    reference = AllAtomDPD(native, **options)
    assert (
        dict(reference.forces_by_category["pair"][0].params)
        == report["pair_coefficients"]
    )
    for kind in ("bonds", "angles", "dihedrals"):
        assert len(getattr(native, kind)) == len(
            getattr(system.gmso_system, kind)
        )
        for expected, actual in zip(
            getattr(native, kind), getattr(system.gmso_system, kind)
        ):
            for name, value in expected.connection_type.parameters.items():
                np.testing.assert_array_equal(
                    actual.connection_type.parameters[name], value
                )
    retained = create_all_atom_frame(
        native,
        type_labels=reference.type_labels,
        positions_nm=raw.positions.to_value("nm"),
        box_lengths_nm=raw.box.lengths.to_value("nm"),
    )
    assert retained.impropers.N == native.n_impropers > 0
    assert exclusions(retained) == exclusions(system.hoomd_snapshot)
    assert set(reference.forces_by_category["pair"][0].nlist.exclusions) == {
        "bond",
        "angle",
        "dihedral",
    }
    report["execution"]["impropers"]["assigned_groups"] = 99
    assert (
        system.assignment_report["execution"]["impropers"]["assigned_groups"]
        == 0
    )


@pytest.mark.parametrize("weighting", [False, True])
def test_default_uff_cpu_run_zero(weighting):
    system = create("C=O")
    apply(system, bonded="uff", epsilon_weighting=weighting)
    simulation = hoomd.Simulation(device=hoomd.device.CPU(), seed=7)
    simulation.create_state_from_snapshot(system.hoomd_snapshot)
    simulation.operations.integrator = hoomd.md.Integrator(
        dt=0.001,
        forces=system.hoomd_forcefield,
        methods=[hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())],
    )
    simulation.run(0)
    assert all(np.isfinite(force.energy) for force in system.hoomd_forcefield)


def test_restrained_candidate_failure_preserves_input_and_prepared_state(
    monkeypatch,
):
    system = create("C=O")
    apply(system, bonded="uff")
    previous = system.gmso_system, system.hoomd_snapshot, system.dpd_forcefield
    report = system.assignment_report
    original = system._construction_topology
    connection = original.impropers[0]
    # This GMSO version has no improper restraint field. Exercise the same
    # forward-compatible guard as the existing native assignment tests.
    restraint = {"phi0": 0 * u.rad, "kphi": 1 * u.kcal / u.mol / u.rad**2}
    monkeypatch.setattr(
        type(connection),
        "restraint",
        property(lambda self: restraint),
        raising=False,
    )
    before = deepcopy(connection.restraint)
    with pytest.raises(ValueError, match="restrained UFF improper"):
        apply(system, bonded=UFFProvider())
    assert connection.restraint == before and original.n_impropers > 0
    assert all(
        a is b
        for a, b in zip(
            previous,
            (system.gmso_system, system.hoomd_snapshot, system.dpd_forcefield),
        )
    )
    assert system.assignment_report == report


def test_typed_input_rejected_and_imported_uff_is_not_silently_dropped():
    class NativeUFF:
        def assign(self, topology, molecule, *, atom_map, assign_nonbonded):
            return assign_uff_parameters(topology, molecule, atom_map=atom_map)

    system = create("C=O")
    apply(system, bonded="uff")
    previous = system.dpd_forcefield
    with pytest.raises(ValueError, match="untyped"):
        UFFProvider().assign(
            system.gmso_system, system._molecule, atom_map=system._atom_map
        )
    with pytest.raises(NotImplementedError, match="UFF out-of-plane"):
        apply(system, bonded=NativeUFF())
    assert system.dpd_forcefield is previous
    apply(system, bonded=NativeUFF(), include_impropers=False)
    assert (
        system.gmso_system.n_impropers == system.hoomd_snapshot.impropers.N > 0
    )
    indices = {s: i for i, s in enumerate(system.gmso_system.sites)}
    np.testing.assert_array_equal(
        system.hoomd_snapshot.impropers.group,
        [
            tuple(indices[s] for s in c.connection_members)
            for c in system.gmso_system.impropers
        ],
    )
    assert system.assignment_report["execution"]["impropers"] == {
        "assigned_groups": system.gmso_system.n_impropers,
        "executed_groups": 0,
        "force_objects": 0,
    }


@pytest.mark.parametrize("enabled", [False, True])
def test_openff_periodic_groups_and_force_objects_have_separate_counts(enabled):
    system = create()
    apply(system, include_impropers=enabled)
    report = system.assignment_report
    assert "improper_omission" not in report["assignment"]
    for kind in ("bonds", "angles", "dihedrals", "impropers"):
        assigned = len(getattr(system.gmso_system, kind))
        execute = enabled if kind == "impropers" else True
        assert report["execution"][kind] == {
            "assigned_groups": assigned,
            "executed_groups": assigned if execute else 0,
            "force_objects": len(
                system.dpd_forcefield.forces_by_category[kind]
            ),
        }
    assert system.gmso_system.n_impropers > 0
    assert (
        report["execution"]["impropers"]["force_objects"]
        < system.gmso_system.n_impropers
    )


@pytest.mark.parametrize(
    "flag,category",
    [
        ("include_bonds", "bonds"),
        ("include_angles", "angles"),
        ("include_torsions", "dihedrals"),
    ],
)
def test_uff_other_ablations_preserve_assigned_groups(flag, category):
    system = create()
    apply(system, bonded="uff", epsilon_weighting=False, **{flag: False})
    summary = system.assignment_report["execution"]
    assert summary[category]["assigned_groups"] > 0
    assert (
        summary[category]["executed_groups"]
        == summary[category]["force_objects"]
        == 0
    )
    for other in {"bonds", "angles", "dihedrals"} - {category}:
        assert (
            summary[other]["assigned_groups"]
            == summary[other]["executed_groups"]
            > 0
        )
    assert summary["impropers"]["assigned_groups"] == 0
