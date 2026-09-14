"""Literal numerical protocol cases, without simulation or force execution."""

from copy import deepcopy

import pytest

from flowermd.internal.aa_dpd import (
    _NUMERICAL_ROLES,
    _aggregate_monitor_energies,
    _dpd_stopping_decision,
    _fire_stopping_decision,
    _numerical_protocol,
    _sample_schedule,
    _stationarity,
    _validate_numerical_protocol,
)


def variant(**settings):
    result = _numerical_protocol("corrected-uff-v1")
    result.update(
        id="test-variant-v1",
        kind="variant",
        description="Explicit synthetic numerical variant",
        epsilon_source="selected-provider maximum assigned epsilon",
    )
    result.update(settings)
    return result


def counts(**roles):
    result = {
        role: dict(semantic=0, components=0, executed=0, force_objects=0)
        for role in _NUMERICAL_ROLES
    }
    for role, values in roles.items():
        result[role].update(values)
    return result


def test_frozen_records_and_copy():
    uff = _numerical_protocol("corrected-uff-v1")
    assert (
        uff["A"],
        uff["gamma"],
        uff["kT"],
        uff["dt"],
        uff["bonded_scale"],
        uff["cutoff_angstrom"],
    ) == (1250, 200, 1, 0.001, 30, 3.5)
    assert uff["stereo_bias_kcal_mol"] == 30000
    assert uff["stereo_scaled"] is False
    assert uff["final_stereo_audit"] is True
    assert uff["planar_threshold"] == 0.05
    assert uff["enabled"]["improper"] is False
    assert uff["epsilon_source"] == "UFF maximum assigned epsilon"
    assert uff["runnable"] is False
    assert uff["fire"] == dict(
        steps=100,
        dt=0.001,
        force_tol=1000,
        energy_tol=1000,
        angmom_tol=1000,
        mode="fixed",
    )
    copied = _validate_numerical_protocol(uff)
    copied["enabled"]["proper"] = False
    assert uff["enabled"]["proper"] is True
    with pytest.raises(ValueError, match="variant ID"):
        _validate_numerical_protocol(copied)
    sage = _numerical_protocol("sage30-v1")
    assert sage["normalization"] == "particles"
    assert sage["epsilon_source"] == uff["epsilon_source"]
    assert sage["enabled"]["improper"] is True
    assert sage["torsion_channel"] == "combined"
    charged = _numerical_protocol("sage30-charged-replay-v1")
    assert charged["coulomb_floor"] == 1
    assert charged["coulomb_model"] == dict(
        cap_kcal_mol=5, switch_on_angstrom=7, cutoff_angstrom=9
    )
    assert charged["runnable"] is False


@pytest.mark.parametrize(
    "provider,resource",
    [("UFF", "RDKit UFF"), ("OpenFF", "openff-2.3.0.offxml")],
)
@pytest.mark.parametrize(
    "epsilon", ["unweighted", "selected-provider maximum assigned epsilon"]
)
def test_selected_provider_variants(provider, resource, epsilon):
    p = variant(provider=provider, resource=resource, epsilon_source=epsilon)
    if provider == "OpenFF":
        p["angle_method"] = "OpenFF harmonic"
    assert _validate_numerical_protocol(p) == p
    p["id"] = "sage30-v1"
    with pytest.raises(ValueError):
        _validate_numerical_protocol(p)


def test_literal_samples_and_earliest_stop():
    p = _numerical_protocol("corrected-uff-v1")
    bulk, windows = _sample_schedule(p)
    assert bulk == 3500
    assert windows[:3] == (
        (3600, 3700, 3800, 3900, 4000),
        (4100, 4200, 4300, 4400, 4500),
        (4600, 4700, 4800, 4900, 5000),
    )
    previous, stable = None, 0
    decisions = []
    for window in windows[:3]:
        current = {"pair": 100.0}
        _, stable = _stationarity(previous, current, stable=stable)
        decisions.append(
            _dpd_stopping_decision(p, completed=window[-1], stable=stable)
        )
        previous = current
    assert decisions == ["continue", "continue", "converged"]
    assert _sample_schedule(_numerical_protocol("sage30-v1")) == (bulk, windows)
    assert _sample_schedule(
        variant(minimum=5, interval=2, samples=5, cap=6)
    ) == (3, ((4, 5), (6,)))
    assert _sample_schedule(
        variant(minimum=7, interval=7, samples=3, cap=7)
    ) == (0, ((3, 5, 7),))
    assert _sample_schedule(
        variant(minimum=2, interval=2, samples=5, cap=2)
    ) == (0, ((1, 2),))


def test_reset_and_cap_order():
    stable = 0
    for old, new, expected in (
        (100, 100, 1),
        (100, 50, 0),
        (50, 50, 1),
        (50, 50, 2),
    ):
        _, stable = _stationarity({"pair": old}, {"pair": new}, stable=stable)
        assert stable == expected
    assert (
        _dpd_stopping_decision(variant(cap=4500), completed=4500, stable=1)
        == "cap_failure"
    )
    assert (
        _dpd_stopping_decision(variant(cap=5000), completed=5000, stable=2)
        == "converged"
    )
    assert (
        _dpd_stopping_decision(variant(cap=5000), completed=5000, stable=0)
        == "cap_failure"
    )
    assert (
        _dpd_stopping_decision(
            variant(cap=5000, require_convergence=False),
            completed=5000,
            stable=0,
        )
        == "cap_allowed"
    )


@pytest.mark.parametrize(
    "old,new,passes",
    [
        (49, 50, True),
        (48.9, 50, False),
        (0, 0, True),
        (-1, 1, False),
        (0, 1e-310, True),
        (-1e308, 1e308, False),
    ],
)
def test_relative_boundaries(old, new, passes):
    changes, stable = _stationarity({"pair": old}, {"pair": new})
    assert (stable == 1) is passes
    assert (changes["pair"] <= 0.02) is passes


@pytest.mark.parametrize(
    "old,new",
    [
        ({"a": 1}, {"b": 1}),
        ({"a": float("nan")}, {"a": 1}),
        ({"a": 1}, {"a": float("inf")}),
    ],
)
def test_invalid_history(old, new):
    with pytest.raises(ValueError):
        _stationarity(old, new)


def test_coulomb_floor():
    assert (
        _stationarity({"coulomb": 0}, {"coulomb": 0.015}, coulomb_floor=1)[1]
        == 1
    )
    assert _stationarity({"pair": 0}, {"pair": 0.015}, coulomb_floor=1)[1] == 0
    assert (
        _stationarity({"coulomb": 0}, {"coulomb": 0.021}, coulomb_floor=1)[1]
        == 0
    )


def test_semantic_aggregation_components_and_padding():
    c = counts(
        pair=dict(force_objects=1, executed=1),
        bond=dict(semantic=2, executed=2, force_objects=1),
        proper=dict(semantic=1, components=3, executed=1, force_objects=2),
        improper=dict(semantic=1, components=2, executed=1, force_objects=1),
    )
    energies = [
        ("pair", [20]),
        ("bond", [20]),
        ("proper", [2, 4]),
        ("improper", [4]),
    ]
    sage = _numerical_protocol("sage30-v1")
    means, copied = _aggregate_monitor_energies(
        sage, energies, particles=10, counts=c
    )
    assert means == dict(pair=2, bond=2, torsion=1)
    assert copied == c and copied is not c
    shared = _numerical_protocol("shared-sage-components-v1")
    assert _aggregate_monitor_energies(
        shared, energies, particles=10, counts=c
    )[0] == dict(pair=2, bond=10, torsion=2)
    # Three proper components include an assigned k=0 term. Padding belongs only
    # to the backend force count and never increases assigned components.
    padded = deepcopy(c)
    padded["proper"]["force_objects"] = 3
    extra = [
        ("pair", [20]),
        ("bond", [20]),
        ("proper", [2, 4, 0]),
        ("improper", [4]),
    ]
    assert (
        _aggregate_monitor_energies(shared, extra, particles=10, counts=padded)[
            0
        ]["torsion"]
        == 2
    )
    padded["proper"]["components"] = 2
    assert (
        _aggregate_monitor_energies(shared, extra, particles=10, counts=padded)[
            0
        ]["torsion"]
        == 2.5
    )
    c["improper"].update(executed=0, force_objects=0)
    native = _numerical_protocol("corrected-uff-v1")
    out, kept = _aggregate_monitor_energies(
        native, energies[:-1], particles=10, counts=c
    )
    assert out == dict(pair=2, bond=10, proper=6)
    assert kept["improper"]["semantic"] == 1
    assert "improper" not in out
    c["bond"]["semantic"] = 0
    assert (
        _aggregate_monitor_energies(
            native, energies[:-1], particles=10, counts=c
        )[0]["bond"]
        == 20
    )


@pytest.mark.parametrize("disabled", ["bond", "angle", "proper", "improper"])
def test_independent_ablations(disabled):
    p = variant(
        provider="OpenFF",
        resource="openff-2.3.0.offxml",
        angle_method="OpenFF harmonic",
    )
    p["enabled"]["improper"] = True
    p["enabled"][disabled] = False
    c = counts(
        **{
            role: dict(
                semantic=3,
                components=4,
                executed=0 if role == disabled else 3,
                force_objects=0 if role == disabled else 1,
            )
            for role in ("bond", "angle", "proper", "improper")
        }
    )
    raw = [
        (role, [0 if role == "proper" else 6])
        for role in ("bond", "angle", "proper", "improper")
        if role != disabled
    ]
    means, kept = _aggregate_monitor_energies(p, raw, particles=10, counts=c)
    assert disabled not in means and kept[disabled]["semantic"] == 3
    if disabled != "proper":
        assert means["proper"] == 0
    if disabled != "improper":
        assert means["improper"] == 2


@pytest.mark.parametrize("bad", [True, -1, 1.5])
def test_invalid_counts(bad):
    c = counts(bond=dict(semantic=bad))
    with pytest.raises(ValueError):
        _aggregate_monitor_energies(
            _numerical_protocol("corrected-uff-v1"), [], particles=10, counts=c
        )
    with pytest.raises(ValueError):
        _sample_schedule(variant(samples=bad))


@pytest.mark.parametrize(
    "raw",
    [
        [],
        [("pair", [1]), ("pair", [1])],
        [("bond", [1])],
        [("pair", [float("nan")])],
    ],
)
def test_missing_extra_duplicate_invalid_energy(raw):
    with pytest.raises(ValueError):
        _aggregate_monitor_energies(
            _numerical_protocol("corrected-uff-v1"),
            raw,
            particles=10,
            counts=counts(pair=dict(force_objects=1)),
        )


def test_fire_fixed_and_driven_gates():
    assert _fire_stopping_decision(100, False) == "fixed_steps_complete"
    assert _fire_stopping_decision(99, True) == "continue"
    with pytest.raises(ValueError, match="exceed cap"):
        _fire_stopping_decision(
            0, False, initial_steps=101, convergence_driven=True, cap=100
        )
    assert (
        _fire_stopping_decision(100, False, convergence_driven=True, cap=100)
        == "cap_failure"
    )
    assert (
        _fire_stopping_decision(100, True, convergence_driven=True, cap=100)
        == "converged"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", True),
        ("minimum", True),
        ("dt", float("nan")),
        ("runnable", True),
        ("gamma", True),
    ],
)
def test_frozen_records_reject_invalid_values(field, value):
    record = _numerical_protocol("corrected-uff-v1")
    record[field] = value
    with pytest.raises(ValueError):
        _validate_numerical_protocol(record)


@pytest.mark.parametrize("particles", [0, True, -1, 1.5])
def test_native_particle_count_is_positive(particles):
    with pytest.raises(ValueError):
        _aggregate_monitor_energies(
            _numerical_protocol("corrected-uff-v1"),
            [],
            particles=particles,
            counts=counts(),
        )


def test_provider_method_contradictions_are_rejected():
    p = variant()
    p["enabled"]["improper"] = True
    with pytest.raises(ValueError, match="UFF variants omit"):
        _validate_numerical_protocol(p)
    p = variant(provider="OpenFF", resource="openff-2.3.0.offxml")
    with pytest.raises(ValueError, match="angle method"):
        _validate_numerical_protocol(p)
