"""Internal coefficient weighting and numerical protocols for all-atom DPD."""

import math
from collections.abc import Mapping, Sequence
from numbers import Real


def dpd_pair_parameters(
    particle_types,
    repulsion,
    gamma,
    *,
    epsilon_weighting=True,
    particle_epsilons=None,
    epsilon_reference=None,
):
    """Construct uniform or epsilon-weighted all-atom DPD coefficients.

    Parameters
    ----------
    particle_types : sequence of str
        Nonempty ordered sequence of unique, nonempty particle type names.
    repulsion, gamma : float
        Finite, nonnegative nominal DPD coefficients.
    epsilon_weighting : bool, optional
        If True, the default, weight both coefficients by the geometric-mean
        epsilon factor. If False, use the nominal coefficients for every pair.
    particle_epsilons : mapping of str to float, optional
        Required when weighting is enabled, with exactly the particle type
        keys and nonnegative finite epsilon values in one common energy unit.
        The caller supplies the epsilon values. This function only uses numbers.
        Omit when weighting is disabled.
    epsilon_reference : float, optional
        Positive reference in the same energy unit. Defaults to the maximum
        supplied epsilon. Omit when weighting is disabled.

    Returns
    -------
    dict
        Unordered type pairs, including self pairs, in particle-type order,
        each mapping to ``A`` and ``gamma``. Inputs are not modified.

    Notes
    -----
    Uniform mode needs no epsilon inputs or force-field parameter extraction.
    Supplying epsilon arguments in that mode is an error. This function does
    not construct forces, assign UFF or Sage parameters, or convert units.
    In weighted mode, a zero epsilon makes both coefficients zero for every
    pair containing that type. See :func:`epsilon_scaled_dpd_parameters`.
    """
    if not isinstance(epsilon_weighting, bool):
        raise ValueError("epsilon_weighting must be a bool")
    if (
        isinstance(particle_types, (str, bytes))
        or not isinstance(particle_types, Sequence)
        or not particle_types
    ):
        raise ValueError("particle_types must be a nonempty ordered sequence")
    names = tuple(particle_types)
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("particle type names must be nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError("particle type names must be unique")
    repulsion = _finite_coefficient(repulsion, "repulsion", allow_zero=True)
    gamma = _finite_coefficient(gamma, "gamma", allow_zero=True)
    if not epsilon_weighting:
        if particle_epsilons is not None or epsilon_reference is not None:
            raise ValueError(
                "omit epsilon arguments when weighting is disabled"
            )
        return {
            (first, second): {"A": repulsion, "gamma": gamma}
            for index, first in enumerate(names)
            for second in names[index:]
        }
    if not isinstance(particle_epsilons, Mapping):
        raise ValueError("particle_epsilons must be a mapping when weighting")
    if set(particle_epsilons) != set(names):
        raise ValueError("epsilon keys must exactly match particle_types")
    return epsilon_scaled_dpd_parameters(
        {name: particle_epsilons[name] for name in names},
        repulsion,
        gamma,
        epsilon_reference,
    )


def epsilon_scaled_dpd_parameters(
    particle_epsilons, repulsion, gamma, epsilon_reference=None
):
    """Weight DPD coefficients by geometric-mean particle epsilons.

    Parameters
    ----------
    particle_epsilons : mapping of str to float
        Nonnegative, finite epsilon magnitudes in one common energy unit.
        Mapping order determines the order of the returned unordered pairs.
    repulsion, gamma : float
        Finite, nonnegative reference DPD coefficients.
    epsilon_reference : float, optional
        Positive, finite reference in the same energy unit. Defaults to the
        maximum supplied epsilon. Pass a reference explicitly to reproduce a
        protocol that used a different value. This includes protocols that
        used the maximum of only overridden epsilons.
        All-zero epsilons require an explicit positive reference.

    Returns
    -------
    dict
        One entry per unordered type pair, including self pairs, containing
        ``A`` and ``gamma`` multiplied by ``sqrt(epsilon_i * epsilon_j) /
        epsilon_reference``.

    Notes
    -----
    This function weights coefficients. It does not construct forces or add
    Lennard-Jones or Coulomb interactions. It takes numeric magnitudes and
    does not convert units. For nonzero coefficients and epsilons, both
    normalized epsilon ratios, their geometric-mean factor and the final
    coefficient must fit in float range.
    An intermediate outside that range raises ValueError even if the final
    mathematical result would be representable.
    A zero epsilon produces exactly zero A and gamma for pairs containing
    that type, disabling their conservative repulsion and DPD thermostat
    coupling. The function applies no epsilon floor or fallback.
    """
    if not isinstance(particle_epsilons, Mapping) or not particle_epsilons:
        raise ValueError("particle_epsilons must be a nonempty mapping")
    epsilons = {}
    for name, epsilon in particle_epsilons.items():
        if not isinstance(name, str) or not name:
            raise ValueError("particle type names must be nonempty strings")
        epsilons[name] = _finite_coefficient(
            epsilon, f"epsilon for {name}", allow_zero=True
        )
    repulsion = _finite_coefficient(repulsion, "repulsion", allow_zero=True)
    gamma = _finite_coefficient(gamma, "gamma", allow_zero=True)
    reference = (
        _finite_coefficient(max(epsilons.values()), "epsilon_reference")
        if epsilon_reference is None
        else _finite_coefficient(epsilon_reference, "epsilon_reference")
    )
    pairs = {}
    names = list(epsilons)
    for index, first in enumerate(names):
        for second in names[index:]:
            pairs[first, second] = {
                "A": _weighted_coefficient(
                    repulsion, epsilons[first], epsilons[second], reference
                ),
                "gamma": _weighted_coefficient(
                    gamma, epsilons[first], epsilons[second], reference
                ),
            }
    return pairs


def _finite_coefficient(value, name, allow_zero=False):
    condition = "nonnegative" if allow_zero else "positive"
    message = f"{name} must be a finite {condition} real number"
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(message)
    try:
        value = float(value)
    except (OverflowError, ValueError):
        raise ValueError(message) from None
    if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise ValueError(message)
    return value


def _weighted_coefficient(coefficient, first, second, reference):
    if coefficient == 0 or first == 0 or second == 0:
        return 0.0
    first_ratio = first / reference
    second_ratio = second / reference
    for value in (first_ratio, second_ratio):
        if not math.isfinite(value):
            raise ValueError("weighted DPD coefficient exceeds float range")
        if value == 0:
            raise ValueError("weighted DPD coefficient underflows float range")
    factor = math.sqrt(first_ratio) * math.sqrt(second_ratio)
    if not math.isfinite(factor):
        raise ValueError("weighted DPD coefficient exceeds float range")
    if factor == 0:
        raise ValueError("weighted DPD coefficient underflows float range")
    result = coefficient * factor
    if not math.isfinite(result):
        raise ValueError("weighted DPD coefficient exceeds float range")
    if result == 0:
        raise ValueError("weighted DPD coefficient underflows float range")
    return result


# These records specify numerical replay, not an executable dynamics controller.
_NUMERICAL_ROLES = (
    "pair",
    "bond",
    "angle",
    "proper",
    "improper",
    "stereo",
    "coulomb",
)


def _numerical_protocol(name):
    """Return an independent copy of a frozen version-1 numerical record."""
    from copy import deepcopy

    record = {
        "id": name,
        "version": 1,
        "kind": "frozen",
        "description": name,
        "provider": "UFF",
        "resource": "RDKit UFF",
        "epsilon_source": "UFF maximum assigned epsilon",
        "A": 1250.0,
        "gamma": 200.0,
        "kT": 1.0,
        "dt": 0.001,
        "bonded_scale": 30.0,
        "cutoff_angstrom": 3.5,
        "angle_method": "frozen local-curvature harmonic UFF surrogate",
        "stereo_bias_kcal_mol": 30000.0,
        "stereo_scaled": False,
        "final_stereo_audit": True,
        "planar_threshold": 0.05,
        "enabled": dict(
            zip(_NUMERICAL_ROLES, (True, True, True, True, False, True, False))
        ),
        "normalization": "interactions",
        "torsion_channel": "separate",
        "schedule": "shared",
        "minimum": 4000,
        "interval": 500,
        "samples": 5,
        "cap": 40000,
        "tolerance": 0.02,
        "comparisons": 2,
        "require_convergence": True,
        "coulomb_floor": 0.0,
        "coulomb_model": None,
        "fire": {
            "steps": 100,
            "dt": 0.001,
            "force_tol": 1000.0,
            "energy_tol": 1000.0,
            "angmom_tol": 1000.0,
            "mode": "fixed",
        },
        "runnable": False,
    }
    if name in (
        "sage30-v1",
        "sage30-charged-replay-v1",
        "shared-sage-components-v1",
    ):
        record.update(
            provider="OpenFF",
            resource="openff-2.3.0.offxml",
            angle_method="OpenFF harmonic",
            torsion_channel="combined",
            normalization="particles",
            schedule="sage30",
        )
        record["enabled"]["improper"] = True
        if name == "shared-sage-components-v1":
            record.update(
                normalization="components",
                schedule="shared",
                kind="compatibility",
            )
        if name == "sage30-charged-replay-v1":
            record["enabled"]["coulomb"] = True
            record["coulomb_floor"] = 1.0
            record["coulomb_model"] = {
                "cap_kcal_mol": 5.0,
                "switch_on_angstrom": 7.0,
                "cutoff_angstrom": 9.0,
            }
    elif name != "corrected-uff-v1":
        raise ValueError("unknown numerical protocol")
    return deepcopy(record)


def _validate_numerical_protocol(record):
    """Copy a frozen record or an explicitly named selected-provider variant.

    Variants retain this schema and must state a new ID and description. They
    use selected-provider epsilon weighting or unweighted pairs, independent
    proper/improper ablations, and the shared sample rule. All records remain
    numerical specifications only. ``enabled`` describes execution roles,
    not requested public ``include_*`` flags. A later adapter must derive roles
    and counts from the bundle's execution summary. UFF's public request for
    impropers can remain True while omitted groups produce no execution role.
    """
    from copy import deepcopy

    if not isinstance(record, Mapping):
        raise ValueError("protocol must be a mapping")
    result = deepcopy(dict(record))
    baseline = _numerical_protocol("corrected-uff-v1")
    if result.keys() != baseline.keys():
        raise ValueError("protocol fields must match the version-1 schema")
    for key in (
        "minimum",
        "interval",
        "samples",
        "cap",
        "comparisons",
        "version",
    ):
        _numerical_count(result[key], key, positive=True)
    if not isinstance(result["enabled"], Mapping) or any(
        type(value) is not bool for value in result["enabled"].values()
    ):
        raise ValueError("role switches must be boolean")
    for key in (
        "runnable",
        "stereo_scaled",
        "final_stereo_audit",
        "require_convergence",
    ):
        if type(result[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    for key in (
        "A",
        "gamma",
        "kT",
        "dt",
        "bonded_scale",
        "cutoff_angstrom",
        "stereo_bias_kcal_mol",
        "planar_threshold",
        "tolerance",
        "coulomb_floor",
    ):
        _finite_coefficient(result[key], key, allow_zero=True)
    frozen = (
        "corrected-uff-v1",
        "sage30-v1",
        "sage30-charged-replay-v1",
        "shared-sage-components-v1",
    )
    if result["id"] in frozen:
        if result != _numerical_protocol(result["id"]):
            raise ValueError("changed settings require a distinct variant ID")
        return result
    if (
        result["kind"] != "variant"
        or not isinstance(result["id"], str)
        or not result["id"]
        or not isinstance(result["description"], str)
        or not result["description"]
    ):
        raise ValueError("variants require an explicit ID and description")
    if (
        type(result["version"]) is not int
        or result["version"] != 1
        or result["runnable"] is not False
    ):
        raise ValueError("only version-1 numerical records are supported")
    if (
        result["provider"] not in ("UFF", "OpenFF")
        or not isinstance(result["resource"], str)
        or not result["resource"]
    ):
        raise ValueError("record the selected provider and resource")
    if result["epsilon_source"] not in (
        "selected-provider maximum assigned epsilon",
        "unweighted",
    ):
        raise ValueError(
            "variant epsilon source must describe the selected provider or unweighted mode"
        )
    if (
        result["normalization"] != "interactions"
        or result["torsion_channel"] != "separate"
        or result["schedule"] != "shared"
    ):
        raise ValueError(
            "selected-provider variants use independent channels and shared scheduling"
        )
    if (
        not isinstance(result["enabled"], Mapping)
        or set(result["enabled"]) != set(_NUMERICAL_ROLES)
        or any(type(v) is not bool for v in result["enabled"].values())
    ):
        raise ValueError("declare every semantic role with a boolean")
    if (
        result["enabled"]["coulomb"]
        or result["coulomb_model"] is not None
        or result["coulomb_floor"] != 0
    ):
        raise ValueError("charged force-model variants are not available")
    for key in ("minimum", "interval", "samples", "cap", "comparisons"):
        _numerical_count(result[key], key, positive=True)
    if result["minimum"] > result["cap"]:
        raise ValueError("minimum must not exceed cap")
    for key in (
        "A",
        "gamma",
        "kT",
        "dt",
        "bonded_scale",
        "cutoff_angstrom",
        "stereo_bias_kcal_mol",
        "planar_threshold",
        "tolerance",
    ):
        _finite_coefficient(
            result[key],
            key,
            allow_zero=key
            in ("A", "gamma", "kT", "stereo_bias_kcal_mol", "tolerance"),
        )
    if (
        type(result["require_convergence"]) is not bool
        or result["stereo_scaled"] is not False
        or result["final_stereo_audit"] is not True
    ):
        raise ValueError(
            "declare convergence and preserve the stereo audit contract"
        )
    if result["fire"] != baseline["fire"]:
        raise ValueError(
            "this record supports only the frozen fixed FIRE specification"
        )
    expected_angle = (
        baseline["angle_method"]
        if result["provider"] == "UFF"
        else "OpenFF harmonic"
    )
    if result["angle_method"] != expected_angle:
        raise ValueError("angle method must match the selected provider")
    if result["provider"] == "UFF" and result["enabled"]["improper"]:
        raise ValueError("supported UFF variants omit inversion execution")
    return result


def _numerical_count(value, name, positive=False):
    from numbers import Integral

    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or value < int(positive)
    ):
        raise ValueError(
            f"{name} must be a {'positive' if positive else 'nonnegative'} integer"
        )
    return int(value)


def _sample_schedule(protocol):
    """Return unsampled bulk and absolute sample boundaries through the cap."""
    p = _validate_numerical_protocol(protocol)
    first = min(p["interval"], p["minimum"])
    bulk = p["minimum"] - first
    completed, window = bulk, first
    windows = []
    while completed < p["cap"]:
        window = min(window, p["cap"] - completed)
        samples = min(p["samples"], window)
        base, remainder = divmod(window, samples)
        steps = []
        for i in range(samples):
            completed += base + (i < remainder)
            steps.append(completed)
        windows.append(tuple(steps))
        window = p["interval"]
    return bulk, tuple(windows)


def _aggregate_monitor_energies(protocol, energies, *, particles, counts):
    """Sum role-attributed force energies before declared normalization.

    ``energies`` is a sequence of unique ``role, force_energies`` pairs.
    ``counts`` contains every role, each with separate ``semantic``,
    ``components``, ``executed`` and ``force_objects`` integer counts.
    Components include assigned zero coefficients and exclude backend padding.
    The caller supplies these facts; this function never infers them from a
    force class or coefficient. Returned counts are an independent copy.
    """
    from copy import deepcopy

    p = _validate_numerical_protocol(protocol)
    particles = _numerical_count(particles, "particles", positive=True)
    if not isinstance(counts, Mapping) or set(counts) != set(_NUMERICAL_ROLES):
        raise ValueError("counts must declare every semantic role")
    for role, values in counts.items():
        if not isinstance(values, Mapping) or set(values) != {
            "semantic",
            "components",
            "executed",
            "force_objects",
        }:
            raise ValueError("retain all four separate count meanings")
        for name, value in values.items():
            _numerical_count(value, f"{role} {name}")
        if not p["enabled"][role] and (
            values["executed"] or values["force_objects"]
        ):
            raise ValueError("disabled roles cannot have executed forces")
    expected = {role for role in counts if counts[role]["force_objects"]}
    raw = {}
    for role, values in energies:
        if role not in expected or role in raw:
            raise ValueError("extra or duplicate energy role")
        if len(values) != counts[role]["force_objects"]:
            raise ValueError("supply one energy per backend force object")
        converted = []
        for value in values:
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
            ):
                raise ValueError("energies must be finite real numbers")
            converted.append(float(value))
        raw[role] = sum(converted)
        if not math.isfinite(raw[role]):
            raise ValueError("summed energy exceeds float range")
    if set(raw) != expected:
        raise ValueError("missing energy role")
    denominators = {
        role: particles
        if p["normalization"] == "particles" or role in ("pair", "coulomb")
        else max(1, counts[role]["semantic"])
        for role in raw
    }
    if p["torsion_channel"] == "combined":
        roles = [role for role in ("proper", "improper") if role in raw]
        if roles:
            raw["torsion"] = sum(raw.pop(role) for role in roles)
            denominators["torsion"] = (
                particles
                if p["normalization"] == "particles"
                else max(
                    1,
                    sum(
                        counts[role]["components"]
                        for role in ("proper", "improper")
                    ),
                )
            )
    normalized = {
        role: value / denominators[role] for role, value in raw.items()
    }
    if any(not math.isfinite(value) for value in normalized.values()):
        raise ValueError("normalized energy exceeds float range")
    return normalized, deepcopy(counts)


def _stationarity(
    previous, current, *, tolerance=0.02, stable=0, coulomb_floor=0.0
):
    """Return literal relative changes and the next passing-comparison count.

    Invalid finite/key histories raise instead of reproducing Sage30's missing
    guards. Finite subtraction overflow yields an infinite change and fails.
    The first mean creates history without a passing comparison.
    """
    import sys

    _numerical_count(stable, "stable")
    tolerance = _finite_coefficient(tolerance, "tolerance", allow_zero=True)
    floor = _finite_coefficient(coulomb_floor, "coulomb_floor", allow_zero=True)
    for means in (current,) if previous is None else (previous, current):
        if not isinstance(means, Mapping) or any(
            isinstance(v, bool)
            or not isinstance(v, Real)
            or not math.isfinite(v)
            for v in means.values()
        ):
            raise ValueError("means must be finite real mappings")
    if previous is None:
        return {}, 0
    if previous.keys() != current.keys():
        raise ValueError("mean channel keys differ")
    changes = {
        role: abs(float(current[role]) - float(previous[role]))
        / max(
            sys.float_info.min,
            floor if role == "coulomb" else 0.0,
            abs(float(previous[role])),
            abs(float(current[role])),
        )
        for role in current
    }
    return changes, stable + 1 if all(
        value <= tolerance for value in changes.values()
    ) else 0


def _dpd_stopping_decision(protocol, *, completed, stable):
    """Distinguish stationarity at the cap from an unconverged cap."""
    p = _validate_numerical_protocol(protocol)
    _numerical_count(completed, "completed")
    _numerical_count(stable, "stable")
    if completed > p["cap"]:
        raise ValueError("completed steps exceed cap")
    if completed >= p["minimum"] and stable >= p["comparisons"]:
        return "converged"
    if completed == p["cap"]:
        return "cap_failure" if p["require_convergence"] else "cap_allowed"
    return "continue"


def _fire_stopping_decision(
    completed,
    converged,
    *,
    initial_steps=100,
    convergence_driven=False,
    interval=100,
    cap=1000,
    require_convergence=True,
):
    """Specify the FIRE gate without running an optimizer.

    Fixed-step completion ignores the recorded optimizer convergence flag.
    Convergence-driven initial steps above cap are rejected, unlike the frozen
    controller's unchecked initial request.
    """
    for name, value in (
        ("completed", completed),
        ("initial_steps", initial_steps),
        ("interval", interval),
        ("cap", cap),
    ):
        _numerical_count(value, name, positive=name != "completed")
    if any(
        type(value) is not bool
        for value in (converged, convergence_driven, require_convergence)
    ):
        raise ValueError("FIRE mode and convergence flags must be boolean")
    if convergence_driven and initial_steps > cap:
        raise ValueError("initial FIRE steps exceed cap")
    limit = cap if convergence_driven else initial_steps
    if completed > limit:
        raise ValueError("completed FIRE steps exceed declared limit")
    if completed < initial_steps:
        return "continue"
    if not convergence_driven:
        return "fixed_steps_complete"
    if converged:
        return "converged"
    if completed == cap:
        return "cap_failure" if require_convergence else "cap_allowed"
    return "continue"
