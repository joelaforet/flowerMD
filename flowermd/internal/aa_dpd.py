"""Internal coefficient weighting for all-atom DPD."""

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
