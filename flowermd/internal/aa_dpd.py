"""Internal coefficient weighting for all-atom DPD force providers."""

import math
from collections.abc import Mapping
from numbers import Real


def epsilon_scaled_dpd_parameters(
    particle_epsilons, repulsion, gamma, epsilon_reference=None
):
    """Weight DPD coefficients by geometric-mean particle epsilons.

    Parameters
    ----------
    particle_epsilons : mapping of str to float
        Positive, finite epsilon magnitudes in one common energy unit.
        Mapping order determines the order of the returned unordered pairs.
    repulsion, gamma : float
        Finite, nonnegative reference DPD coefficients.
    epsilon_reference : float, optional
        Positive, finite reference in the same energy unit. Defaults to the
        maximum supplied epsilon. Historical protocols using a different
        reference (including the maximum of only overridden epsilons) must
        pass it explicitly.

    Returns
    -------
    dict
        One entry per unordered type pair, including self pairs, containing
        ``A`` and ``gamma`` multiplied by ``sqrt(epsilon_i * epsilon_j) /
        epsilon_reference``.

    Notes
    -----
    This weights coefficients only; it does not construct forces or add
    Lennard-Jones or Coulomb interactions. Both UFF and Sage bonded providers
    can supply UFF pair epsilons. Inputs are numeric magnitudes; no unit
    conversion is performed. Nonzero outputs outside the representable float
    range raise ValueError instead of silently becoming infinity or zero.
    """
    if not isinstance(particle_epsilons, Mapping) or not particle_epsilons:
        raise ValueError("particle_epsilons must be a nonempty mapping")
    epsilons = {}
    for name, epsilon in particle_epsilons.items():
        if not isinstance(name, str) or not name:
            raise ValueError("particle type names must be nonempty strings")
        epsilons[name] = _finite_coefficient(epsilon, f"epsilon for {name}")
    repulsion = _finite_coefficient(repulsion, "repulsion", allow_zero=True)
    gamma = _finite_coefficient(gamma, "gamma", allow_zero=True)
    reference = (
        max(epsilons.values())
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
    if coefficient == 0:
        return 0.0
    # Keep powers of two separate so intermediate products cannot overflow
    # or underflow when the final coefficient is representable.
    first_mantissa, first_exponent = math.frexp(first)
    second_mantissa, second_exponent = math.frexp(second)
    ref_mantissa, ref_exponent = math.frexp(reference)
    coeff_mantissa, coeff_exponent = math.frexp(coefficient)
    exponent, remainder = divmod(first_exponent + second_exponent, 2)
    mantissa = (
        coeff_mantissa
        * math.sqrt(first_mantissa * second_mantissa * 2**remainder)
        / ref_mantissa
    )
    try:
        result = math.ldexp(mantissa, coeff_exponent + exponent - ref_exponent)
    except OverflowError:
        raise ValueError(
            "weighted DPD coefficient exceeds float range"
        ) from None
    if result == 0:
        raise ValueError("weighted DPD coefficient underflows float range")
    return result
