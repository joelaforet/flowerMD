"""Native HOOMD forces for numerically parameterized all-atom DPD systems."""

import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real

import hoomd
import unyt as u

from flowermd.base.forcefield import BaseHOOMDForcefield
from flowermd.internal.aa_dpd import _finite_coefficient, dpd_pair_parameters


class AllAtomDPD(BaseHOOMDForcefield):
    """Build fresh AA-DPD forces from numeric atom and bonded parameter tables.

    Parameters
    ----------
    parameters : mapping
        Atom/type and bond/angle/proper-dihedral tables following the internal
        AA parameter schema. Atom types are in particle order. Bond stiffness
        is in kcal/mol/angstrom**2, angle stiffness in kcal/mol/radian**2,
        torsion barriers in kcal/mol, lengths in angstroms and angles in
        radians. This class does not parameterize chemistry or build a frame.
    repulsion, gamma : float
        Nonnegative nominal DPD coefficients in the canonical reduced units.
        Repulsion has energy/length units; gamma has mass/time units, with
        time = sqrt(mass * length**2 / energy) from the fixed references.
    kT : float
        Positive thermal energy in kcal/mol.
    r_cut : float
        Positive DPD cutoff in angstroms.
    bonded_scale : float
        Positive multiplier applied once to bonded stiffnesses and barriers.
    epsilon_weighting : bool, default True
        Weight A and gamma with epsilon values from the selected force field,
        supplied as ``epsilon_kcal_mol`` in the particle parameter table.
        If False, epsilon fields are not accessed.
    conservative : bool, default False
        Use DPDConservative instead of the DPD thermostat. The conservative
        coefficients, cutoff and exclusions are identical in either mode.
    epsilon_reference : float, optional
        Positive reference in kcal/mol; defaults to the maximum epsilon.
        Omit when weighting is disabled. All-zero epsilons require an explicit
        positive reference. A zero epsilon disables pair repulsion and DPD
        thermostat coupling for that type.
    include_bonds, include_angles, include_torsions : bool, default True
        Include each bonded force family. Disabled terms are still validated,
        and the supplied topology and pair exclusions stay unchanged. These
        switches permit energy-term ablations without changing the pair graph.

    Notes
    -----
    ``reference_values`` contains fresh unyt quantities: 1 angstrom,
    1 kcal/mol and 1 amu. Coordinates and frames must use these same references.
    The neighbor list buffer is 0.4 angstrom with bond/angle/dihedral exclusions.
    No LJ, Coulomb or improper forces are added. Empty bonded terms are omitted.
    """

    def __init__(
        self,
        parameters,
        *,
        repulsion,
        gamma,
        kT,
        r_cut,
        bonded_scale,
        epsilon_weighting=True,
        conservative=False,
        epsilon_reference=None,
        include_bonds=True,
        include_angles=True,
        include_torsions=True,
    ):
        if not isinstance(parameters, Mapping):
            raise ValueError("parameters must be a mapping")
        if not isinstance(epsilon_weighting, bool) or not isinstance(
            conservative, bool
        ):
            raise ValueError("epsilon_weighting and conservative must be bools")
        for flag, value in (
            ("include_bonds", include_bonds),
            ("include_angles", include_angles),
            ("include_torsions", include_torsions),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"{flag} must be a bool")
            setattr(self, flag, value)
        kT = _finite_coefficient(kT, "kT")
        r_cut = _finite_coefficient(r_cut, "r_cut")
        scale = _finite_coefficient(bonded_scale, "bonded_scale")
        particle_names = _type_names(
            _field(parameters, "particle_types"), "particle_types"
        )
        if not particle_names:
            raise ValueError("particle_types must be nonempty")
        particle_table = _type_table(
            parameters, "particle_type_params", particle_names
        )
        epsilons = None
        if epsilon_weighting:
            epsilons = {
                name: _field(particle_table[name], "epsilon_kcal_mol")
                for name in particle_names
            }
        pair_parameters = dpd_pair_parameters(
            particle_names,
            repulsion,
            gamma,
            epsilon_weighting=epsilon_weighting,
            particle_epsilons=epsilons,
            epsilon_reference=epsilon_reference,
        )
        forces = []
        for groups, types, table_key, force_class, enabled in (
            (
                "bonds",
                "bond_types",
                "bond_params",
                hoomd.md.bond.Harmonic,
                include_bonds,
            ),
            (
                "angles",
                "angle_types",
                "angle_params",
                hoomd.md.angle.Harmonic,
                include_angles,
            ),
            (
                "dihedrals",
                "dihedral_types",
                "dihedral_params",
                hoomd.md.dihedral.Periodic,
                include_torsions,
            ),
        ):
            group_values = _field(parameters, groups)
            type_values = _field(parameters, types)
            names = _type_names(type_values, types)
            if not isinstance(group_values, Sequence) or isinstance(
                group_values, (str, bytes)
            ):
                raise ValueError(f"{groups} must be an ordered sequence")
            if len(group_values) != len(type_values):
                raise ValueError(
                    f"{groups} and {types} must have matching counts"
                )
            table = _type_table(parameters, table_key, names)
            if not names:
                continue
            force = force_class()
            for name in names:
                values = table[name]
                if groups == "bonds":
                    force.params[name] = {
                        "k": _scaled(values, "k_kcal_mol_a2", scale, name),
                        "r0": _finite_coefficient(
                            _field(values, "r0_a"), f"{name} r0_a"
                        ),
                    }
                elif groups == "angles":
                    theta = _finite_coefficient(
                        _field(values, "theta0_rad"), f"{name} theta0_rad"
                    )
                    if theta > math.pi:
                        raise ValueError(
                            f"{name} theta0_rad must be in (0, pi]"
                        )
                    force.params[name] = {
                        "k": _scaled(values, "k_kcal_mol_rad2", scale, name),
                        "t0": theta,
                    }
                else:
                    order, sign = _field(values, "n"), _field(values, "d")
                    if (
                        isinstance(order, bool)
                        or not isinstance(order, Integral)
                        or order <= 0
                    ):
                        raise ValueError(f"{name} n must be a positive integer")
                    if (
                        isinstance(sign, bool)
                        or not isinstance(sign, Real)
                        or sign not in (-1, 1)
                    ):
                        raise ValueError(f"{name} d must be -1 or 1")
                    phase = _field(values, "phi0_rad")
                    if (
                        isinstance(phase, bool)
                        or not isinstance(phase, Real)
                        or not math.isfinite(phase)
                    ):
                        raise ValueError(f"{name} phi0_rad must be finite")
                    force.params[name] = {
                        "k": _scaled(
                            values, "k_kcal_mol", scale, name, allow_zero=True
                        ),
                        "n": int(order),
                        "d": int(sign),
                        "phi0": float(phase),
                    }
            if enabled:
                forces.append(force)
        neighbor_list = hoomd.md.nlist.Cell(
            buffer=0.4, exclusions=("bond", "angle", "dihedral")
        )
        if conservative:
            pair = hoomd.md.pair.DPDConservative(
                neighbor_list, default_r_cut=r_cut
            )
        else:
            pair = hoomd.md.pair.DPD(neighbor_list, default_r_cut=r_cut, kT=kT)
        for names, values in pair_parameters.items():
            pair.params[names] = {"A": values["A"]} if conservative else values
        forces.append(pair)
        self.reference_values = {
            "length": 1.0 * u.angstrom,
            "energy": 1.0 * u.kcal / u.mol,
            "mass": 1.0 * u.amu,
        }
        super().__init__(hoomd_forces=forces)


def _field(values, key):
    if key not in values:
        raise ValueError(f"missing parameter field {key}")
    return values[key]


def _type_names(values, field):
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{field} must be an ordered sequence")
    if any(not isinstance(name, str) or not name for name in values):
        raise ValueError(f"{field} must contain nonempty type names")
    return tuple(dict.fromkeys(values))


def _type_table(parameters, key, names):
    table = _field(parameters, key)
    if not isinstance(table, Mapping):
        raise ValueError(f"{key} must be a mapping")
    for name in names:
        if name not in table or not isinstance(table[name], Mapping):
            raise ValueError(f"{key} is missing parameters for type {name}")
    return table


def _scaled(values, key, scale, name, allow_zero=False):
    original = _finite_coefficient(
        _field(values, key), f"{name} {key}", allow_zero=allow_zero
    )
    result = original * scale
    if not math.isfinite(result) or (original > 0 and result == 0):
        raise ValueError(f"scaled stiffness for {name} is outside float range")
    return result
