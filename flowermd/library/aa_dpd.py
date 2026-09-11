"""Native HOOMD forces from assigned GMSO all-atom topology objects."""

import math

import gmso
import hoomd
import sympy
import unyt as u
from gmso.lib.potential_templates import PotentialTemplateLibrary

from flowermd.base.forcefield import BaseHOOMDForcefield
from flowermd.internal.aa_dpd import _finite_coefficient, dpd_pair_parameters


class AllAtomDPD(BaseHOOMDForcefield):
    """Build fresh AA-DPD forces directly from assigned GMSO potentials.

    Parameters
    ----------
    topology : gmso.Topology
        Assigned atom and bonded potentials with explicit units. Supported
        bonded forms are harmonic bonds/angles and scalar HOOMD periodic
        proper torsions. Expressions and variables identify forms, not names.
        Coordinates, topology, potentials and their units are not modified.
    repulsion, gamma : float
        Nonnegative nominal DPD coefficients under fixed references. Repulsion
        has energy/length units; gamma has mass/time units, where time is
        sqrt(mass * length**2 / energy).
    kT, r_cut, bonded_scale : float
        Positive thermal energy in kcal/mol, cutoff in angstroms and
        dimensionless bonded multiplier. Scaling is applied exactly once.
    epsilon_weighting : bool, default True
        Weight pairs using assigned AtomType epsilon quantities. Uniform mode
        reads neither epsilon nor sigma.
    conservative : bool, default False
        Build DPDConservative instead of the DPD thermostat, with identical
        conservative coefficients, cutoff and exclusions.
    epsilon_reference : float, optional
        Positive reference in kcal/mol, defaulting to the largest epsilon.
        Omit in uniform mode. Zero epsilon disables repulsion and thermostat
        coupling for that type; all-zero epsilons require an explicit reference.
    include_bonds, include_angles, include_torsions, include_impropers : bool
        Defaults are True. Disabled recognized potentials still validate.
        No improper backend is implemented: enabled UFF inversions raise an
        explicit error. Retained untyped impropers may be explicitly disabled.

    Notes
    -----
    Fresh references are 1 angstrom, 1 kcal/mol and 1 amu. The neighbor-list
    buffer is 0.4 angstrom, with bond/angle/dihedral exclusions independent of
    ablations. No LJ, Coulomb or implicit improper substitution is performed.
    Scalar periodic coefficients may be signed, as fitted force fields require.

    ``type_labels`` maps actual potential objects to backend labels separately
    for sites, bonds, angles, dihedrals and impropers, in first-occurrence order.
    A None improper key labels explicitly disabled untyped topology only.
    ``disabled_term_counts`` and ``untyped_improper_count`` report omissions.
    Frame construction must use these same maps, ordered GMSO connections and
    site indices as particle tags. Rebuild frames and forces after assignments
    change. This class does not construct a frame.
    """

    def __init__(
        self,
        topology,
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
        include_impropers=True,
    ):
        if not isinstance(topology, gmso.Topology) or not topology.n_sites:
            raise ValueError("topology must be a nonempty GMSO Topology")
        flags = {
            "include_bonds": include_bonds,
            "include_angles": include_angles,
            "include_torsions": include_torsions,
            "include_impropers": include_impropers,
            "epsilon_weighting": epsilon_weighting,
            "conservative": conservative,
        }
        for name, value in flags.items():
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a bool")
            setattr(self, name, value)
        kT = _finite_coefficient(kT, "kT")
        r_cut = _finite_coefficient(r_cut, "r_cut")
        scale = _finite_coefficient(bonded_scale, "bonded_scale")
        if any(site.atom_type is None for site in topology.sites):
            raise ValueError("all sites require assigned AtomType potentials")
        self.type_labels = {
            "sites": _labels(
                (site.atom_type for site in topology.sites), "atom"
            )
        }
        self.disabled_term_counts = {}
        self.untyped_improper_count = sum(
            item.improper_type is None for item in topology.impropers
        )
        epsilons = None
        if epsilon_weighting:
            epsilons = {
                label: _quantity(potential, "epsilon", "kcal/mol")
                for potential, label in self.type_labels["sites"].items()
            }
        pair_parameters = dpd_pair_parameters(
            tuple(self.type_labels["sites"].values()),
            repulsion,
            gamma,
            epsilon_weighting=epsilon_weighting,
            particle_epsilons=epsilons,
            epsilon_reference=epsilon_reference,
        )
        templates = PotentialTemplateLibrary()
        forces = []
        site_indices = {
            site: index for index, site in enumerate(topology.sites)
        }
        for category, enabled, force_class, template_name in (
            (
                "bonds",
                include_bonds,
                hoomd.md.bond.Harmonic,
                "HarmonicBondPotential",
            ),
            (
                "angles",
                include_angles,
                hoomd.md.angle.Harmonic,
                "HarmonicAnglePotential",
            ),
            (
                "dihedrals",
                include_torsions,
                hoomd.md.dihedral.Periodic,
                "HOOMDPeriodicDihedralPotential",
            ),
            ("impropers", include_impropers, None, None),
        ):
            connections = tuple(getattr(topology, category))
            labels = _labels(
                (item.connection_type for item in connections), category[:-1]
            )
            self.type_labels[category] = labels
            if not enabled:
                self.disabled_term_counts[category] = len(connections)
            representatives = {}
            for connection in connections:
                representatives.setdefault(
                    connection.connection_type,
                    tuple(
                        site_indices[site]
                        for site in connection.connection_members
                    ),
                )
            converted = {}
            for potential, label in labels.items():
                location = f"{category} at sites {representatives[potential]}"
                if potential is None:
                    if not enabled and category == "impropers":
                        continue
                    raise ValueError(f"{location} require assigned potentials")
                if category == "impropers":
                    if enabled:
                        if potential.tags.get("form") == "uff_inversion":
                            raise NotImplementedError(
                                f"{location}: native UFF inversion requires its Wilson out-of-plane force backend; explicitly disable impropers for an ablation"
                            )
                        raise NotImplementedError(
                            f"{location}: no improper force backend is implemented for this assigned form"
                        )
                    continue
                template = templates[template_name]
                if (
                    potential.independent_variables
                    != template.independent_variables
                    or sympy.simplify(
                        potential.expression - template.expression
                    )
                    != 0
                ):
                    if enabled:
                        raise ValueError(
                            f"unsupported {location} expression or independent variables for {potential.name}"
                        )
                    continue
                try:
                    converted[label] = _bonded_parameters(
                        potential, category, scale
                    )
                except ValueError as error:
                    raise ValueError(f"{location}: {error}") from error
            if enabled and converted:
                force = force_class()
                for label, values in converted.items():
                    force.params[label] = values
                forces.append(force)
        neighbor_list = hoomd.md.nlist.Cell(
            buffer=0.4, exclusions=("bond", "angle", "dihedral")
        )
        pair = (
            hoomd.md.pair.DPDConservative(neighbor_list, default_r_cut=r_cut)
            if conservative
            else hoomd.md.pair.DPD(neighbor_list, default_r_cut=r_cut, kT=kT)
        )
        for names, values in pair_parameters.items():
            pair.params[names] = {"A": values["A"]} if conservative else values
        forces.append(pair)
        self.reference_values = {
            "length": 1.0 * u.angstrom,
            "energy": 1.0 * u.kcal / u.mol,
            "mass": 1.0 * u.amu,
        }
        super().__init__(hoomd_forces=forces)


def _labels(potentials, prefix):
    return {
        potential: f"aa_{prefix}_{index}"
        for index, potential in enumerate(dict.fromkeys(potentials))
    }


def _quantity(potential, name, unit):
    value = potential.parameters.get(name)
    if not isinstance(value, u.unyt_array) or value.ndim != 0:
        raise ValueError(f"{potential.name} {name} must be a scalar quantity")
    try:
        result = float(value.to_value(unit))
    except (ValueError, u.exceptions.UnitConversionError) as error:
        raise ValueError(
            f"{potential.name} {name} has incompatible units; expected {unit}"
        ) from error
    if not math.isfinite(result):
        raise ValueError(f"{potential.name} {name} must be finite")
    return result


def _bonded_parameters(potential, category, scale):
    units = {
        "bonds": "kcal/(mol*angstrom**2)",
        "angles": "kcal/(mol*rad**2)",
        "dihedrals": "kcal/mol",
    }
    k = _quantity(potential, "k", units[category])
    if category != "dihedrals":
        _finite_coefficient(k, f"{potential.name} k")
    scaled = k * scale
    if not math.isfinite(scaled) or (k != 0 and scaled == 0):
        raise ValueError(
            f"scaled stiffness for {potential.name} is outside float range"
        )
    if category == "bonds":
        return {
            "k": scaled,
            "r0": _finite_coefficient(
                _quantity(potential, "r_eq", "angstrom"), "r_eq"
            ),
        }
    if category == "angles":
        target = _finite_coefficient(
            _quantity(potential, "theta_eq", "rad"), "theta_eq"
        )
        if target > math.pi:
            raise ValueError("theta_eq must be in (0, pi]")
        return {"k": scaled, "t0": target}
    n = _quantity(potential, "n", "dimensionless")
    d = _quantity(potential, "d", "dimensionless")
    if not n.is_integer() or n <= 0:
        raise ValueError("periodic n must be a positive integer")
    if d not in (-1, 1):
        raise ValueError("periodic d must be -1 or 1")
    return {
        "k": scaled,
        "n": int(n),
        "d": int(d),
        "phi0": _quantity(potential, "phi0", "rad"),
    }
