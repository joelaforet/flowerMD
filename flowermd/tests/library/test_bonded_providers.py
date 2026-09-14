"""Exercise structural assignment providers through AllAtomSystem."""

from copy import deepcopy
from itertools import product

import gmso
import numpy as np
import pytest
import unyt as u

from flowermd.base import BondedParameterProvider
from flowermd.internal.openff_gmso import assign_openff_parameters
from flowermd.library import OpenFFProvider, SageProvider, UFFProvider
from flowermd.tests.library.test_aa_system import apply, create


class MassOnlyProvider:
    """Supply native harmonic terms without any nonbonded parameters."""

    def __init__(self):
        self.calls = []
        self.output = None

    def assign(self, topology, molecule, *, atom_map, assign_nonbonded=True):
        self.calls.append(assign_nonbonded)
        for site in topology.sites:
            site.atom_type = gmso.AtomType(
                name=site.element.symbol,
                mass=site.element.mass,
                expression="0",
                independent_variables=set(),
                parameters={},
            )
        indices = {s: i for i, s in enumerate(topology.sites)}
        orders = {
            frozenset(
                (atom_map[b.GetBeginAtomIdx()], atom_map[b.GetEndAtomIdx()])
            ): b.GetBondTypeAsDouble()
            for b in molecule.GetBonds()
        }
        bond_type = gmso.BondType(
            expression="0.5*k*(r-r_eq)**2",
            independent_variables={"r"},
            parameters={
                "k": 200 * u.kcal / u.mol / u.angstrom**2,
                "r_eq": 1.2 * u.angstrom,
            },
        )
        angle_type = gmso.AngleType(
            expression="0.5*k*(theta-theta_eq)**2",
            independent_variables={"theta"},
            parameters={
                "k": 40 * u.kcal / u.mol / u.rad**2,
                "theta_eq": 1.9 * u.rad,
            },
        )
        dihedral_type = gmso.DihedralType(
            expression="k*(1+cos(n*phi-phi_eq))",
            independent_variables={"phi"},
            parameters={
                "k": 0.5 * u.kcal / u.mol,
                "n": 3 * u.dimensionless,
                "phi_eq": 0 * u.rad,
            },
        )
        for bond in topology.bonds:
            bond.bond_order = orders[
                frozenset(indices[s] for s in bond.connection_members)
            ]
            bond.connection_type = bond_type
        for angle in topology.angles:
            angle.connection_type = angle_type
        for torsion in topology.dihedrals:
            torsion.connection_type = dihedral_type
        for improper in tuple(topology.impropers):
            topology.remove_connection(improper)
        self.output = (
            topology,
            {"source": "Example native harmonics", "revision": [1]},
        )
        return self.output


def test_protocol_is_structural_and_mass_only_is_unweighted():
    provider = MassOnlyProvider()
    assert BondedParameterProvider not in type(provider).__mro__
    system = create("CC")
    apply(system, bonded=provider, epsilon_weighting=False)
    assert provider.calls == [False] and provider.calls[0] is False
    assert all(not s.atom_type.parameters for s in system.gmso_system.sites)
    assert system.assignment_report["epsilon_source"] is None
    assert (
        system.assignment_report["assignment"]["source"]
        == "Example native harmonics"
    )
    assert system.assignment_report["bonded"].endswith(".MassOnlyProvider")
    assert all(
        v == {"A": 40, "gamma": 20}
        for v in system.assignment_report["pair_coefficients"].values()
    )
    previous = system.dpd_forcefield
    with pytest.raises(ValueError, match="epsilon"):
        apply(system, bonded=provider, epsilon_weighting=True)
    assert provider.calls == [False, True]
    assert system.dpd_forcefield is previous


@pytest.mark.parametrize("flags", product([False, True], repeat=4))
def test_provider_receives_only_weighting_not_force_ablations(flags):
    provider = MassOnlyProvider()
    system = create("CC")
    names = (
        "include_bonds",
        "include_angles",
        "include_torsions",
        "include_impropers",
    )
    apply(
        system,
        bonded=provider,
        epsilon_weighting=False,
        **dict(zip(names, flags)),
    )
    assert provider.calls == [False]
    assert all(b.connection_type is not None for b in system.gmso_system.bonds)
    assert all(a.connection_type is not None for a in system.gmso_system.angles)
    assert all(
        d.connection_type is not None for d in system.gmso_system.dihedrals
    )
    for category, enabled in zip(("bonds", "angles", "dihedrals"), flags):
        assert (
            bool(system.dpd_forcefield.forces_by_category[category]) is enabled
        )


@pytest.mark.parametrize(
    "provider,string",
    [
        (UFFProvider(), "uff"),
        (OpenFFProvider(), "openff"),
        (SageProvider(), "sage"),
    ],
)
@pytest.mark.parametrize("weighting", [False, True])
def test_builtin_instances_equal_legacy_strings(provider, string, weighting):
    system = create()
    apply(system, bonded=string, epsilon_weighting=weighting)
    before = system.assignment_report
    masses = system.hoomd_snapshot.particles.mass.copy()
    parameters = [
        deepcopy(c.connection_type.parameters)
        for c in system.gmso_system.connections
    ]
    apply(system, bonded=provider, epsilon_weighting=weighting)
    assert system.assignment_report["assignment"] == before["assignment"]
    assert (
        system.assignment_report["pair_coefficients"]
        == before["pair_coefficients"]
    )
    np.testing.assert_array_equal(system.hoomd_snapshot.particles.mass, masses)
    for expected, connection in zip(parameters, system.gmso_system.connections):
        for name, value in expected.items():
            np.testing.assert_array_equal(
                connection.connection_type.parameters[name], value
            )


def test_openff_provider_copies_configuration_and_matches_adapter():
    from openff.toolkit import ForceField

    force_field = ForceField("openff-1.3.1.offxml")
    provider = OpenFFProvider(force_field)
    expected_xml = force_field.to_string()
    force_field.deregister_parameter_handler("Bonds")
    returned = provider.force_field
    returned.deregister_parameter_handler("Angles")
    assert provider.force_field.to_string() == expected_xml
    system = create("CC")
    typed, report = assign_openff_parameters(
        system._construction_topology,
        system._molecule,
        atom_map=system._atom_map,
        force_field=provider.force_field,
        assign_nonbonded=False,
    )
    apply(system, bonded=provider, epsilon_weighting=False)
    assert system.assignment_report["assignment"] == report
    for expected, actual in zip(typed.bonds, system.gmso_system.bonds):
        assert expected.bond_type.parameters == actual.bond_type.parameters
    assert provider.force_field.to_string() == expected_xml


def test_provider_output_and_report_are_owned():
    provider = MassOnlyProvider()
    system = create("CC")
    apply(system, bonded=provider, epsilon_weighting=False)
    positions = system.gmso_system.positions.copy()
    report = system.assignment_report
    topology, retained_report = provider.output
    topology.sites[0].position += 5 * u.nm
    topology.sites[0].atom_type.mass = 99 * u.amu
    topology.remove_connection(topology.bonds[0])
    retained_report["revision"].append(2)
    np.testing.assert_array_equal(system.gmso_system.positions, positions)
    assert system.assignment_report == report
    assert system.gmso_system.sites[0].atom_type.mass != 99 * u.amu


def test_mutating_failed_provider_cannot_change_retained_input_or_state():
    class Fails:
        def assign(self, topology, molecule, *, atom_map, assign_nonbonded):
            topology.sites[0].position += 7 * u.nm
            topology.box.lengths = [9, 9, 9] * u.nm
            molecule.GetAtomWithIdx(0).SetAtomicNum(15)
            atom_map.clear()
            raise RuntimeError("assignment failed")

    system = create("CC")
    apply(system)
    topology, frame, bundle, report = (
        system.gmso_system,
        system.hoomd_snapshot,
        system.dpd_forcefield,
        system.assignment_report,
    )
    with pytest.raises(RuntimeError, match="assignment failed"):
        apply(system, bonded=Fails())
    assert system.gmso_system is topology and system.hoomd_snapshot is frame
    assert (
        system.dpd_forcefield is bundle and system.assignment_report == report
    )
    apply(system)
    np.testing.assert_array_equal(
        system.gmso_system.positions, topology.positions
    )
    assert system.assignment_report == report


@pytest.mark.parametrize(
    "bad",
    [
        "arity",
        "topology",
        "report",
        "source",
        "blank",
        "positions",
        "box",
        "graph",
        "order",
        "element",
        "name",
        "label",
        "charge",
        "untyped",
        "mass",
        "unsupported",
    ],
)
def test_invalid_output_preserves_previous_state(bad):
    class Broken(MassOnlyProvider):
        def assign(self, *args, **kwargs):
            topology, report = super().assign(*args, **kwargs)
            if bad == "arity":
                return (topology,)
            if bad == "topology":
                return object(), report
            if bad == "report":
                return topology, []
            if bad == "source":
                report.pop("source")
            elif bad == "blank":
                report["source"] = " "
            elif bad == "positions":
                topology.sites[0].position += 1 * u.nm
            elif bad == "box":
                topology.box.lengths = [8, 8, 8] * u.nm
            elif bad == "graph":
                topology.remove_connection(topology.bonds[0])
            elif bad == "order":
                sites = list(topology.sites)
                topology._sites = type(topology.sites)(
                    [sites[1], sites[0], *sites[2:]]
                )
            elif bad == "element":
                topology.sites[0].element = topology.sites[-1].element
            elif bad in ("name", "label"):
                setattr(topology.sites[0], bad, "changed")
            elif bad == "charge":
                topology.sites[0].charge = 1 * u.elementary_charge
            elif bad == "untyped":
                topology.bonds[0].bond_type = None
            elif bad == "mass":
                topology.sites[0].atom_type.mass = 0 * u.amu
            elif bad == "unsupported":
                topology.bonds[0].bond_type.expression = "k*(r-r_eq)**4"
            return topology, report

    system = create("CC")
    apply(system)
    original = system.gmso_system, system.hoomd_snapshot, system.dpd_forcefield
    report = system.assignment_report
    with pytest.raises((ValueError, NotImplementedError)):
        apply(system, bonded=Broken(), epsilon_weighting=False)
    assert all(
        a is b
        for a, b in zip(
            original,
            (system.gmso_system, system.hoomd_snapshot, system.dpd_forcefield),
        )
    )
    assert system.assignment_report == report


def test_explicit_charge_and_site_metadata_survive_assignment():
    system = create("CC")
    # Exercise metadata supported by the native GMSO assignment boundary.
    site = system._construction_topology.sites[0]
    site.charge = -0.3 * u.elementary_charge
    site.label, site.group = "original label", "original group"
    site.molecule = ("chain", 3)
    site.residue = ("repeat", 2)
    apply(system, bonded=MassOnlyProvider(), epsilon_weighting=False)
    actual = system.gmso_system.sites[0]
    assert actual.charge == site.charge
    assert actual.label == site.label and actual.group == site.group
    assert actual.molecule.name == "chain" and actual.residue.number == 2


@pytest.mark.parametrize(
    "provider",
    [UFFProvider(), OpenFFProvider(), SageProvider(), MassOnlyProvider()],
)
def test_explicit_provider_rejects_compatibility_resource(provider):
    with pytest.raises(ValueError, match="configure the provider"):
        apply(create("CC"), bonded=provider, force_field="openff-2.3.0.offxml")


@pytest.mark.parametrize("flag", [1, np.bool_(True), "false"])
def test_invalid_weighting_never_calls_provider(flag):
    provider = MassOnlyProvider()
    with pytest.raises(ValueError, match="epsilon_weighting must be a bool"):
        apply(create("CC"), bonded=provider, epsilon_weighting=flag)
    assert provider.calls == []


def test_invalid_epsilon_is_ignored_only_when_unweighted():
    class BadEpsilon(MassOnlyProvider):
        def assign(self, *args, **kwargs):
            topology, report = super().assign(*args, **kwargs)
            for site in topology.sites:
                site.atom_type.parameters["epsilon"] = 5 * u.angstrom
            return topology, report

    system = create("CC")
    provider = BadEpsilon()
    apply(system, bonded=provider, epsilon_weighting=False)
    previous = system.dpd_forcefield
    with pytest.raises(ValueError, match="epsilon"):
        apply(system, bonded=provider, epsilon_weighting=True)
    assert system.dpd_forcefield is previous


@pytest.mark.parametrize(
    "change", ["order", "connection_name", "restraint", "remove"]
)
def test_provider_preserves_bond_orders_and_connection_metadata(change):
    class Broken(MassOnlyProvider):
        def assign(self, *args, **kwargs):
            topology, report = super().assign(*args, **kwargs)
            if change == "order":
                topology.bonds[0].bond_order = 2
            elif change == "connection_name":
                topology.angles[0].name = "changed"
            elif change == "restraint":
                topology.angles[0].restraint = None
            else:
                topology.remove_connection(topology.angles[0])
            return topology, report

    system = create("CC")
    system._construction_topology.angles[0].restraint = {
        "theta0": 110 * u.degree,
        "ktheta": 2 * u.kcal / u.mol / u.rad**2,
    }
    with pytest.raises(
        ValueError, match="bond orders|connection metadata|restrained"
    ):
        apply(system, bonded=Broken(), epsilon_weighting=False)
    with pytest.raises(RuntimeError, match="apply_dpd"):
        _ = system.assignment_report


def test_custom_provider_reports_its_actual_weighted_source():
    class Weighted(MassOnlyProvider):
        def assign(self, *args, **kwargs):
            topology, report = super().assign(*args, **kwargs)
            for site in topology.sites:
                site.atom_type.parameters["epsilon"] = 2 * u.kcal / u.mol
            report["source"] = "Example native epsilon"
            return topology, report

    system = create("CC")
    provider = Weighted()
    apply(system, bonded=provider)
    assert provider.calls == [True] and provider.calls[0] is True
    assert (
        system.assignment_report["epsilon_source"] == "Example native epsilon"
    )
    assert system.assignment_report["epsilon_reference_kcal_mol"] == 2
