"""Check native Fourier execution against independent OpenMM forces."""

from copy import deepcopy

import gmso
import hoomd
import numpy as np
import pytest
import unyt as u
from gmso.lib.potential_templates import PotentialTemplateLibrary

from flowermd.tests.library.test_aa_dpd import bundle, evaluate, topology


def periodic(k=(1.2, -0.7, 0), n=(1, 3, 2), phase=(0.4, -0.9, 1.1)):
    return gmso.DihedralType.from_template(
        PotentialTemplateLibrary()["PeriodicTorsionPotential"],
        {
            "k": u.unyt_array(k, "kcal/mol"),
            "n": u.unyt_array(n, "dimensionless"),
            "phi_eq": u.unyt_array(phase, "rad"),
        },
        name="same",
    )


def mixed_topology():
    top = topology()
    native = periodic()
    shorter = periodic(k=(-0.3,), n=(2,), phase=(0.7,))
    scalar = periodic(k=-0.6, n=4, phase=1.4)
    legacy = top.dihedrals[0].dihedral_type
    legacy.name = "same"
    legacy.parameters["d"] *= -1
    legacy.parameters["phi0"] = -0.5 * u.rad
    potentials = [legacy, native, shorter, scalar]
    for potential in potentials[1:]:
        members = []
        for _ in range(4):
            site = gmso.Atom(atom_type=top.sites[0].atom_type)
            top.add_site(site)
            members.append(site)
        top.add_connection(
            gmso.Dihedral(connection_members=members, dihedral_type=potential)
        )
    return top, potentials


def torsion_forces(ff):
    return [
        f for f in ff.hoomd_forces if isinstance(f, hoomd.md.dihedral.Periodic)
    ]


def test_mixed_components_labels_padding_and_preservation():
    top, potentials = mixed_topology()
    before = [deepcopy(p.parameters) for p in potentials]
    connections = tuple(top.dihedrals)
    members = [tuple(c.connection_members) for c in connections]
    ff = bundle(top, bonded_scale=0.7)
    forces = torsion_forces(ff)
    assert len(forces) == 3
    assert len(ff.type_labels["dihedrals"]) == 4
    assert len({p.name for p in potentials}) == 1
    for index, force in enumerate(forces):
        assert set(force.params) == set(ff.type_labels["dihedrals"].values())
        for p, label in ff.type_labels["dihedrals"].items():
            if p is potentials[1]:
                expected = {
                    "k": 1.4 * before[1]["k"][index].value,
                    "n": int(before[1]["n"][index]),
                    "d": 1,
                    "phi0": before[1]["phi_eq"][index].value,
                }
            elif index:
                expected = {"k": 0, "n": 1, "d": 1, "phi0": 0}
            elif p is potentials[0]:
                expected = {"k": 2.1, "n": 3, "d": -1, "phi0": -0.5}
            else:
                expected = {
                    "k": 1.4 * float(p.parameters["k"].reshape(-1)[0]),
                    "n": int(p.parameters["n"].reshape(-1)[0]),
                    "d": 1,
                    "phi0": float(p.parameters["phi_eq"].reshape(-1)[0]),
                }
            assert dict(force.params[label]) == pytest.approx(expected)
    assert tuple(top.dihedrals) == connections
    assert [tuple(c.connection_members) for c in top.dihedrals] == members
    for p, original in zip(potentials, before):
        for name, value in p.parameters.items():
            np.testing.assert_array_equal(value, original[name])
    disabled = bundle(top, include_torsions=False)
    assert not torsion_forces(disabled)
    assert disabled.type_labels == ff.type_labels
    assert disabled.disabled_term_counts["dihedrals"] == 4
    assert set(disabled.hoomd_forces[-1].nlist.exclusions) == {
        "bond",
        "angle",
        "dihedral",
    }


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize(
    "name, value, message",
    [
        ("k", u.unyt_array([], "kcal/mol"), "nonempty"),
        ("k", u.unyt_array([[1]], "kcal/mol"), "one-dimensional"),
        ("k", 1 * u.kcal / u.mol, "aligned"),
        ("n", u.unyt_array([1, 2], "dimensionless"), "aligned"),
        ("n", u.unyt_array([1, 2.5, 3], "dimensionless"), "positive integer"),
        ("n", u.unyt_array([1, 0, 3], "dimensionless"), "positive integer"),
        ("n", u.unyt_array([1, 2**31, 3], "dimensionless"), "int32"),
        ("phi_eq", u.unyt_array([0, float("inf"), 0], "rad"), "finite"),
        ("k", u.unyt_array([1, float("nan"), 0], "kcal/mol"), "finite"),
        ("k", u.unyt_array([1, 1, 0], "nm"), "incompatible units"),
        ("k", u.unyt_array([1e308, 1, 0], "kcal/mol"), "float range"),
        ("k", None, "quantity"),
    ],
)
def test_invalid_native_arrays_even_when_disabled(
    enabled, name, value, message
):
    top = topology()
    p = periodic()
    top.dihedrals[0].dihedral_type = p
    p.parameters[name] = value
    with pytest.raises(ValueError, match=f"dihedrals at sites.*{message}"):
        bundle(top, include_torsions=enabled)


def test_native_units_scalar_and_scaling_range():
    top = topology()
    p = periodic(k=1, n=2, phase=0.5)
    top.dihedrals[0].dihedral_type = p
    p.parameters["k"] = p.parameters["k"].to("kJ/mol")
    p.parameters["phi_eq"] = p.parameters["phi_eq"].to("degree")
    assert dict(
        torsion_forces(bundle(top))[0].params["aa_dihedral_0"]
    ) == pytest.approx({"k": 14, "n": 2, "d": 1, "phi0": 0.5})
    for coefficient, scale, expected in [
        (1e308, 1e-308, 2),
        (np.nextafter(0.0, 1.0), 0.5, np.nextafter(0.0, 1.0)),
    ]:
        p.parameters["k"] = coefficient * u.kcal / u.mol
        actual = torsion_forces(bundle(top, bonded_scale=scale))[0].params[
            "aa_dihedral_0"
        ]["k"]
        assert actual == pytest.approx(expected, rel=1e-14, abs=0)
    p.parameters["k"] = 1e-300 * u.kcal / u.mol
    with pytest.raises(ValueError, match="float range"):
        bundle(top, bonded_scale=1e-300, include_torsions=False)
    p.parameters["k"] *= 0
    assert (
        torsion_forces(bundle(top, bonded_scale=1e-300))[0].params[
            "aa_dihedral_0"
        ]["k"]
        == 0
    )


def reference_force_energy_forces(top, xyz, scale):
    mm = pytest.importorskip("openmm")
    force = mm.PeriodicTorsionForce()
    index = {s: i for i, s in enumerate(top.sites)}
    for connection in top.dihedrals:
        p = connection.dihedral_type.parameters
        group = [index[s] for s in connection.connection_members]
        if "phi_eq" in p:
            coefficients = np.atleast_1d(p["k"].to_value("kcal/mol"))
            ns = np.atleast_1d(p["n"].value)
            phases = np.atleast_1d(p["phi_eq"].to_value("rad"))
            for k, n, phase in zip(coefficients, ns, phases):
                force.addTorsion(*group, int(n), phase, float(k) * scale)
        else:
            # OpenMM lacks HOOMD's d=-1 form. A phase shift and signed
            # amplitude would alter the additive constant, so use a phase shift.
            phase = float(p["phi0"].to_value("rad"))
            if int(p["d"]) == -1:
                phase += np.pi
            force.addTorsion(
                *group,
                int(p["n"]),
                phase,
                float(p["k"].to_value("kcal/mol")) * scale / 2,
            )
    system = mm.System()
    for _ in top.sites:
        system.addParticle(1)
    system.addForce(force)
    integrator = mm.VerletIntegrator(0.001)
    context = mm.Context(
        system, integrator, mm.Platform.getPlatformByName("Reference")
    )
    # Match numerical reduced coordinates and energy, without imposing OpenMM
    # physical units on HOOMD's angstrom and kcal/mol references.
    context.setPositions(xyz)
    state = context.getState(getEnergy=True, getForces=True)
    return state.getPotentialEnergy().value_in_unit(
        mm.unit.kilojoule_per_mole
    ), state.getForces(asNumpy=True).value_in_unit(
        mm.unit.kilojoule_per_mole / mm.unit.nanometer
    )


@pytest.mark.parametrize("mirror", [1, -1])
@pytest.mark.parametrize("scale", [0.3, 7.0])
def test_cpu_mixed_fourier_openmm_parity(mirror, scale):
    top, _ = mixed_topology()
    positions = np.array(
        [[0, 0, 0], [1.3, 0.1, 0], [1.8, 1.1, 0.1], [2.7, 1.4, 0.8]]
    )
    xyz = np.concatenate([positions + [0, 0, 4 * i] for i in range(4)])
    xyz[:, 2] *= mirror
    ff = bundle(
        top,
        bonded_scale=scale,
        conservative=True,
        repulsion=0,
        include_bonds=False,
        include_angles=False,
    )
    simulation = evaluate(top, ff, xyz)
    expected_energy, expected_forces = reference_force_energy_forces(
        top, xyz, scale
    )
    forces = torsion_forces(ff)
    assert sum(f.energy for f in forces) == pytest.approx(
        expected_energy, abs=1e-10
    )
    np.testing.assert_allclose(
        sum(f.forces for f in forces), expected_forces, atol=1e-10
    )
    snapshot = simulation.state.get_snapshot()
    assert snapshot.dihedrals.N == top.n_dihedrals
    np.testing.assert_array_equal(
        snapshot.dihedrals.group, np.arange(16).reshape(4, 4)
    )
    np.testing.assert_array_equal(snapshot.dihedrals.typeid, np.arange(4))
    assert simulation.timestep == 0


def test_sage_adapter_to_cpu_fourier():
    pytest.importorskip("openff.toolkit")
    from flowermd.internal.sage_gmso import assign_sage_parameters
    from flowermd.tests.utils.test_uff_gmso import inputs

    raw, molecule, mapping = inputs("CC(=O)NC")
    top, _ = assign_sage_parameters(
        raw, molecule, atom_map=mapping, assign_nonbonded=False
    )
    ff = bundle(
        top,
        epsilon_weighting=False,
        include_impropers=False,
        include_bonds=False,
        include_angles=False,
        conservative=True,
        repulsion=0,
        bonded_scale=0.4,
    )
    xyz = top.positions.to_value("angstrom") + np.random.default_rng(
        331
    ).normal(0, 0.12, (top.n_sites, 3))
    simulation = evaluate(top, ff, xyz)
    expected_energy, expected_forces = reference_force_energy_forces(
        top, xyz, 0.4
    )
    forces = torsion_forces(ff)
    assert len(forces) > 1
    assert sum(f.energy for f in forces) == pytest.approx(
        expected_energy, abs=1e-9
    )
    np.testing.assert_allclose(
        sum(f.forces for f in forces), expected_forces, atol=1e-9
    )
    assert ff.disabled_term_counts["impropers"] == top.n_impropers
    assert simulation.state.get_snapshot().dihedrals.N == top.n_dihedrals
    with pytest.raises(NotImplementedError, match="no improper force backend"):
        bundle(top, epsilon_weighting=False)
