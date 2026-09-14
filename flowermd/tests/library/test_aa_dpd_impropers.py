"""Compare periodic improper execution with public OpenFF/OpenMM export."""

from copy import deepcopy

import gmso
import hoomd
import numpy as np
import pytest
import unyt as u
from gmso.lib.potential_templates import PotentialTemplateLibrary

from flowermd.internal.aa_snapshot import create_all_atom_frame
from flowermd.internal.openff_gmso import assign_openff_parameters
from flowermd.internal.uff_gmso import assign_uff_parameters
from flowermd.tests.library.test_aa_dpd import bundle
from flowermd.tests.library.test_aa_dpd_fourier import periodic
from flowermd.tests.utils.test_uff_gmso import inputs


def run(top, ff, xyz_nm, box_nm):
    frame = create_all_atom_frame(
        top,
        type_labels=ff.type_labels,
        positions_nm=xyz_nm,
        box_lengths_nm=box_nm,
    )
    snapshot = hoomd.Snapshot.from_gsd_frame(
        frame, hoomd.communicator.Communicator()
    )
    simulation = hoomd.Simulation(device=hoomd.device.CPU(), seed=14)
    simulation.create_state_from_snapshot(snapshot)
    simulation.operations.integrator = hoomd.md.Integrator(
        dt=0.001,
        methods=[hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())],
        forces=ff.hoomd_forces,
    )
    simulation.run(0)
    return simulation, frame


def exported_reference(molecule, ff):
    mm = pytest.importorskip("openmm")
    off = pytest.importorskip("openff.toolkit")
    unit = pytest.importorskip("openff.units").unit
    mol = off.Molecule.from_rdkit(molecule, hydrogens_are_explicit=True)
    mol.partial_charges = np.zeros(mol.n_atoms) * unit.elementary_charge
    copied = deepcopy(ff)
    copied.deregister_parameter_handler("Constraints")
    system = copied.create_openmm_system(
        mol.to_topology(), charge_from_molecules=[mol]
    )
    edges = {
        frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
        for b in molecule.GetBonds()
    }
    terms = {"dihedrals": [], "impropers": []}
    for force in system.getForces():
        if isinstance(force, mm.PeriodicTorsionForce):
            for i in range(force.getNumTorsions()):
                parameters = force.getTorsionParameters(i)
                kind = (
                    "dihedrals"
                    if all(
                        frozenset(pair) in edges
                        for pair in zip(parameters[:3], parameters[1:4])
                    )
                    else "impropers"
                )
                terms[kind].append(parameters)
    return terms


def reference(terms, frame, scale):
    mm = pytest.importorskip("openmm")
    system = mm.System()
    for _ in range(frame.particles.N):
        system.addParticle(1)
    force = mm.PeriodicTorsionForce()
    force.setUsesPeriodicBoundaryConditions(True)
    for parameters in terms:
        force.addTorsion(*parameters)
    system.addForce(force)
    lengths = frame.configuration.box[:3].astype(float) / 10
    system.setDefaultPeriodicBoxVectors(*np.diag(lengths))
    integrator = mm.VerletIntegrator(0.001)
    context = mm.Context(
        system, integrator, mm.Platform.getPlatformByName("Reference")
    )
    context.setPositions(frame.particles.position.astype(float) / 10)
    state = context.getState(getEnergy=True, getForces=True)
    return (
        state.getPotentialEnergy().value_in_unit(mm.unit.kilocalorie_per_mole)
        * scale,
        state.getForces(asNumpy=True).value_in_unit(
            mm.unit.kilocalorie_per_mole / mm.unit.angstrom
        )
        * scale,
    )


@pytest.fixture(params=[False, True])
def assigned(request):
    off = pytest.importorskip("openff.toolkit")
    unit = pytest.importorskip("openff.units").unit
    raw, molecule, mapping = inputs("CC(=O)NC")
    ff = off.ForceField("openff-2.3.0.offxml")
    if request.param:
        for handler in ("ProperTorsions", "ImproperTorsions"):
            for parameter in ff[handler].parameters:
                parameter.k = [
                    (-abs(k) if i % 2 == 0 else 0 * k)
                    for i, k in enumerate(parameter.k)
                ]
                parameter.phase = [
                    (37 + i * 91) * unit.degree for i in range(len(parameter.k))
                ]
                parameter.idivf = [2 + i for i in range(len(parameter.k))]
    top, _ = assign_openff_parameters(
        raw, molecule, atom_map=mapping, force_field=ff, assign_nonbonded=False
    )
    return top, exported_reference(molecule, ff)


@pytest.mark.parametrize(
    "proper, improper",
    [(True, True), (True, False), (False, True), (False, False)],
)
@pytest.mark.parametrize(
    "mirror, scale, crossing", [(1, 0.4, False), (-1, 2.3, True)]
)
def test_public_openff_openmm_parity_and_ablations(
    assigned, proper, improper, mirror, scale, crossing
):
    top, terms = assigned
    before = [
        (c, tuple(c.connection_members), deepcopy(c.connection_type.parameters))
        for c in top.connections
    ]
    ff = bundle(
        top,
        epsilon_weighting=False,
        conservative=True,
        repulsion=0,
        include_bonds=False,
        include_angles=False,
        include_torsions=proper,
        include_impropers=improper,
        bonded_scale=scale,
    )
    xyz = top.positions.to_value("nm") + np.random.default_rng(863).normal(
        0, 0.02, (top.n_sites, 3)
    )
    xyz[:, 2] *= mirror
    xyz += [2.98 if crossing else 1.5, 2, 2.5]
    simulation, frame = run(top, ff, xyz, [3, 4, 5])
    if crossing:
        assert np.unique(frame.particles.image[:, 0]).size > 1
    labels = list(ff.type_labels["dihedrals"].values()) + list(
        ff.type_labels["impropers"].values()
    )
    assert frame.dihedrals.types == labels
    indices = {s: i for i, s in enumerate(top.sites)}
    expected_groups = [
        [indices[s] for s in c.connection_members]
        for c in (*top.dihedrals, *top.impropers)
    ]
    np.testing.assert_array_equal(frame.dihedrals.group, expected_groups)
    original_exclusions = {
        frozenset(
            (
                indices[c.connection_members[0]],
                indices[c.connection_members[-1]],
            )
        )
        for kind in ("bonds", "angles", "dihedrals")
        for c in getattr(top, kind)
    }
    routed_exclusions = {
        frozenset((int(group[0]), int(group[-1])))
        for kind in ("bonds", "angles", "dihedrals")
        for group in getattr(frame, kind).group
    }
    assert routed_exclusions == original_exclusions
    expected_ids = [
        labels.index(ff.type_labels[kind][c.connection_type])
        for kind in ("dihedrals", "impropers")
        for c in getattr(top, kind)
    ]
    np.testing.assert_array_equal(frame.dihedrals.typeid, expected_ids)
    assert frame.impropers.N == 0
    assert list(ff.forces_by_category) == [
        "bonds",
        "angles",
        "dihedrals",
        "impropers",
        "pair",
    ]
    assert all(isinstance(v, tuple) for v in ff.forces_by_category.values())
    assert list(ff.hoomd_forces) == [
        f for group in ff.forces_by_category.values() for f in group
    ]
    assert set(ff.forces_by_category["pair"][0].nlist.exclusions) == {
        "bond",
        "angle",
        "dihedral",
    }
    expected_total_energy = 0
    expected_total_force = np.zeros((top.n_sites, 3))
    for kind, enabled in (("dihedrals", proper), ("impropers", improper)):
        forces = ff.forces_by_category[kind]
        assert bool(forces) is enabled
        if not enabled:
            assert ff.disabled_term_counts[kind] == len(getattr(top, kind))
        expected_energy, expected_force = reference(
            terms[kind] if enabled else [], frame, scale
        )
        actual_force = sum(
            (f.forces for f in forces), start=np.zeros_like(expected_force)
        )
        assert sum(f.energy for f in forces) == pytest.approx(
            expected_energy, abs=2e-9
        )
        np.testing.assert_allclose(
            actual_force, expected_force, atol=2e-8, rtol=2e-9
        )
        for force in forces:
            assert set(force.params) == set(labels)
            other = "impropers" if kind == "dihedrals" else "dihedrals"
            for label in ff.type_labels[other].values():
                assert dict(force.params[label]) == {
                    "k": 0,
                    "n": 1,
                    "d": 1,
                    "phi0": 0,
                }
        expected_total_energy += expected_energy
        expected_total_force += expected_force
    assert sum(f.energy for f in ff.hoomd_forces) == pytest.approx(
        expected_total_energy, abs=2e-9
    )
    np.testing.assert_allclose(
        sum(f.forces for f in ff.hoomd_forces),
        expected_total_force,
        atol=2e-8,
        rtol=2e-9,
    )
    for connection, members, params in before:
        assert tuple(connection.connection_members) == members
        for name, value in params.items():
            np.testing.assert_array_equal(
                connection.connection_type.parameters[name], value
            )
    assert simulation.timestep == 0


def test_mixed_scalar_and_unequal_array_components(assigned):
    top, _ = assigned
    legacy = gmso.DihedralType.from_template(
        PotentialTemplateLibrary()["HOOMDPeriodicDihedralPotential"],
        {
            "k": -0.3 * u.kcal / u.mol,
            "n": 2 * u.dimensionless,
            "d": -1 * u.dimensionless,
            "phi0": 0.4 * u.rad,
        },
        name="same",
    )
    top.dihedrals[0].dihedral_type = legacy
    top.dihedrals[1].dihedral_type = periodic(
        k=(1, -0.3, 0), n=(1, 2, 4), phase=(0.2, -0.4, 0.9)
    )
    p = top.impropers[0].improper_type.clone()
    p.name = top.impropers[1].improper_type.name
    p.parameters = {
        "k": [1, -0.4, 0, 0.3] * u.kcal / u.mol,
        "n": [1, 2, 3, 4] * u.dimensionless,
        "phi_eq": [0.2, -0.3, 0.4, -0.5] * u.rad,
    }
    top.impropers[0].improper_type = p
    scalar = p.clone()
    scalar.parameters = {
        "k": -0.2 * u.kcal / u.mol,
        "n": 3 * u.dimensionless,
        "phi_eq": -0.6 * u.rad,
    }
    top.impropers[1].improper_type = scalar
    ff = bundle(
        top,
        epsilon_weighting=False,
        include_bonds=False,
        include_angles=False,
        bonded_scale=0.7,
    )
    assert len(ff.forces_by_category["dihedrals"]) == 3
    assert len(ff.forces_by_category["impropers"]) == 4
    for kind in ("dihedrals", "impropers"):
        for component, force in enumerate(ff.forces_by_category[kind]):
            for potential, label in ff.type_labels[kind].items():
                native = "phi_eq" in potential.parameters
                ks = np.atleast_1d(
                    potential.parameters["k"].to_value("kcal/mol")
                )
                expected = (
                    0.7 * (2 if native else 1) * ks[component]
                    if component < len(ks)
                    else 0
                )
                assert force.params[label]["k"] == pytest.approx(expected)
    mm = pytest.importorskip("openmm")

    xyz = (
        top.positions.to_value("nm")
        + np.random.default_rng(8).normal(0, 0.01, (top.n_sites, 3))
        + 1.5
    )
    simulation, frame = run(top, ff, xyz, [3, 3, 3])
    indices = {site: i for i, site in enumerate(top.sites)}
    for kind in ("dihedrals", "impropers"):
        terms = []
        for connection in getattr(top, kind):
            group = [indices[site] for site in connection.connection_members]
            p = connection.connection_type.parameters
            if "phi_eq" in p:
                components = zip(
                    np.atleast_1d(p["k"].to_value("kcal/mol")),
                    np.atleast_1d(p["n"].value),
                    np.atleast_1d(p["phi_eq"].to_value("rad")),
                )
            else:
                phase = float(p["phi0"].to_value("rad")) + (
                    np.pi if int(p["d"]) == -1 else 0
                )
                components = [
                    (float(p["k"].to_value("kcal/mol")) / 2, int(p["n"]), phase)
                ]
            for k, n, phase in components:
                terms.append(
                    [
                        *group,
                        int(n),
                        float(phase),
                        float(k) * mm.unit.kilocalorie_per_mole,
                    ]
                )
        energy, force = reference(terms, frame, 0.7)
        assert sum(
            f.energy for f in ff.forces_by_category[kind]
        ) == pytest.approx(energy, abs=1e-9)
        np.testing.assert_allclose(
            sum(f.forces for f in ff.forces_by_category[kind]),
            force,
            atol=2e-8,
            rtol=2e-9,
        )
    assert simulation.timestep == 0


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize(
    "name, value, message",
    [
        ("k", [1] * u.kcal / u.mol, "aligned"),
        ("n", [0] * u.dimensionless, "positive integer|aligned"),
        ("phi_eq", [float("nan")] * u.rad, "finite"),
        ("k", [1] * u.nm, "incompatible"),
        ("k", [] * u.kcal / u.mol, "nonempty"),
    ],
)
def test_invalid_improper_arrays_even_when_disabled(
    assigned, enabled, name, value, message
):
    top, _ = assigned
    p = top.impropers[0].improper_type
    p.parameters = {
        "k": [1, 2] * u.kcal / u.mol,
        "n": [1, 2] * u.dimensionless,
        "phi_eq": [0, 0.3] * u.rad,
    }
    p.parameters[name] = value
    with pytest.raises(ValueError, match=f"impropers at sites.*{message}"):
        bundle(top, epsilon_weighting=False, include_impropers=enabled)


def test_uff_unknown_and_untyped_storage():
    raw, molecule, mapping = inputs("CC(=O)NC")
    top, _ = assign_uff_parameters(raw, molecule, atom_map=mapping)
    with pytest.raises(NotImplementedError, match="UFF out-of-plane"):
        bundle(top)
    top.impropers[0].improper_type = None
    top.impropers[1].improper_type = gmso.ImproperType(
        expression="k*phi**4",
        independent_variables={"phi"},
        parameters={"k": 1 * u.kcal / u.mol},
    )
    ff = bundle(top, include_impropers=False)
    out = create_all_atom_frame(
        top,
        type_labels=ff.type_labels,
        positions_nm=top.positions.to_value("nm") + 1.5,
        box_lengths_nm=[3] * 3,
    )
    assert out.impropers.N == top.n_impropers
    assert out.dihedrals.N == top.n_dihedrals
    assert ff.forces_by_category["impropers"] == ()
    top.impropers[0].improper_type = top.impropers[1].improper_type
    with pytest.raises(NotImplementedError, match="no improper force backend"):
        bundle(top)


def test_unknown_disabled_proper_has_improper_force_padding(assigned):
    top, _ = assigned
    top.dihedrals[0].dihedral_type = gmso.DihedralType(
        expression="k*phi**4",
        independent_variables={"phi"},
        parameters={"k": 1 * u.kcal / u.mol},
    )
    ff = bundle(
        top,
        epsilon_weighting=False,
        include_torsions=False,
        include_bonds=False,
        include_angles=False,
    )
    simulation, frame = run(
        top, ff, top.positions.to_value("nm") + 1.5, [3] * 3
    )
    assert ff.forces_by_category["dihedrals"] == ()
    assert frame.dihedrals.N == top.n_dihedrals + top.n_impropers
    for force in ff.forces_by_category["impropers"]:
        for label in ff.type_labels["dihedrals"].values():
            assert force.params[label]["k"] == 0
    assert simulation.timestep == 0
