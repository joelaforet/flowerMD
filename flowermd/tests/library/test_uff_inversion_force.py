"""Compare Wilson energies, Cartesian forces and virials with independent checks."""

import time
from itertools import combinations
from types import SimpleNamespace

import gmso
import hoomd
import numpy as np
import pytest
import unyt as u

from flowermd.internal.aa_snapshot import create_all_atom_frame
from flowermd.internal.uff_gmso import assign_uff_parameters
from flowermd.internal.uff_inversion import (
    UFFInversionForce,
    wilson_energy_forces,
)
from flowermd.tests.library.test_aa_dpd import bundle
from flowermd.tests.utils.test_uff_gmso import inputs

Chem = pytest.importorskip("rdkit.Chem")
uff = pytest.importorskip("rdkit.Chem.rdForceFieldHelpers")


def typed(smiles):
    top, molecule, mapping = inputs(smiles)
    result, _ = assign_uff_parameters(top, molecule, atom_map=mapping)
    return result, molecule


def prepare(
    top, xyz, scale=0.7, enabled=True, box=(30, 40, 50), zero_pairs=True
):
    ff = bundle(
        top,
        conservative=True,
        repulsion=0 if zero_pairs else 40,
        bonded_scale=scale,
        include_bonds=False,
        include_angles=False,
        include_torsions=False,
        include_impropers=enabled,
    )
    frame = create_all_atom_frame(
        top,
        type_labels=ff.type_labels,
        positions_nm=xyz / 10,
        box_lengths_nm=np.array(box) / 10,
    )
    sim = hoomd.Simulation(device=hoomd.device.CPU(), seed=31)
    sim.create_state_from_snapshot(frame)
    sim.always_compute_pressure = True
    sim.operations.integrator = hoomd.md.Integrator(
        dt=1e-5,
        methods=[hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())],
        forces=ff.hoomd_forces,
    )
    return sim, ff, frame


def unwrapped(frame):
    lengths = frame.configuration.box[:3].astype(float)
    return (
        frame.particles.position.astype(float)
        + frame.particles.image * lengths
        + lengths / 2
    )


def angle(a, b):
    return np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b))


def non_inversion_energy(molecule, xyz):
    """Evaluate RDKit's bond, full UFF angle and SP2 torsion terms."""
    energy = 0.0
    for bond in molecule.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        k, target = uff.GetUFFBondStretchParams(molecule, i, j)
        energy += 0.5 * k * (np.linalg.norm(xyz[j] - xyz[i]) - target) ** 2
    for atom in molecule.GetAtoms():
        center = atom.GetIdx()
        for a, b in combinations((n.GetIdx() for n in atom.GetNeighbors()), 2):
            k, degrees = uff.GetUFFAngleBendParams(molecule, a, center, b)
            theta = angle(xyz[a] - xyz[center], xyz[b] - xyz[center])
            if atom.GetHybridization() == Chem.HybridizationType.SP2:
                energy += k / 9 * (1 - np.cos(3 * theta))
            else:
                target = np.deg2rad(degrees)
                c2 = 1 / (4 * np.sin(target) ** 2)
                c1 = -4 * c2 * np.cos(target)
                c0 = c2 * (2 * np.cos(target) ** 2 + 1)
                energy += k * (c0 + c1 * np.cos(theta) + c2 * np.cos(2 * theta))
    for bond in molecule.GetBonds():
        b, c = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        terms = [
            (a.GetIdx(), b, c, d.GetIdx())
            for a in molecule.GetAtomWithIdx(b).GetNeighbors()
            if a.GetIdx() != c
            for d in molecule.GetAtomWithIdx(c).GetNeighbors()
            if d.GetIdx() not in (b, a.GetIdx())
        ]
        for group in terms:
            assert all(
                molecule.GetAtomWithIdx(i).GetHybridization()
                == Chem.HybridizationType.SP2
                for i in (b, c)
            )
            k = uff.GetUFFTorsionParams(molecule, *group) / len(terms)
            points = xyz[list(group)]
            left, middle, right = np.diff(points, axis=0)
            phi = angle(np.cross(left, middle), np.cross(middle, right))
            energy += 0.5 * k * (1 - np.cos(2 * phi))
    return energy


def isolated_rdkit(molecule, xyz):
    mol = Chem.Mol(molecule)
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for i, position in enumerate(xyz):
        conformer.SetAtomPosition(i, position)
    mol.RemoveAllConformers()
    mol.AddConformer(conformer)
    native = uff.UFFGetMoleculeForceField(mol, vdwThresh=0)
    energy = native.CalcEnergy(xyz.ravel().tolist()) - non_inversion_energy(
        mol, xyz
    )
    force = -np.array(native.CalcGrad(xyz.ravel().tolist())).reshape(xyz.shape)
    step = 1e-6
    for index in np.ndindex(xyz.shape):
        plus, minus = xyz.copy(), xyz.copy()
        plus[index] += step
        minus[index] -= step
        force[index] += (
            non_inversion_energy(mol, plus) - non_inversion_energy(mol, minus)
        ) / (2 * step)
    return energy, force


@pytest.mark.parametrize(
    "smiles", ["C=O", "C=C", "P", "[AsH3]", "[SbH3]", "[BiH3]"]
)
@pytest.mark.parametrize(
    "mirror,scale,planar,crossing",
    [(1, 0.3, True, False), (1, 1.7, False, True), (-1, 0.7, False, False)],
)
def test_rdkit_isolated_energy_force_parity(
    smiles, mirror, scale, planar, crossing
):
    top, molecule = typed(smiles)
    assert top.n_impropers in (3, 6)
    xyz = top.positions.to_value("angstrom") + np.random.default_rng(
        202
    ).normal(0, 0.17, (top.n_sites, 3))
    if planar:
        xyz[:, 2] = 0
    xyz[:, 2] *= mirror
    xyz += [29.9 if crossing else 15, 20, 25]
    sim, ff, frame = prepare(top, xyz, scale)
    sim.run(0)
    force = ff.forces_by_category["impropers"][0]
    assert isinstance(force, UFFInversionForce)
    assert frame.impropers.N == top.n_impropers
    actual_xyz = unwrapped(frame)
    energy, expected = isolated_rdkit(molecule, actual_xyz)
    assert force.energy == pytest.approx(scale * energy, abs=3e-8)
    np.testing.assert_allclose(
        force.forces, scale * expected, atol=3e-5, rtol=3e-6
    )
    np.testing.assert_allclose(force.forces.sum(axis=0), 0, atol=1e-10)
    relative = actual_xyz - actual_xyz.mean(axis=0)
    np.testing.assert_allclose(
        np.cross(relative, force.forces).sum(axis=0), 0, atol=1e-9
    )
    if crossing:
        assert np.unique(frame.particles.image[:, 0]).size > 1


@pytest.mark.parametrize("c1", [-1.0, 0.0])
def test_singular_and_smooth_perpendicular_cases(c1):
    vectors = np.eye(3)
    if c1:
        with pytest.raises(ValueError, match="nondifferentiable"):
            wilson_energy_forces(vectors, [1, 1, c1, 1])
    else:
        energy, forces = wilson_energy_forces(vectors, [1, 1, 0, 1])
        assert energy == 0
        np.testing.assert_array_equal(forces, 0)


@pytest.mark.parametrize(
    "vectors",
    [
        np.zeros((3, 3)),
        [[1, 0, 0], [2, 0, 0], [0, 1, 0]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 0]],
    ],
)
def test_undefined_geometry_and_zero_stiffness(vectors):
    vectors = np.array(vectors, dtype=float)
    with pytest.raises(ValueError, match="undefined"):
        wilson_energy_forces(vectors, [1, 1, -1, 0])
    energy, force = wilson_energy_forces(vectors, [0, 1, -1, 0])
    assert energy == 0
    np.testing.assert_array_equal(force, 0)


def test_analytic_gradients_and_strain_virials():
    top, _ = typed("P")
    xyz = (
        np.array(
            [[0, 0, 0], [1.4, 0.1, 0.2], [-0.4, 1.3, -0.2], [-0.3, -0.6, 1.2]]
        )
        + 15
    )
    sim, ff, frame = prepare(top, xyz)
    sim.run(0)
    force = ff.forces_by_category["impropers"][0]
    actual = unwrapped(frame)
    groups = force._groups.copy()
    coefficients = force._coefficients.copy()

    def energy(positions):
        total = 0.0
        for group, (k, c0, c1, c2) in zip(groups, coefficients):
            a, b, out = positions[group[1:]] - positions[group[0]]
            normal = np.cross(a, b)
            omega = np.arctan2(
                np.dot(normal, out), np.linalg.norm(np.cross(normal, out))
            )
            total += k * (c0 + c1 * np.cos(omega) + c2 * np.cos(2 * omega))
        return total

    numerical = np.zeros_like(actual)
    step = 1e-6
    for index in np.ndindex(actual.shape):
        plus, minus = actual.copy(), actual.copy()
        plus[index] += step
        minus[index] -= step
        numerical[index] = -(energy(plus) - energy(minus)) / (2 * step)
    np.testing.assert_allclose(force.forces, numerical, atol=1e-7, rtol=1e-7)
    relative = actual - actual[0]
    strain = []
    for i, j in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)):
        plus, minus = relative.copy(), relative.copy()
        plus[:, i] += step * relative[:, j]
        minus[:, i] -= step * relative[:, j]
        strain.append(-(energy(plus) - energy(minus)) / (2 * step))
    np.testing.assert_allclose(
        force.virials.sum(axis=0), strain, atol=1e-7, rtol=1e-7
    )
    assert np.sum(force.virials[:, [0, 3, 5]]) == pytest.approx(0, abs=1e-10)
    for _ in range(3):
        force.set_forces(0)
        np.testing.assert_allclose(
            force.forces, numerical, atol=1e-7, rtol=1e-7
        )
    start = time.perf_counter()
    for _ in range(100):
        force.set_forces(0)
    elapsed = time.perf_counter() - start
    print(f"100 callbacks for three inversion terms: {elapsed:.6f} seconds")


def test_sorting_permutation_and_short_dynamics():
    top, _ = typed("CC.C=O")
    xyz = (
        top.positions.to_value("angstrom")
        + np.random.default_rng(9).normal(0, 0.1, (top.n_sites, 3))
        + 15
    )
    sim, ff, _ = prepare(top, xyz)
    force = ff.forces_by_category["impropers"][0]
    for tuner in sim.operations.tuners:
        if isinstance(tuner, hoomd.tune.ParticleSorter):
            tuner.trigger = 1
    sim.run(0)
    sim.run(4)
    with sim.state.cpu_local_snapshot as snapshot:
        order = np.array(snapshot.particles.tag, copy=True)
    assert not np.array_equal(order, np.arange(top.n_sites))
    state = sim.state.get_snapshot()
    expected_force = np.zeros((top.n_sites, 3))
    expected_energy = 0
    for group, coefficients in zip(force._groups, force._coefficients):
        vectors = (
            state.particles.position[group[1:]]
            - state.particles.position[group[0]]
        )
        vectors -= np.array(state.configuration.box[:3]) * np.rint(
            vectors / np.array(state.configuration.box[:3])
        )
        energy, forces = wilson_energy_forces(vectors, coefficients)
        np.add.at(expected_force, group, forces)
        expected_energy += energy
    np.testing.assert_allclose(force.forces, expected_force, atol=1e-10)
    assert force.energy == pytest.approx(expected_energy, abs=1e-10)
    assert np.isfinite(state.particles.position).all()
    # A fresh force can also attach through HOOMD's conservative FIRE path.
    fire_top, _ = typed("C=O")
    fire_xyz = (
        fire_top.positions.to_value("angstrom")
        + np.random.default_rng(3).normal(0, 0.1, (fire_top.n_sites, 3))
        + 15
    )
    fire_sim, fire_ff, _ = prepare(fire_top, fire_xyz)
    fire_sim.operations.integrator = hoomd.md.minimize.FIRE(
        dt=0.0001,
        force_tol=1e-5,
        angmom_tol=1e-5,
        energy_tol=1e-8,
        methods=[hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())],
        forces=fire_ff.hoomd_forces,
    )
    fire_sim.run(0)
    before = fire_ff.forces_by_category["impropers"][0].energy
    fire_sim.run(20)
    assert fire_ff.forces_by_category["impropers"][0].energy <= before + 1e-10


def test_ablation_and_copied_parameters():
    top, _ = typed("C=O")
    original = top.impropers[0].improper_type
    distinct = original.clone()
    distinct.parameters["k"] *= 0.5
    top.impropers[0].improper_type = distinct
    xyz = top.positions.to_value("angstrom") + 15
    sim, ff, frame = prepare(top, xyz)
    assert original.name == distinct.name
    assert len(ff.type_labels["impropers"]) == 2
    assert (
        ff.type_labels["impropers"][original]
        != ff.type_labels["impropers"][distinct]
    )
    disabled, off, off_frame = prepare(top, xyz, enabled=False)
    for kind in ("bonds", "angles", "dihedrals", "impropers"):
        np.testing.assert_array_equal(
            getattr(frame, kind).group, getattr(off_frame, kind).group
        )
        np.testing.assert_array_equal(
            getattr(frame, kind).typeid, getattr(off_frame, kind).typeid
        )
    assert off.forces_by_category["impropers"] == ()
    assert set(ff.hoomd_forces[-1].nlist.exclusions) == set(
        off.hoomd_forces[-1].nlist.exclusions
    )
    force = ff.forces_by_category["impropers"][0]
    coefficients = force._coefficients.copy()
    for potential in top.improper_types:
        potential.parameters["k"] *= 2
    np.testing.assert_array_equal(force._coefficients, coefficients)
    sim.run(0)
    disabled.run(0)
    assert np.isfinite(force.energy)


def test_attachment_scope_guards():
    force = UFFInversionForce([[0, 1, 2, 3]], [[1, 1, -1, 0]], 4)

    class SimulationStub:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    simulation = SimulationStub(device=object())
    force._simulation = simulation
    with pytest.raises(RuntimeError, match="CPU"):
        force._attach_hook()

    class Device:
        communicator = SimpleNamespace(num_ranks=2)

    # Exercise the rank guard without requiring a multi-rank allocation.
    original = hoomd.device.CPU
    try:
        hoomd.device.CPU = Device
        simulation = SimulationStub(device=Device())
        force._simulation = simulation
        with pytest.raises(RuntimeError, match="single MPI rank"):
            force._attach_hook()
    finally:
        hoomd.device.CPU = original
    for box in (hoomd.Box(30, 30, 0), hoomd.Box(30, 30, 30, xy=0.1)):
        simulation = SimulationStub(state=SimpleNamespace(box=box))
        force._simulation = simulation
        with pytest.raises(ValueError, match="orthorhombic 3D"):
            force._check_box()


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize(
    "name,value",
    [
        ("k", -1 * u.kcal / u.mol),
        ("k", 1 * u.nm),
        ("c1", float("nan") * u.dimensionless),
    ],
)
def test_invalid_coefficients_disabled_too(enabled, name, value):
    top, _ = typed("C=O")
    top.impropers[0].improper_type.parameters[name] = value
    with pytest.raises(ValueError, match="impropers at sites"):
        bundle(top, include_impropers=enabled)


@pytest.mark.parametrize("c1", [-1.0, -2.4492935982947064e-16])
def test_one_sided_cusp_and_regular_near_perpendicular(c1):
    exact = np.eye(3)
    with pytest.raises(ValueError, match="nondifferentiable"):
        wilson_energy_forces(exact, [1, 0, c1, 0])
    step = 1e-6
    plus, minus = exact.copy(), exact.copy()
    plus[2, 0] = step
    minus[2, 0] = -step
    right = wilson_energy_forces(plus, [1, 0, c1, 0])[0] / step
    left = wilson_energy_forces(minus, [1, 0, c1, 0])[0] / (-step)
    assert right == pytest.approx(c1, rel=1e-10, abs=0)
    assert left == pytest.approx(-c1, rel=1e-10, abs=0)
    near = exact.copy()
    near[2, 0] = 1e-8
    energy, forces = wilson_energy_forces(near, [1, 0, c1, 0])
    assert np.isfinite(energy) and np.isfinite(forces).all()
    assert forces[3, 0] == pytest.approx(-c1, rel=1e-12, abs=0)


def test_zero_stiffness_skips_singular_callback_geometry():
    top, _ = typed("C=O")
    for potential in top.improper_types:
        potential.parameters["k"] *= 0
    sim, ff, _ = prepare(top, np.zeros((top.n_sites, 3)))
    sim.run(0)
    force = ff.forces_by_category["impropers"][0]
    assert force.energy == 0
    np.testing.assert_array_equal(force.forces, 0)
    np.testing.assert_array_equal(force.virials, 0)


def test_atom_mapping_permutation_preserves_inversion_forces():
    original, molecule, mapping = inputs("P")
    base, _ = assign_uff_parameters(original, molecule, atom_map=mapping)
    permutation = list(reversed(range(molecule.GetNumAtoms())))
    reordered = Chem.RenumberAtoms(molecule, permutation)
    mapped, _ = assign_uff_parameters(
        original, reordered, atom_map=dict(enumerate(permutation))
    )
    xyz = (
        base.positions.to_value("angstrom")
        + np.random.default_rng(18).normal(0, 0.1, (base.n_sites, 3))
        + 15
    )
    first, first_ff, _ = prepare(base, xyz)
    second, second_ff, _ = prepare(mapped, xyz)
    first.run(0)
    second.run(0)
    a, b = (
        ff.forces_by_category["impropers"][0] for ff in (first_ff, second_ff)
    )
    assert a.energy == pytest.approx(b.energy, abs=1e-12)
    np.testing.assert_allclose(a.forces, b.forces, atol=1e-12)


def test_actual_attachment_and_box_change_reject_tilt():
    top, _ = typed("C=O")
    xyz = top.positions.to_value("angstrom") + 15
    sim, _, _ = prepare(top, xyz)
    sim.state.set_box(hoomd.Box(30, 40, 50, xy=0.1))
    with pytest.raises(ValueError, match="orthorhombic 3D"):
        sim.run(0)
    sim, ff, _ = prepare(top, xyz)
    sim.run(0)
    sim.state.set_box(hoomd.Box(30, 40, 50, xy=0.1))
    with pytest.raises(ValueError, match="orthorhombic 3D"):
        ff.forces_by_category["impropers"][0].set_forces(1)


@pytest.mark.parametrize("sign", [-1, 1])
def test_nonaxis_perpendicular_cusp_and_smooth_case(sign):
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([-0.2, 1.0, 0.3])
    vectors = np.array([a, b, sign * np.cross(a, b)])
    with pytest.raises(ValueError, match="nondifferentiable"):
        wilson_energy_forces(vectors, [1, 1, -1, 0])
    _, forces = wilson_energy_forces(vectors, [1, 1, 0, 1])
    np.testing.assert_allclose(forces, 0, atol=1e-14)


def test_planar_callback_clears_prior_nonzero_buffers():
    top, _ = typed("C=O")
    xyz = (
        np.array(
            [[0, 0, 0], [1.2, 0, 0.3], [-0.6, 0.9, 0.1], [-0.6, -0.9, -0.2]]
        )
        + 15
    )
    sim, ff, _ = prepare(top, xyz)
    sim.run(0)
    force = ff.forces_by_category["impropers"][0]
    assert force.energy > 0 and np.linalg.norm(force.forces) > 0
    snapshot = sim.state.get_snapshot()
    snapshot.particles.position[:, 2] = 0
    sim.state.set_snapshot(snapshot)
    force.set_forces(1)
    np.testing.assert_allclose(force.forces, 0, atol=1e-14)
    np.testing.assert_allclose(force.energies, 0, atol=1e-14)
    np.testing.assert_allclose(force.virials, 0, atol=1e-14)


def test_mixed_periodic_and_uff_improper_backend_groups():
    from gmso.lib.potential_templates import PotentialTemplateLibrary

    top, _ = typed("C=O.C=O")
    potential = gmso.ImproperType.from_template(
        PotentialTemplateLibrary()["PeriodicImproperPotential"],
        {
            "k": [-0.3, 0.2] * u.kcal / u.mol,
            "n": [1, 3] * u.dimensionless,
            "phi_eq": [0.4, -0.2] * u.rad,
        },
    )
    uff_center = top.impropers[0].connection_members[0]
    for connection in top.impropers:
        if connection.connection_members[0] is not uff_center:
            connection.improper_type = potential
    xyz = (
        top.positions.to_value("angstrom")
        + np.random.default_rng(43).normal(0, 0.15, (top.n_sites, 3))
        + 15
    )
    sim, ff, frame = prepare(top, xyz)
    sim.run(0)
    assert frame.impropers.N == 3 and frame.dihedrals.N == 3
    assert len(ff.forces_by_category["impropers"]) == 3
    periodic = ff.forces_by_category["impropers"][:2]
    inversion = ff.forces_by_category["impropers"][-1]
    assert isinstance(inversion, UFFInversionForce)
    inversion_tags = set(inversion._groups.ravel())
    periodic_tags = set(frame.dihedrals.group.ravel())
    assert inversion_tags.isdisjoint(periodic_tags)
    np.testing.assert_array_equal(inversion.forces[list(periodic_tags)], 0)
    for force in periodic:
        np.testing.assert_array_equal(force.forces[list(inversion_tags)], 0)
    assert all(
        np.isfinite(f.energy) for f in ff.forces_by_category["impropers"]
    )
