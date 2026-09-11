import copy
import math

import hoomd
import numpy as np
import pytest
import unyt as u

from flowermd.base.forcefield import BaseHOOMDForcefield
from flowermd.library import AllAtomDPD


def parameters():
    return {
        "particle_types": ("b", "a", "b", "a"),
        "particle_type_params": {
            "a": {"epsilon_kcal_mol": 0.25},
            "b": {"epsilon_kcal_mol": 1.0},
        },
        "bonds": ((0, 1),),
        "bond_types": ("bond",),
        "bond_params": {"bond": {"k_kcal_mol_a2": 10.0, "r0_a": 1.1}},
        "angles": ((0, 1, 2),),
        "angle_types": ("angle",),
        "angle_params": {"angle": {"k_kcal_mol_rad2": 20.0, "theta0_rad": 1.8}},
        "dihedrals": ((0, 1, 2, 3),),
        "dihedral_types": ("torsion",),
        "dihedral_params": {
            "torsion": {"k_kcal_mol": 3.0, "n": 3, "d": 1, "phi0_rad": 0.0}
        },
    }


def bundle(data=None, **kwargs):
    options = dict(repulsion=40, gamma=20, kT=1, r_cut=3, bonded_scale=7)
    options.update(kwargs)
    return AllAtomDPD(parameters() if data is None else data, **options)


def empty_bonded(data):
    for groups, types, table in [
        ("bonds", "bond_types", "bond_params"),
        ("angles", "angle_types", "angle_params"),
        ("dihedrals", "dihedral_types", "dihedral_params"),
    ]:
        data[groups], data[types], data[table] = (), (), {}
    return data


def evaluate(data, forces, positions):
    snapshot = hoomd.Snapshot()
    snapshot.configuration.box = [30, 30, 30, 0, 0, 0]
    snapshot.particles.N = len(positions)
    names = list(dict.fromkeys(data["particle_types"]))
    snapshot.particles.types = names
    snapshot.particles.typeid[:] = [
        names.index(name) for name in data["particle_types"]
    ]
    snapshot.particles.position[:] = positions
    snapshot.particles.mass[:] = 1
    for groups, types in [
        ("bonds", "bond_types"),
        ("angles", "angle_types"),
        ("dihedrals", "dihedral_types"),
    ]:
        block = getattr(snapshot, groups)
        block.N = len(data[groups])
        if block.N:
            block.types = list(dict.fromkeys(data[types]))
            block.typeid[:] = [block.types.index(name) for name in data[types]]
            block.group[:] = data[groups]
    simulation = hoomd.Simulation(device=hoomd.device.CPU(), seed=10)
    simulation.create_state_from_snapshot(snapshot)
    simulation.operations.integrator = hoomd.md.Integrator(
        dt=0.001,
        methods=[hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())],
        forces=forces,
    )
    simulation.run(0)
    return simulation


class TestAllAtomDPD:
    def test_parameters_units_and_single_scaling(self):
        data = parameters()
        original = copy.deepcopy(data)
        ff = bundle(data)
        assert isinstance(ff, BaseHOOMDForcefield)
        bond, angle, torsion, pair = ff.hoomd_forces
        assert isinstance(bond, hoomd.md.bond.Harmonic)
        assert isinstance(angle, hoomd.md.angle.Harmonic)
        assert isinstance(torsion, hoomd.md.dihedral.Periodic)
        assert isinstance(pair, hoomd.md.pair.DPD)
        assert bond.params["bond"] == {"k": 70, "r0": 1.1}
        assert angle.params["angle"] == {"k": 140, "t0": 1.8}
        assert torsion.params["torsion"] == {"k": 21, "n": 3, "d": 1, "phi0": 0}
        assert pair.params["a", "b"] == {"A": 20, "gamma": 10}
        assert pair.params["a", "a"] == {"A": 10, "gamma": 5}
        assert pair.params["b", "b"] == {"A": 40, "gamma": 20}
        assert pair.kT(0) == 1
        assert pair.r_cut["a", "b"] == 3
        assert pair.nlist.buffer == 0.4
        assert set(pair.nlist.exclusions) == {"bond", "angle", "dihedral"}
        assert ff.reference_values == {
            "length": 1 * u.angstrom,
            "energy": 1 * u.kcal / u.mol,
            "mass": 1 * u.amu,
        }
        assert data == original

    def test_fresh_forces_and_references(self):
        first, second = bundle(), bundle()
        assert all(
            a is not b for a, b in zip(first.hoomd_forces, second.hoomd_forces)
        )
        assert first.hoomd_forces[-1].nlist is not second.hoomd_forces[-1].nlist
        assert first.reference_values is not second.reference_values
        first.reference_values["length"] *= 2
        assert second.reference_values["length"] == 1 * u.angstrom

    def test_unweighted_does_not_read_epsilon_fields(self):
        class NoEpsilon(dict):
            def __contains__(self, key):
                raise AssertionError("epsilon field accessed")

            def __getitem__(self, key):
                raise AssertionError("epsilon field accessed")

        data = parameters()
        data["particle_type_params"] = {"a": NoEpsilon(), "b": NoEpsilon()}
        ff = bundle(data, epsilon_weighting=False)
        for values in ff.hoomd_forces[-1].params.values():
            assert values == {"A": 40, "gamma": 20}
        with pytest.raises(ValueError, match="omit epsilon"):
            bundle(data, epsilon_weighting=False, epsilon_reference=1)

    def test_zero_epsilon_and_conservative_parity(self):
        data = parameters()
        data["particle_type_params"]["a"]["epsilon_kcal_mol"] = 0
        dpd = bundle(data).hoomd_forces[-1]
        conservative = bundle(data, conservative=True).hoomd_forces[-1]
        assert isinstance(conservative, hoomd.md.pair.DPDConservative)
        for pair in [("a", "a"), ("a", "b"), ("b", "b")]:
            assert conservative.params[pair] == {"A": dpd.params[pair]["A"]}
            assert conservative.r_cut[pair] == dpd.r_cut[pair]
        assert dpd.params["a", "b"] == {"A": 0, "gamma": 0}
        data["particle_type_params"]["b"]["epsilon_kcal_mol"] = 0
        with pytest.raises(ValueError, match="epsilon_reference"):
            bundle(data)
        assert bundle(data, epsilon_reference=1).hoomd_forces[-1].params[
            "a", "b"
        ] == {"A": 0, "gamma": 0}

    def test_empty_bonded_terms(self):
        assert len(bundle(empty_bonded(parameters())).hoomd_forces) == 1

    @pytest.mark.parametrize(
        "flag,missing",
        [
            ("include_bonds", hoomd.md.bond.Harmonic),
            ("include_angles", hoomd.md.angle.Harmonic),
            ("include_torsions", hoomd.md.dihedral.Periodic),
        ],
    )
    def test_term_ablation_preserves_other_forces_and_pair_graph(
        self, flag, missing
    ):
        data = parameters()
        before = copy.deepcopy(data)
        reference = bundle(data)
        ablated = bundle(data, **{flag: False})
        assert getattr(ablated, flag) is False
        assert not any(
            isinstance(force, missing) for force in ablated.hoomd_forces
        )
        for force in ablated.hoomd_forces:
            other = next(
                item
                for item in reference.hoomd_forces
                if type(item) is type(force)
            )
            assert dict(force.params) == dict(other.params)
        assert set(ablated.hoomd_forces[-1].nlist.exclusions) == {
            "bond",
            "angle",
            "dihedral",
        }
        assert data == before
        with pytest.raises(ValueError, match="must be a bool"):
            bundle(**{flag: 0})

    def test_all_bonded_terms_disabled(self):
        ff = bundle(
            include_bonds=False, include_angles=False, include_torsions=False
        )
        assert len(ff.hoomd_forces) == 1
        assert isinstance(ff.hoomd_forces[0], hoomd.md.pair.DPD)

    @pytest.mark.parametrize(
        "flag,table,name,key",
        [
            ("include_bonds", "bond_params", "bond", "k_kcal_mol_a2"),
            ("include_angles", "angle_params", "angle", "k_kcal_mol_rad2"),
            ("include_torsions", "dihedral_params", "torsion", "k_kcal_mol"),
        ],
    )
    def test_disabled_parameters_are_still_validated(
        self, flag, table, name, key
    ):
        data = parameters()
        data[table][name][key] = -1
        with pytest.raises(ValueError):
            bundle(data, **{flag: False})

    @pytest.mark.parametrize(
        "key,value",
        [
            ("kT", 0),
            ("bonded_scale", 0),
            ("r_cut", 0),
            ("repulsion", -1),
            ("gamma", -1),
            ("kT", math.inf),
            ("bonded_scale", math.nan),
            ("r_cut", True),
            ("epsilon_weighting", 1),
            ("conservative", 1),
        ],
    )
    def test_invalid_options(self, key, value):
        with pytest.raises(ValueError):
            bundle(**{key: value})

    @pytest.mark.parametrize(
        "table,name,key,value",
        [
            ("bond_params", "bond", "k_kcal_mol_a2", 0),
            ("angle_params", "angle", "k_kcal_mol_rad2", -1),
            ("angle_params", "angle", "theta0_rad", 4),
            ("dihedral_params", "torsion", "k_kcal_mol", -1),
            ("dihedral_params", "torsion", "n", True),
            ("dihedral_params", "torsion", "n", 1.5),
            ("dihedral_params", "torsion", "d", True),
            ("dihedral_params", "torsion", "d", 0),
            ("dihedral_params", "torsion", "phi0_rad", math.inf),
            ("bond_params", "bond", "k_kcal_mol_a2", 1e308),
        ],
    )
    def test_invalid_bonded_values(self, table, name, key, value):
        data = parameters()
        data[table][name][key] = value
        with pytest.raises(ValueError):
            bundle(data)

    @pytest.mark.parametrize(
        "table,name",
        [
            ("particle_type_params", "a"),
            ("bond_params", "bond"),
            ("angle_params", "angle"),
            ("dihedral_params", "torsion"),
        ],
    )
    def test_missing_type_coverage(self, table, name):
        data = parameters()
        del data[table][name]
        with pytest.raises(ValueError, match="missing parameters"):
            bundle(data)

    def test_mismatched_type_count(self):
        data = parameters()
        data["bond_types"] = ()
        with pytest.raises(ValueError, match="matching counts"):
            bundle(data)


@pytest.mark.parametrize("kind", ["bond", "angle", "torsion"])
def test_cpu_bonded_energies_and_forces(kind):
    data = parameters()
    ff = bundle(data, conservative=True, repulsion=0)
    xyz = np.array([[0, 0, 0], [1.3, 0, 0], [1.8, 1.1, 0], [2.7, 1.4, 0.8]])
    simulation = evaluate(data, ff.hoomd_forces, xyz)
    force = ff.hoomd_forces[{"bond": 0, "angle": 1, "torsion": 2}[kind]]

    def energy(positions):
        if kind == "bond":
            return (
                0.5
                * 70
                * (np.linalg.norm(positions[0] - positions[1]) - 1.1) ** 2
            )
        left, right = positions[0] - positions[1], positions[2] - positions[1]
        if kind == "angle":
            angle = math.acos(
                np.dot(left, right)
                / np.linalg.norm(left)
                / np.linalg.norm(right)
            )
            return 0.5 * 140 * (angle - 1.8) ** 2
        normal1 = np.cross(left, right)
        normal2 = np.cross(-right, positions[3] - positions[2])
        phi = math.acos(
            np.clip(
                np.dot(normal1, normal2)
                / np.linalg.norm(normal1)
                / np.linalg.norm(normal2),
                -1,
                1,
            )
        )
        return 0.5 * 21 * (1 + math.cos(3 * phi))

    expected_forces = np.zeros_like(xyz)
    step = 1e-6
    for index in np.ndindex(xyz.shape):
        plus, minus = xyz.copy(), xyz.copy()
        plus[index] += step
        minus[index] -= step
        expected_forces[index] = -(energy(plus) - energy(minus)) / (2 * step)
    assert force.energy == pytest.approx(energy(xyz), abs=1e-8)
    np.testing.assert_allclose(force.forces, expected_forces, atol=1e-6)
    assert simulation.timestep == 0


def test_cpu_conservative_pair_energy_and_force():
    data = empty_bonded(parameters())
    data["particle_types"] = ("a", "a")
    ff = bundle(data, conservative=True, epsilon_weighting=False)
    simulation = evaluate(data, ff.hoomd_forces, [[0, 0, 0], [1.2, 0, 0]])
    pair = ff.hoomd_forces[-1]
    expected = 40 * 3 / 2 * (1 - 1.2 / 3) ** 2
    assert pair.energy == pytest.approx(expected)
    np.testing.assert_allclose(
        pair.forces, [[-24, 0, 0], [24, 0, 0]], atol=1e-8
    )
    assert simulation.timestep == 0


@pytest.mark.parametrize("include_bonded", [True, False])
def test_cpu_bond_angle_dihedral_pair_exclusions(include_bonded):
    data = parameters()
    data["bonds"] = ((0, 1), (1, 2), (2, 3))
    data["bond_types"] = ("bond",) * 3
    data["angles"] = ((0, 1, 2), (1, 2, 3))
    data["angle_types"] = ("angle",) * 2
    ff = bundle(
        data,
        conservative=True,
        epsilon_weighting=False,
        include_bonds=include_bonded,
        include_angles=include_bonded,
        include_torsions=include_bonded,
    )
    simulation = evaluate(
        data, ff.hoomd_forces, [[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0.5]]
    )
    assert ff.hoomd_forces[-1].energy == 0
    np.testing.assert_array_equal(ff.hoomd_forces[-1].forces, np.zeros((4, 3)))
    assert simulation.timestep == 0
