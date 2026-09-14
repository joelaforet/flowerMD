import copy
import math

import gmso
import hoomd
import numpy as np
import pytest
import unyt as u
from gmso.lib.potential_templates import PotentialTemplateLibrary

from flowermd.base.forcefield import BaseHOOMDForcefield
from flowermd.internal.aa_snapshot import route_all_atom_connections
from flowermd.internal.uff_gmso import assign_uff_parameters
from flowermd.library import AllAtomDPD
from flowermd.tests.utils.test_uff_gmso import inputs


def topology():
    templates = PotentialTemplateLibrary()
    top = gmso.Topology()
    atom_types = [
        gmso.AtomType.from_template(
            templates["LennardJonesPotential"],
            {"epsilon": epsilon * u.kcal / u.mol, "sigma": 1 * u.angstrom},
            name="same",
        )
        for epsilon in (1, 0.25)
    ]
    for index in range(4):
        top.add_site(
            gmso.Atom(name=str(index), atom_type=atom_types[index % 2])
        )
    bond = gmso.BondType.from_template(
        templates["HarmonicBondPotential"],
        {"k": 10 * u.kcal / u.mol / u.angstrom**2, "r_eq": 1.1 * u.angstrom},
    )
    angle = gmso.AngleType.from_template(
        templates["HarmonicAnglePotential"],
        {"k": 20 * u.kcal / u.mol / u.rad**2, "theta_eq": 1.8 * u.rad},
    )
    torsion = gmso.DihedralType.from_template(
        templates["HOOMDPeriodicDihedralPotential"],
        {
            "k": 3 * u.kcal / u.mol,
            "n": 3 * u.dimensionless,
            "d": 1 * u.dimensionless,
            "phi0": 0 * u.rad,
        },
    )
    top.add_connection(
        gmso.Bond(connection_members=list(top.sites[:2]), bond_type=bond)
    )
    top.add_connection(
        gmso.Angle(connection_members=list(top.sites[:3]), angle_type=angle)
    )
    top.add_connection(
        gmso.Dihedral(connection_members=list(top.sites), dihedral_type=torsion)
    )
    return top


def bundle(top=None, **kwargs):
    options = dict(repulsion=40, gamma=20, kT=1, r_cut=3, bonded_scale=7)
    options.update(kwargs)
    return AllAtomDPD(topology() if top is None else top, **options)


def evaluate(top, ff, positions):
    snapshot = hoomd.Snapshot()
    snapshot.configuration.box = [30, 30, 30, 0, 0, 0]
    snapshot.particles.N = len(positions)
    snapshot.particles.types = list(ff.type_labels["sites"].values())
    snapshot.particles.typeid[:] = [
        snapshot.particles.types.index(ff.type_labels["sites"][site.atom_type])
        for site in top.sites
    ]
    snapshot.particles.position[:] = positions
    snapshot.particles.mass[:] = 1
    indices = {site: index for index, site in enumerate(top.sites)}
    physical_labels, routed = route_all_atom_connections(
        top, type_labels=ff.type_labels
    )
    for category in ("bonds", "angles", "dihedrals", "impropers"):
        connections = routed[category]
        block = getattr(snapshot, category)
        block.N = len(connections)
        if connections:
            block.types = list(physical_labels[category].values())
            block.typeid[:] = [
                block.types.index(
                    physical_labels[category][item.connection_type]
                )
                for item in connections
            ]
            block.group[:] = [
                [indices[site] for site in item.connection_members]
                for item in connections
            ]
    simulation = hoomd.Simulation(device=hoomd.device.CPU(), seed=10)
    simulation.create_state_from_snapshot(snapshot)
    simulation.operations.integrator = hoomd.md.Integrator(
        dt=0.001,
        methods=[hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())],
        forces=ff.hoomd_forces,
    )
    simulation.run(0)
    return simulation


class TestAllAtomDPD:
    def test_native_parameters_labels_references_and_preservation(self):
        top = topology()
        before = [
            (item, copy.deepcopy(item.parameters))
            for item in [
                top.sites[0].atom_type,
                top.bonds[0].bond_type,
                top.angles[0].angle_type,
                top.dihedrals[0].dihedral_type,
            ]
        ]
        ff = bundle(top)
        assert isinstance(ff, BaseHOOMDForcefield)
        bond, angle, torsion, pair = ff.hoomd_forces
        assert dict(bond.params["aa_bond_0"]) == pytest.approx(
            {"k": 70, "r0": 1.1}, rel=1e-14
        )
        assert dict(angle.params["aa_angle_0"]) == pytest.approx(
            {"k": 140, "t0": 1.8}, rel=1e-14
        )
        assert dict(torsion.params["aa_dihedral_0"]) == pytest.approx(
            {
                "k": 21,
                "n": 3,
                "d": 1,
                "phi0": 0,
            },
            rel=1e-14,
        )
        assert pair.params["aa_atom_0", "aa_atom_1"] == {"A": 20, "gamma": 10}
        assert pair.params["aa_atom_1", "aa_atom_1"] == {"A": 10, "gamma": 5}
        assert pair.kT(0) == 1
        assert pair.r_cut["aa_atom_0", "aa_atom_1"] == 3
        assert pair.nlist.buffer == 0.4
        assert set(pair.nlist.exclusions) == {"bond", "angle", "dihedral"}
        assert ff.reference_values == {
            "length": 1 * u.angstrom,
            "energy": 1 * u.kcal / u.mol,
            "mass": 1 * u.amu,
        }
        for potential, parameters in before:
            assert potential.parameters == parameters

    def test_labels_use_identity_not_name(self):
        top = topology()
        shared = top.bonds[0].bond_type
        distinct = shared.clone()
        distinct.parameters["k"] *= 2
        top.add_connection(
            gmso.Bond(
                connection_members=[top.sites[1], top.sites[2]],
                bond_type=distinct,
            )
        )
        top.add_connection(
            gmso.Bond(
                connection_members=[top.sites[2], top.sites[3]],
                bond_type=shared,
            )
        )
        ff = bundle(top)
        assert shared.name == distinct.name
        assert ff.type_labels["bonds"] == {
            shared: "aa_bond_0",
            distinct: "aa_bond_1",
        }
        assert len(ff.type_labels["sites"]) == 2
        assert ff.hoomd_forces[0].params["aa_bond_1"]["k"] == pytest.approx(
            140, rel=1e-14
        )

    def test_compatible_units_and_fresh_objects(self):
        original, converted = topology(), topology()
        for site in converted.sites:
            site.atom_type.parameters["epsilon"] = site.atom_type.parameters[
                "epsilon"
            ].to("kJ/mol")
        for item in converted.bonds:
            item.bond_type.parameters["k"] = item.bond_type.parameters["k"].to(
                "kJ/(mol*nm**2)"
            )
            item.bond_type.parameters["r_eq"] = item.bond_type.parameters[
                "r_eq"
            ].to("nm")
        converted.angles[0].angle_type.parameters["theta_eq"] = (
            converted.angles[0].angle_type.parameters["theta_eq"].to("degree")
        )
        first, second = bundle(original), bundle(converted)
        for a, b in zip(first.hoomd_forces, second.hoomd_forces):
            assert a is not b
            for key in a.params:
                assert dict(a.params[key]) == pytest.approx(dict(b.params[key]))
        first.reference_values["length"] *= 2
        assert second.reference_values["length"] == 1 * u.angstrom
        assert (
            str(converted.bonds[0].bond_type.parameters["r_eq"].units) == "nm"
        )

    def test_uniform_reads_no_nonbonded_fields(self):
        top = topology()
        for site in top.sites:
            site.atom_type.parameters.clear()
        ff = bundle(top, epsilon_weighting=False)
        for values in ff.hoomd_forces[-1].params.values():
            assert values == {"A": 40, "gamma": 20}
        with pytest.raises(ValueError, match="epsilon.*scalar"):
            bundle(top)
        with pytest.raises(ValueError, match="omit epsilon"):
            bundle(top, epsilon_weighting=False, epsilon_reference=1)

    def test_zero_epsilon_and_conservative_parity(self):
        top = topology()
        top.sites[1].atom_type.parameters["epsilon"] *= 0
        first, second = bundle(top), bundle(top, conservative=True)
        for key in first.hoomd_forces[-1].params:
            assert (
                second.hoomd_forces[-1].params[key]["A"]
                == first.hoomd_forces[-1].params[key]["A"]
            )
            assert (
                second.hoomd_forces[-1].r_cut[key]
                == first.hoomd_forces[-1].r_cut[key]
            )
        assert first.hoomd_forces[-1].params["aa_atom_0", "aa_atom_1"] == {
            "A": 0,
            "gamma": 0,
        }
        top.sites[0].atom_type.parameters["epsilon"] *= 0
        with pytest.raises(ValueError, match="epsilon_reference"):
            bundle(top)
        assert (
            bundle(top, epsilon_reference=1)
            .hoomd_forces[-1]
            .params["aa_atom_0", "aa_atom_0"]["A"]
            == 0
        )

    @pytest.mark.parametrize(
        "flag,missing",
        [
            ("include_bonds", hoomd.md.bond.Harmonic),
            ("include_angles", hoomd.md.angle.Harmonic),
            ("include_torsions", hoomd.md.dihedral.Periodic),
        ],
    )
    def test_ablation_and_disabled_validation(self, flag, missing):
        top = topology()
        ff = bundle(top, **{flag: False})
        assert getattr(ff, flag) is False
        assert not any(isinstance(force, missing) for force in ff.hoomd_forces)
        assert set(ff.hoomd_forces[-1].nlist.exclusions) == {
            "bond",
            "angle",
            "dihedral",
        }
        category = {
            "include_bonds": "bonds",
            "include_angles": "angles",
            "include_torsions": "dihedrals",
        }[flag]
        potential = getattr(top, category)[0].connection_type
        potential.parameters["k"] *= math.nan
        with pytest.raises(ValueError, match="finite"):
            bundle(top, **{flag: False})

    def test_untyped_and_native_improper_errors(self):
        raw, mol, mapping = inputs("C=O")
        partial, _ = assign_uff_parameters(
            raw, mol, atom_map=mapping, include_impropers=False
        )
        with pytest.raises(
            ValueError, match="impropers at sites.*require assigned"
        ):
            bundle(partial)
        off = bundle(partial, include_impropers=False)
        assert off.type_labels["impropers"] == {None: "aa_improper_0"}
        assert off.untyped_improper_count == partial.n_impropers
        assert off.disabled_term_counts["impropers"] == partial.n_impropers
        native, _ = assign_uff_parameters(raw, mol, atom_map=mapping)
        with pytest.raises(NotImplementedError, match="Wilson out-of-plane"):
            bundle(native)
        assert (
            bundle(native, include_impropers=False).untyped_improper_count == 0
        )

    @pytest.mark.parametrize("category", ["bonds", "angles", "dihedrals"])
    def test_untyped_unknown_arrays_and_incompatible_units(self, category):
        top = topology()
        connection = getattr(top, category)[0]
        potential = connection.connection_type
        connection.connection_type = None
        with pytest.raises(ValueError, match=rf"{category} at sites.*assigned"):
            bundle(top)
        connection.connection_type = potential
        saved = potential.parameters["k"]
        potential.parameters["k"] = 1 * u.nm
        with pytest.raises(
            ValueError, match=rf"{category} at sites.*incompatible units"
        ):
            bundle(top)
        potential.parameters["k"] = u.unyt_array([float(saved)], saved.units)
        with pytest.raises(ValueError, match="scalar quantity"):
            bundle(top)
        potential.parameters["k"] = saved
        potential.expression = 2 * potential.expression
        with pytest.raises(ValueError, match="unsupported"):
            bundle(top)

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
            ("include_impropers", 0),
        ],
    )
    def test_invalid_options(self, key, value):
        with pytest.raises(ValueError):
            bundle(**{key: value})

    @pytest.mark.parametrize(
        "category,key,value",
        [
            ("bonds", "k", -1 * u.kcal / u.mol / u.angstrom**2),
            ("bonds", "r_eq", 0 * u.angstrom),
            ("angles", "k", 0 * u.kcal / u.mol / u.rad**2),
            ("angles", "theta_eq", 4 * u.rad),
            ("dihedrals", "n", 1.5 * u.dimensionless),
            ("dihedrals", "n", 0 * u.dimensionless),
            ("dihedrals", "d", 0 * u.dimensionless),
            ("dihedrals", "phi0", math.inf * u.rad),
            ("dihedrals", "k", 1e308 * u.kcal / u.mol),
        ],
    )
    def test_invalid_native_scalar_domains(self, category, key, value):
        top = topology()
        getattr(top, category)[0].connection_type.parameters[key] = value
        with pytest.raises(ValueError):
            bundle(top)

    def test_mixed_disabled_improper_labels_and_reject_legacy_dictionary(self):
        raw, mol, mapping = inputs("C=O")
        top, _ = assign_uff_parameters(raw, mol, atom_map=mapping)
        potential = top.impropers[0].improper_type
        top.impropers[0].improper_type = None
        ff = bundle(top, include_impropers=False)
        assert ff.type_labels["impropers"] == {
            None: "aa_improper_0",
            potential: "aa_improper_1",
        }
        assert ff.untyped_improper_count == 1
        with pytest.raises(ValueError, match="GMSO Topology"):
            bundle({})


@pytest.mark.parametrize(
    "kind,k_sign", [("bond", 1), ("angle", 1), ("torsion", 1), ("torsion", -1)]
)
def test_cpu_bonded_energies_and_forces(kind, k_sign):
    top = topology()
    top.dihedrals[0].dihedral_type.parameters["k"] *= k_sign
    ff = bundle(top, conservative=True, repulsion=0)
    xyz = np.array([[0, 0, 0], [1.3, 0, 0], [1.8, 1.1, 0], [2.7, 1.4, 0.8]])
    simulation = evaluate(top, ff, xyz)
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
        normal1, normal2 = (
            np.cross(left, right),
            np.cross(-right, positions[3] - positions[2]),
        )
        phi = math.acos(
            np.clip(
                np.dot(normal1, normal2)
                / np.linalg.norm(normal1)
                / np.linalg.norm(normal2),
                -1,
                1,
            )
        )
        return 0.5 * 21 * k_sign * (1 + math.cos(3 * phi))

    expected = np.zeros_like(xyz)
    for index in np.ndindex(xyz.shape):
        plus, minus = xyz.copy(), xyz.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        expected[index] = -(energy(plus) - energy(minus)) / 2e-6
    assert force.energy == pytest.approx(energy(xyz), abs=1e-8)
    np.testing.assert_allclose(force.forces, expected, atol=1e-6)
    assert simulation.timestep == 0


def test_adapter_uff_to_cpu_hydrogen_bond():
    raw, mol, mapping = inputs("[H][H]")
    top, _ = assign_uff_parameters(raw, mol, atom_map=mapping)
    ff = bundle(top, conservative=True)
    p = top.bonds[0].bond_type.parameters
    r0, k = (
        float(p["r_eq"].to_value("angstrom")),
        float(p["k"].to_value("kcal/(mol*angstrom**2)")) * 7,
    )
    simulation = evaluate(top, ff, [[0, 0, 0], [r0 + 0.1, 0, 0]])
    assert ff.hoomd_forces[0].energy == pytest.approx(0.5 * k * 0.1**2)
    np.testing.assert_allclose(
        ff.hoomd_forces[0].forces,
        [[k * 0.1, 0, 0], [-k * 0.1, 0, 0]],
        atol=1e-7,
    )
    assert simulation.timestep == 0


def test_cpu_conservative_pair_energy_and_force():
    top = topology()
    for connection in tuple(top.connections):
        top.remove_connection(connection)
    # Keep four particles; only the first pair is within the cutoff.
    ff = bundle(top, conservative=True, epsilon_weighting=False)
    simulation = evaluate(
        top, ff, [[0, 0, 0], [1.2, 0, 0], [6, 0, 0], [-6, 0, 0]]
    )
    pair = ff.hoomd_forces[-1]
    assert pair.energy == pytest.approx(40 * 3 / 2 * (1 - 1.2 / 3) ** 2)
    np.testing.assert_allclose(
        pair.forces, [[-24, 0, 0], [24, 0, 0], [0, 0, 0], [0, 0, 0]], atol=1e-8
    )
    assert simulation.timestep == 0


@pytest.mark.parametrize("enabled", [True, False])
def test_cpu_exclusions_independent_of_ablation(enabled):
    top = topology()
    for pair in [(1, 2), (2, 3)]:
        top.add_connection(
            gmso.Bond(
                connection_members=[top.sites[i] for i in pair],
                bond_type=top.bonds[0].bond_type,
            )
        )
    top.add_connection(
        gmso.Angle(
            connection_members=[top.sites[i] for i in (1, 2, 3)],
            angle_type=top.angles[0].angle_type,
        )
    )
    ff = bundle(
        top,
        conservative=True,
        epsilon_weighting=False,
        include_bonds=enabled,
        include_angles=enabled,
        include_torsions=enabled,
    )
    simulation = evaluate(
        top, ff, [[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0.5]]
    )
    assert ff.hoomd_forces[-1].energy == 0
    np.testing.assert_array_equal(ff.hoomd_forces[-1].forces, np.zeros((4, 3)))
    assert simulation.timestep == 0


def test_uff_adapter_all_bonded_coefficients_reach_native_forces():
    from flowermd.internal.uff import extract_uff_parameters

    raw, molecule, mapping = inputs("CCO")
    expected = extract_uff_parameters(molecule)
    top, _ = assign_uff_parameters(raw, molecule, atom_map=mapping)
    ff = bundle(top, include_impropers=False)
    for category, types_key, params_key, force, conversions in (
        (
            "bonds",
            "bond_types",
            "bond_params",
            ff.hoomd_forces[0],
            {"k": ("k_kcal_mol_a2", 7), "r0": ("r0_a", 1)},
        ),
        (
            "angles",
            "angle_types",
            "angle_params",
            ff.hoomd_forces[1],
            {"k": ("k_kcal_mol_rad2", 7), "t0": ("theta0_rad", 1)},
        ),
        (
            "dihedrals",
            "dihedral_types",
            "dihedral_params",
            ff.hoomd_forces[2],
            {
                "k": ("k_kcal_mol", 7),
                "n": ("n", 1),
                "d": ("d", 1),
                "phi0": ("phi0_rad", 1),
            },
        ),
    ):
        connections = getattr(top, category)
        assert len(connections) == len(expected[types_key])
        for connection, type_name in zip(connections, expected[types_key]):
            native = force.params[
                ff.type_labels[category][connection.connection_type]
            ]
            assert dict(native) == pytest.approx(
                {
                    name: expected[params_key][type_name][source] * scale
                    for name, (source, scale) in conversions.items()
                },
                rel=1e-14,
            )
