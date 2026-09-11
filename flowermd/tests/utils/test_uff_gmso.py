import gmso
import mbuild as mb
import numpy as np
import pytest
import sympy
import unyt as u

from flowermd.internal.uff import extract_uff_parameters
from flowermd.internal.uff_gmso import assign_uff_parameters
from flowermd.library import mbuildSystem

Chem = pytest.importorskip("rdkit.Chem")


def inputs(smiles="CC"):
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    compound = mb.load(smiles, smiles=True)
    compound.box = mb.Box(lengths=[3, 4, 5])
    topology = mbuildSystem(compound).gmso_system
    return topology, molecule, dict(enumerate(range(molecule.GetNumAtoms())))


def assign(topology, molecule, atom_map):
    return assign_uff_parameters(
        topology, molecule, atom_map=atom_map, include_impropers=False
    )


def groups(topology, kind):
    indices = {site: index for index, site in enumerate(topology.sites)}
    return tuple(
        tuple(indices[site] for site in connection.connection_members)
        for connection in getattr(topology, kind)
    )


class TestUFFGMSOAssignment:
    def test_existing_flower_conversion_copy_and_partial_coverage(self):
        original, molecule, atom_map = inputs()
        original.sites[0].group = "original_group"
        original.sites[0].charge = 0.2 * u.elementary_charge
        original.bonds[0].name = "retained_bond"
        original.angles[0].name = "retained_angle"
        original.dihedrals[0].name = "retained_torsion"
        before = molecule.ToBinary(Chem.PropertyPickleOptions.AllProps)
        connection_groups = {
            kind: groups(original, kind)
            for kind in ("bonds", "angles", "dihedrals", "impropers")
        }
        result, report = assign(original, molecule, atom_map)
        assert result is not original
        assert all(a is not b for a, b in zip(result.sites, original.sites))
        np.testing.assert_array_equal(result.positions, original.positions)
        np.testing.assert_array_equal(result.box.lengths, original.box.lengths)
        assert [site.name for site in result.sites] == [
            site.name for site in original.sites
        ]
        assert result.sites[0].group == "original_group"
        assert result.sites[0].charge == 0.2 * u.elementary_charge
        assert result.bonds[0].name == "retained_bond"
        assert result.angles[0].name == "retained_angle"
        assert result.dihedrals[0].name == "retained_torsion"
        for kind, expected in connection_groups.items():
            assert groups(result, kind) == expected
        for kind in ("sites", "bonds", "angles", "dihedrals"):
            assert result.is_fully_typed(group=kind)
        assert all(item.improper_type is None for item in result.impropers)
        assert report["retained_untyped_impropers"] == original.n_impropers > 0
        assert report["include_impropers"] is False
        assert report["charges_parameterized"] is False
        assert report["absent_charge_default"] == "GMSO atom-type zero"
        assert report["source"] == "UFF"
        assert report["angle_model"] == "frozen harmonic surrogate"
        assert report["rdkit_version"]
        assert report["assigned_counts"] == {
            "atoms": 8,
            "bonds": 7,
            "angles": 12,
            "proper_dihedrals": 9,
            "impropers": 0,
        }
        assert not original.is_typed()
        assert molecule.ToBinary(Chem.PropertyPickleOptions.AllProps) == before
        result.sites[0].position += 1 * u.nm
        assert not np.array_equal(result.positions, original.positions)

    def test_nonidentity_mapping_and_isotope_mass(self):
        topology, mol, _ = inputs("[2H]C")
        permutation = list(reversed(range(mol.GetNumAtoms())))
        reordered = Chem.RenumberAtoms(mol, permutation)
        result, _ = assign(topology, reordered, dict(enumerate(permutation)))
        for atom_index, site_index in enumerate(permutation):
            atom = reordered.GetAtomWithIdx(atom_index)
            assert (
                result.sites[site_index].element.atomic_number
                == atom.GetAtomicNum()
            )
            assert result.sites[site_index].mass.to_value(
                "amu"
            ) == pytest.approx(atom.GetMass())
            assert result.sites[site_index].atom_type.mass.to_value(
                "amu"
            ) == pytest.approx(atom.GetMass())

    @pytest.mark.parametrize("smiles", ["CC", "c1ccccc1"])
    def test_native_potential_units_and_energy_expressions(self, smiles):
        topology, mol, atom_map = inputs(smiles)
        result, _ = assign(topology, mol, atom_map)
        raw = extract_uff_parameters(mol)
        bond_orders = dict(zip(raw["bonds"], raw["bond_orders"]))
        for group, bond in zip(groups(result, "bonds"), result.bonds):
            assert bond.bond_order == bond_orders[tuple(sorted(group))]
        for atom_index, site in enumerate(result.sites):
            expected = raw["particle_type_params"][
                raw["particle_types"][atom_index]
            ]
            assert isinstance(site.atom_type, gmso.AtomType)
            assert site.atom_type.parameters["sigma"].to_value(
                "angstrom"
            ) == pytest.approx(expected["r_min_a"] / 2 ** (1 / 6))
            assert (
                site.atom_type.parameters["epsilon"].to_value("kcal/mol")
                == expected["epsilon_kcal_mol"]
            )
        for (
            group_name,
            table_name,
            variable,
            point,
            unit_map,
            expected_energy,
        ) in (
            (
                "bonds",
                "bond_params",
                "r",
                1.4,
                {"k": "kcal/(mol*angstrom**2)", "r_eq": "angstrom"},
                lambda p: 0.5 * p["k"] * (1.4 - p["r_eq"]) ** 2,
            ),
            (
                "angles",
                "angle_params",
                "theta",
                1.7,
                {"k": "kcal/(mol*rad**2)", "theta_eq": "rad"},
                lambda p: 0.5 * p["k"] * (1.7 - p["theta_eq"]) ** 2,
            ),
            (
                "dihedrals",
                "dihedral_params",
                "phi",
                0.8,
                {
                    "k": "kcal/mol",
                    "n": "dimensionless",
                    "d": "dimensionless",
                    "phi0": "rad",
                },
                lambda p: (
                    0.5
                    * p["k"]
                    * (1 + p["d"] * np.cos(p["n"] * 0.8 - p["phi0"]))
                ),
            ),
        ):
            potential = getattr(result, group_name)[0].connection_type
            assert potential.member_types is None
            values = {
                key: value.to_value(unit_map[key])
                for key, value in potential.parameters.items()
            }
            expression = potential.expression.subs(
                {sympy.Symbol(key): value for key, value in values.items()}
            )
            assert float(
                expression.subs(sympy.Symbol(variable), point)
            ) == pytest.approx(expected_energy(values))
            source = raw[table_name][potential.name]
            source_k = next(
                value
                for key, value in source.items()
                if key.startswith("k_kcal")
            )
            assert values["k"] == pytest.approx(source_k, rel=1e-14)

    def test_add_missing_groups_and_reuse_retained_reversed_metadata(self):
        topology, mol, atom_map = inputs()
        for angle in tuple(topology.angles):
            topology.remove_connection(angle)
        retained = topology.dihedrals[0]
        retained.connection_members = tuple(
            reversed(retained.connection_members)
        )
        retained.name = "reversed"
        old_group = tuple(retained.connection_members)
        result, _ = assign(topology, mol, atom_map)
        assert result.n_angles == 12
        assert result.dihedrals[0].name == "reversed"
        assert [
            site.name for site in result.dihedrals[0].connection_members
        ] == [site.name for site in old_group]
        assert result.is_fully_typed(group="angles")

    def test_linear_unassigned_torsions_removed_and_reported(self):
        topology, mol, atom_map = inputs("CC#CC")
        original_groups = groups(topology, "dihedrals")
        assert original_groups
        result, report = assign(topology, mol, atom_map)
        assert result.n_dihedrals == 0
        assert (
            report["removed_unassigned_groups"]["dihedrals"] == original_groups
        )
        assert groups(topology, "dihedrals") == original_groups

    def test_unassigned_restrained_torsion_is_rejected(self):
        topology, mol, atom_map = inputs("CC#CC")
        topology.dihedrals[0].restraint = {
            "phi_eq": 0 * u.rad,
            "k": 1 * u.kcal / u.mol,
        }
        with pytest.raises(ValueError, match="with a restraint"):
            assign(topology, mol, atom_map)
        assert not topology.is_typed()

    @pytest.mark.parametrize(
        "bad_map",
        [
            {},
            {i: 0 for i in range(8)},
            {i: float(i) for i in range(8)},
            {**dict(enumerate(range(8))), 0: True},
        ],
    )
    def test_invalid_mapping(self, bad_map):
        topology, mol, _ = inputs()
        with pytest.raises(ValueError, match="atom_map"):
            assign(topology, mol, bad_map)
        assert not topology.is_typed()

    @pytest.mark.parametrize("fault", ["element", "bond", "angle", "typed"])
    def test_invalid_topology_is_not_mutated(self, fault):
        topology, mol, atom_map = inputs()
        if fault == "element":
            topology.sites[0].element = None
        elif fault == "bond":
            topology.remove_connection(topology.bonds[0])
        elif fault == "angle":
            topology.add_connection(
                gmso.Angle(
                    connection_members=[topology.sites[i] for i in (2, 5, 7)]
                )
            )
        else:
            topology.sites[0].atom_type = gmso.AtomType()
        old_types = [site.atom_type for site in topology.sites]
        old_counts = (
            topology.n_sites,
            topology.n_bonds,
            topology.n_angles,
            topology.n_dihedrals,
        )
        with pytest.raises(ValueError):
            assign(topology, mol, atom_map)
        assert [site.atom_type for site in topology.sites] == old_types
        assert old_counts == (
            topology.n_sites,
            topology.n_bonds,
            topology.n_angles,
            topology.n_dihedrals,
        )

    def test_improper_assignment_defaults_on_and_accepts_explicit_off(self):
        topology, mol, atom_map = inputs()
        result, report = assign_uff_parameters(topology, mol, atom_map=atom_map)
        assert report["include_impropers"] is True
        assert result.n_impropers == 0
        assert report["retained_untyped_impropers"] == 0
        with pytest.raises(ValueError, match="must be a bool"):
            assign_uff_parameters(
                topology, mol, atom_map=atom_map, include_impropers=0
            )
