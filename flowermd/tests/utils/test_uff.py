import math

import numpy as np
import pytest

from flowermd.internal.uff import extract_uff_parameters

Chem = pytest.importorskip("rdkit.Chem")
uff = pytest.importorskip("rdkit.Chem.rdForceFieldHelpers")


def explicit_molecule(smiles):
    return Chem.AddHs(Chem.MolFromSmiles(smiles))


def bond_parameters(result):
    return {
        pair: result["bond_params"][name]
        for pair, name in zip(result["bonds"], result["bond_types"])
    }


class TestUFFAtomsAndBonds:
    @pytest.mark.parametrize("smiles", ["CC", "CCO", "CC=O", "C#N", "c1ccccc1"])
    def test_parameters_and_bond_orders(self, smiles):
        mol = explicit_molecule(smiles)
        result = extract_uff_parameters(mol)
        assert set(result) == {
            "particle_types",
            "particle_type_params",
            "bonds",
            "bond_orders",
            "bond_types",
            "bond_params",
            "angles",
            "angle_types",
            "angle_params",
            "dihedrals",
            "dihedral_types",
            "dihedral_params",
        }
        assert isinstance(result["particle_types"], tuple)
        assert len(result["particle_types"]) == mol.GetNumAtoms()
        keys = []
        for atom, name in zip(mol.GetAtoms(), result["particle_types"]):
            distance, epsilon = uff.GetUFFVdWParams(
                mol, atom.GetIdx(), atom.GetIdx()
            )
            params = result["particle_type_params"][name]
            assert params == {
                "r_min_a": distance,
                "epsilon_kcal_mol": epsilon,
                "mass_amu": atom.GetMass(),
            }
            key = (distance, epsilon, atom.GetMass())
            if key not in keys:
                keys.append(key)
            assert name == f"uff_vdw_{keys.index(key)}"
        expected_orders = {
            tuple(
                sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
            ): b.GetBondTypeAsDouble()
            for b in mol.GetBonds()
        }
        assert result["bonds"] == tuple(sorted(expected_orders))
        assert result["bond_orders"] == tuple(
            expected_orders[p] for p in result["bonds"]
        )
        bond_keys = []
        for pair, name in zip(result["bonds"], result["bond_types"]):
            key = uff.GetUFFBondStretchParams(mol, *pair)
            if key not in bond_keys:
                bond_keys.append(key)
            assert name == f"uff_bond_{bond_keys.index(key)}"
            assert result["bond_params"][name] == dict(
                zip(("k_kcal_mol_a2", "r0_a"), key)
            )

    def test_isotope_mass_and_disconnected_order(self):
        mol = explicit_molecule("[2H]C.CCO")
        mol = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
        result = extract_uff_parameters(mol)
        masses = [
            result["particle_type_params"][name]["mass_amu"]
            for name in result["particle_types"]
        ]
        assert masses == [atom.GetMass() for atom in mol.GetAtoms()]
        hydrogens = [
            (atom.GetIsotope(), name)
            for atom, name in zip(mol.GetAtoms(), result["particle_types"])
            if atom.GetAtomicNum() == 1
        ]
        assert {name for isotope, name in hydrogens if isotope == 2}.isdisjoint(
            {name for isotope, name in hydrogens if isotope == 0}
        )
        assert len(result["bonds"]) == mol.GetNumBonds()

    def test_input_graph_properties_conformers_and_stereo_unchanged(self):
        mol = explicit_molecule("C[C@H](O)F")
        mol.SetProp("sentinel", "unchanged")
        mol.GetAtomWithIdx(0).SetProp("atom_sentinel", "unchanged")
        mol.GetBondWithIdx(0).SetProp("bond_sentinel", "unchanged")
        mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()))
        before = mol.ToBinary(Chem.PropertyPickleOptions.AllProps)
        extract_uff_parameters(mol)
        assert mol.ToBinary(Chem.PropertyPickleOptions.AllProps) == before

    def test_kekulized_input_orders_and_aromatic_assignment(self):
        aromatic = explicit_molecule("c1ccccc1")
        kekulized = Chem.Mol(aromatic)
        Chem.Kekulize(kekulized, clearAromaticFlags=True)
        before = kekulized.ToBinary(Chem.PropertyPickleOptions.AllProps)
        result = extract_uff_parameters(kekulized)
        assert set(result["bond_orders"]) == {1.0, 2.0}
        for pair, order in zip(result["bonds"], result["bond_orders"]):
            assert (
                order
                == kekulized.GetBondBetweenAtoms(*pair).GetBondTypeAsDouble()
            )
        assert bond_parameters(result) == bond_parameters(
            extract_uff_parameters(aromatic)
        )
        assert kekulized.ToBinary(Chem.PropertyPickleOptions.AllProps) == before

    @pytest.mark.parametrize("offset", [-0.04, 0.05, 0.2])
    def test_hydrogen_stretch_energy_and_gradient(self, offset):
        mol = explicit_molecule("[H][H]")
        mol.AddConformer(Chem.Conformer(2))
        result = extract_uff_parameters(mol)
        params = next(iter(result["bond_params"].values()))
        distance = params["r0_a"] + offset
        positions = [0.0, 0.0, 0.0, distance, 0.0, 0.0]
        forcefield = uff.UFFGetMoleculeForceField(mol)
        assert forcefield.CalcEnergy(positions) == pytest.approx(
            0.5 * params["k_kcal_mol_a2"] * offset**2
        )
        derivative = params["k_kcal_mol_a2"] * offset
        np.testing.assert_allclose(
            forcefield.CalcGrad(positions),
            [-derivative, 0, 0, derivative, 0, 0],
            atol=1e-10,
        )

    @pytest.mark.parametrize("molecule", [None, "CC", Chem.Mol()])
    def test_invalid_molecule(self, molecule):
        with pytest.raises(ValueError, match="nonempty RDKit Mol"):
            extract_uff_parameters(molecule)

    @pytest.mark.parametrize("smiles", ["CC", "[CH4]"])
    def test_missing_graph_hydrogens(self, smiles):
        with pytest.raises(
            ValueError, match="atom 0.*explicit graph hydrogens"
        ):
            extract_uff_parameters(Chem.MolFromSmiles(smiles))

    def test_dummy_atom(self):
        with pytest.raises(ValueError, match="dummy atom at index 0"):
            extract_uff_parameters(explicit_molecule("*C"))

    @pytest.mark.parametrize(
        "order",
        [
            Chem.BondType.UNSPECIFIED,
            Chem.BondType.DATIVE,
            Chem.BondType.QUADRUPLE,
        ],
    )
    def test_unsupported_bond_orders(self, order):
        mol = explicit_molecule("CC")
        mol.GetBondWithIdx(0).SetBondType(order)
        with pytest.raises(ValueError, match="unsupported bond order at atoms"):
            extract_uff_parameters(mol)

    def test_unsupported_uff_assignment(self):
        with pytest.raises(ValueError, match="incomplete UFF.*0:He"):
            extract_uff_parameters(Chem.MolFromSmiles("[He]"))

    def test_invalid_valence(self):
        mol = explicit_molecule("C")
        mol.GetAtomWithIdx(0).SetAtomicNum(9)
        with pytest.raises(ValueError, match="cannot sanitize molecule.*0"):
            extract_uff_parameters(mol)

    @pytest.mark.parametrize("value", [0, -1, math.inf, math.nan])
    @pytest.mark.parametrize("kind", ["atom", "bond"])
    def test_invalid_assigned_parameters(self, monkeypatch, value, kind):
        getter = (
            "GetUFFVdWParams" if kind == "atom" else "GetUFFBondStretchParams"
        )
        monkeypatch.setattr(uff, getter, lambda *args: (value, 1.0))
        with pytest.raises(ValueError, match=f"{kind}.*finite and positive"):
            extract_uff_parameters(explicit_molecule("CC"))


class TestUFFHarmonicAngles:
    @pytest.mark.parametrize("smiles,expected_count", [("O", 1), ("CC", 12)])
    def test_enumeration_and_exact_types(self, smiles, expected_count):
        mol = explicit_molecule(smiles)
        result = extract_uff_parameters(mol)
        expected = []
        keys = []
        for atom in mol.GetAtoms():
            neighbors = sorted(n.GetIdx() for n in atom.GetNeighbors())
            for index, first in enumerate(neighbors):
                for third in neighbors[index + 1 :]:
                    expected.append((first, atom.GetIdx(), third))
        assert result["angles"] == tuple(expected)
        assert len(expected) == expected_count
        for group, name in zip(result["angles"], result["angle_types"]):
            ka, degrees = uff.GetUFFAngleBendParams(mol, *group)
            key = (ka, math.radians(degrees), 0)
            if key not in keys:
                keys.append(key)
            assert name == f"uff_angle_{keys.index(key)}"
            assert result["angle_params"][name] == dict(
                zip(("k_kcal_mol_rad2", "theta0_rad", "uff_order"), key)
            )

    @pytest.mark.parametrize(
        "smiles,order,target", [("C=C", 3, 120), ("O=C=O", 1, 180)]
    )
    def test_ordinary_sp2_and_linear_sp(self, smiles, order, target):
        result = extract_uff_parameters(explicit_molecule(smiles))
        for parameters in result["angle_params"].values():
            assert parameters["uff_order"] == order
            assert parameters["theta0_rad"] == pytest.approx(
                math.radians(target)
            )

    @pytest.mark.parametrize(
        "smiles,inside,outside", [("C1=CC1", 60, 150), ("C1=CCC1", 90, 135)]
    )
    def test_frozen_small_ring_targets_keep_getter_stiffness(
        self, smiles, inside, outside
    ):
        mol = explicit_molecule(smiles)
        result = extract_uff_parameters(mol)
        targets = set()
        for group, name in zip(result["angles"], result["angle_types"]):
            if (
                mol.GetAtomWithIdx(group[1]).GetHybridization()
                != Chem.HybridizationType.SP2
            ):
                continue
            parameters = result["angle_params"][name]
            both_ring = all(
                mol.GetAtomWithIdx(i).IsInRing() for i in (group[0], group[2])
            )
            target = inside if both_ring else outside
            targets.add(target)
            assert parameters["theta0_rad"] == math.radians(target)
            assert parameters["uff_order"] == 0
            assert (
                parameters["k_kcal_mol_rad2"]
                == uff.GetUFFAngleBendParams(mol, *group)[0]
            )
        assert targets == {inside, outside}

    def test_renumbered_disconnected_angles(self):
        mol = explicit_molecule("O.C=C")
        original = extract_uff_parameters(mol)
        permutation = list(reversed(range(mol.GetNumAtoms())))
        reordered = extract_uff_parameters(Chem.RenumberAtoms(mol, permutation))
        expected = {
            (min(first, third), center, max(first, third)): original[
                "angle_params"
            ][name]
            for (first, center, third), name in zip(
                original["angles"], original["angle_types"]
            )
        }
        for group, name in zip(reordered["angles"], reordered["angle_types"]):
            first, center, third = (permutation[i] for i in group)
            assert (
                reordered["angle_params"][name]
                == expected[min(first, third), center, max(first, third)]
            )
        assert len(reordered["angles"]) == len(expected)

    def test_no_angles(self):
        result = extract_uff_parameters(explicit_molecule("[H][H]"))
        assert result["angles"] == result["angle_types"] == ()
        assert result["angle_params"] == {}

    @pytest.mark.parametrize("smiles", ["O", "O=C=O"])
    def test_independent_uff_curvature_at_fixed_bond_lengths(self, smiles):
        mol = explicit_molecule(smiles)
        mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()))
        result = extract_uff_parameters(mol)
        assert len(result["angles"]) == 1
        first, center, third = result["angles"][0]
        params = result["angle_params"][result["angle_types"][0]]
        bond_params = bond_parameters(result)
        first_length = bond_params[tuple(sorted((first, center)))]["r0_a"]
        third_length = bond_params[tuple(sorted((third, center)))]["r0_a"]
        forcefield = uff.UFFGetMoleculeForceField(mol)

        def energy(theta):
            positions = np.zeros((3, 3))
            positions[first] = [first_length, 0, 0]
            positions[third] = [
                third_length * math.cos(theta),
                third_length * math.sin(theta),
                0,
            ]
            return forcefield.CalcEnergy(positions.ravel().tolist())

        target = params["theta0_rad"]
        minimum = energy(target)
        curvatures = []
        for step in (0.02, 0.005):
            if target == math.pi:
                curvature = 2 * (energy(target - step) - minimum) / step**2
            else:
                curvature = (
                    energy(target + step) - 2 * minimum + energy(target - step)
                ) / step**2
            curvatures.append(curvature)
        ka = params["k_kcal_mol_rad2"]
        assert abs(curvatures[1] - ka) < abs(curvatures[0] - ka)
        assert curvatures[1] == pytest.approx(ka, rel=1e-4)

    @pytest.mark.parametrize(
        "smiles,hybridization",
        [("FS(F)(F)(F)(F)F", "SP3D2"), ("FP(F)(F)(F)F", "SP3D")],
    )
    def test_unsupported_geometry_specific_centers(self, smiles, hybridization):
        with pytest.raises(
            ValueError,
            match=f"angle center 1 with degree .*{hybridization}.*geometry-specific",
        ):
            extract_uff_parameters(explicit_molecule(smiles))

    def test_missing_angle_assignment(self, monkeypatch):
        monkeypatch.setattr(uff, "GetUFFAngleBendParams", lambda *args: None)
        with pytest.raises(
            ValueError, match=r"angle parameters to atoms \(1, 0, 2\)"
        ):
            extract_uff_parameters(explicit_molecule("O"))

    @pytest.mark.parametrize(
        "values",
        [
            (0, 100),
            (-1, 100),
            (math.inf, 100),
            (math.nan, 100),
            (1, 0),
            (1, -1),
            (1, math.inf),
            (1, math.nan),
            (1, 181),
        ],
    )
    def test_invalid_angle_parameters(self, monkeypatch, values):
        monkeypatch.setattr(uff, "GetUFFAngleBendParams", lambda *args: values)
        with pytest.raises(ValueError, match=r"angle \(1, 0, 2\)"):
            extract_uff_parameters(explicit_molecule("O"))
