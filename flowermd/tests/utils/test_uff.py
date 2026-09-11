import math

import numpy as np
import pytest

from flowermd.internal.uff import extract_uff_atoms_and_bonds

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
        result = extract_uff_atoms_and_bonds(mol)
        assert set(result) == {
            "particle_types",
            "particle_type_params",
            "bonds",
            "bond_orders",
            "bond_types",
            "bond_params",
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
        result = extract_uff_atoms_and_bonds(mol)
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
        extract_uff_atoms_and_bonds(mol)
        assert mol.ToBinary(Chem.PropertyPickleOptions.AllProps) == before

    def test_kekulized_input_orders_and_aromatic_assignment(self):
        aromatic = explicit_molecule("c1ccccc1")
        kekulized = Chem.Mol(aromatic)
        Chem.Kekulize(kekulized, clearAromaticFlags=True)
        before = kekulized.ToBinary(Chem.PropertyPickleOptions.AllProps)
        result = extract_uff_atoms_and_bonds(kekulized)
        assert set(result["bond_orders"]) == {1.0, 2.0}
        for pair, order in zip(result["bonds"], result["bond_orders"]):
            assert (
                order
                == kekulized.GetBondBetweenAtoms(*pair).GetBondTypeAsDouble()
            )
        assert bond_parameters(result) == bond_parameters(
            extract_uff_atoms_and_bonds(aromatic)
        )
        assert kekulized.ToBinary(Chem.PropertyPickleOptions.AllProps) == before

    @pytest.mark.parametrize("offset", [-0.04, 0.05, 0.2])
    def test_hydrogen_stretch_energy_and_gradient(self, offset):
        mol = explicit_molecule("[H][H]")
        mol.AddConformer(Chem.Conformer(2))
        result = extract_uff_atoms_and_bonds(mol)
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
            extract_uff_atoms_and_bonds(molecule)

    @pytest.mark.parametrize("smiles", ["CC", "[CH4]"])
    def test_missing_graph_hydrogens(self, smiles):
        with pytest.raises(
            ValueError, match="atom 0.*explicit graph hydrogens"
        ):
            extract_uff_atoms_and_bonds(Chem.MolFromSmiles(smiles))

    def test_dummy_atom(self):
        with pytest.raises(ValueError, match="dummy atom at index 0"):
            extract_uff_atoms_and_bonds(explicit_molecule("*C"))

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
            extract_uff_atoms_and_bonds(mol)

    def test_unsupported_uff_assignment(self):
        with pytest.raises(ValueError, match="incomplete UFF.*0:He"):
            extract_uff_atoms_and_bonds(Chem.MolFromSmiles("[He]"))

    def test_invalid_valence(self):
        mol = explicit_molecule("C")
        mol.GetAtomWithIdx(0).SetAtomicNum(9)
        with pytest.raises(ValueError, match="cannot sanitize molecule.*0"):
            extract_uff_atoms_and_bonds(mol)

    @pytest.mark.parametrize("value", [0, -1, math.inf, math.nan])
    @pytest.mark.parametrize("kind", ["atom", "bond"])
    def test_invalid_assigned_parameters(self, monkeypatch, value, kind):
        getter = (
            "GetUFFVdWParams" if kind == "atom" else "GetUFFBondStretchParams"
        )
        monkeypatch.setattr(uff, getter, lambda *args: (value, 1.0))
        with pytest.raises(ValueError, match=f"{kind}.*finite and positive"):
            extract_uff_atoms_and_bonds(explicit_molecule("CC"))
