import mbuild as mb
import pytest

from flowermd.internal.all_atom_parameters import compound_to_rdkit
from flowermd.library import PolyEthylene
from flowermd.tests import BaseTest

pytest.importorskip("rdkit")


def _formal_charges(compound):
    mol = compound_to_rdkit(compound)
    return [atom.GetFormalCharge() for atom in mol.GetAtoms()], mol


class TestFormalCharges(BaseTest):
    def test_carboxylate_keeps_its_charge(self):
        acetate = mb.load("CC(=O)[O-]", smiles=True)
        charges, mol = _formal_charges(acetate)
        assert sum(charges) == -1
        assert sorted(charges) == [-1] + [0] * (len(charges) - 1)
        oxygen = mol.GetAtomWithIdx(charges.index(-1))
        assert oxygen.GetSymbol() == "O"
        # no implicit hydrogen was added to the anionic oxygen
        assert all(
            atom.GetTotalNumHs() == atom.GetNumExplicitHs() == 0
            for atom in mol.GetAtoms()
            if atom.GetSymbol() == "O"
        )

    def test_sodium_ion(self):
        sodium = mb.load("[Na+]", smiles=True)
        charges, mol = _formal_charges(sodium)
        assert charges == [1]
        assert mol.GetAtomWithIdx(0).GetTotalNumHs() == 0

    def test_ammonium(self):
        ammonium = mb.load("C[NH3+]", smiles=True)
        charges, _ = _formal_charges(ammonium)
        assert sum(charges) == 1

    def test_neutral_polymer_unchanged(self):
        chain = PolyEthylene(lengths=3, num_mols=1).molecules[0]
        charges, _ = _formal_charges(chain)
        assert set(charges) == {0}

    def test_missing_hydrogen_raises(self):
        methane = mb.load("C", smiles=True)
        hydrogen = [p for p in methane.particles() if p.name == "H"][0]
        methane.remove(hydrogen)
        with pytest.raises(ValueError, match="valence"):
            compound_to_rdkit(methane)
