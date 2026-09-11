import math
from collections import Counter

import numpy as np
import pytest

from flowermd.internal.uff import extract_uff_parameters

Chem = pytest.importorskip("rdkit.Chem")
uff = pytest.importorskip("rdkit.Chem.rdForceFieldHelpers")


def molecule(smiles):
    return Chem.AddHs(Chem.MolFromSmiles(smiles))


def indexed_parameters(result):
    return {
        group: result["dihedral_params"][name]
        for group, name in zip(result["dihedrals"], result["dihedral_types"])
    }


class TestUFFProperTorsions:
    @pytest.mark.parametrize(
        "smiles,count,n,d",
        [("CC", 9, 3, 1), ("OO", 1, 2, 1), ("C=C", 4, 2, -1)],
    )
    def test_groups_divisor_and_form(self, smiles, count, n, d):
        mol = molecule(smiles)
        result = extract_uff_parameters(mol)
        assert len(result["dihedrals"]) == count
        assert list(result["dihedrals"]) == sorted(
            result["dihedrals"], key=lambda g: (g[1], g[2], g[0], g[3])
        )
        for group, params in indexed_parameters(result).items():
            assert len(set(group)) == 4
            assert group[1] < group[2]
            assert params == {
                "k_kcal_mol": uff.GetUFFTorsionParams(mol, *group) / count,
                "n": n,
                "d": d,
                "phi0_rad": 0.0,
            }
        assert len(result["dihedral_params"]) == 1

    def test_mixed_outer_sp2_exception(self):
        result = extract_uff_parameters(molecule("CC=C"))
        forms = {
            (group[3], params["n"], params["d"])
            for group, params in indexed_parameters(result).items()
            if group[1:3] == (0, 1)
        }
        assert (2, 3, 1) in forms
        assert any(n == 6 and d == -1 for _, n, d in forms)

    def test_mixed_chalcogen_precedence(self):
        mol = molecule("C=CS")
        assert (
            mol.GetAtomWithIdx(2).GetHybridization()
            == Chem.HybridizationType.SP3
        )
        assert (
            mol.GetAtomWithIdx(1).GetHybridization()
            == Chem.HybridizationType.SP2
        )
        pairs = indexed_parameters(extract_uff_parameters(mol))
        mixed = [
            params for group, params in pairs.items() if group[1:3] == (1, 2)
        ]
        assert mixed
        assert all((params["n"], params["d"]) == (2, 1) for params in mixed)

    def test_aromatic_kekule_parity(self):
        aromatic = molecule("c1ccccc1")
        kekule = Chem.Mol(aromatic)
        Chem.Kekulize(kekule, clearAromaticFlags=True)
        before = kekule.ToBinary(Chem.PropertyPickleOptions.AllProps)
        assert indexed_parameters(
            extract_uff_parameters(aromatic)
        ) == indexed_parameters(extract_uff_parameters(kekule))
        assert before == kekule.ToBinary(Chem.PropertyPickleOptions.AllProps)

    def test_three_ring_has_distinct_endpoints(self):
        mol = molecule("C1CC1")
        result = extract_uff_parameters(mol)
        assert len(result["dihedrals"]) == 24
        counts = Counter(group[1:3] for group in result["dihedrals"])
        assert set(counts.values()) == {8}
        for group, params in indexed_parameters(result).items():
            assert len(set(group)) == 4
            assert (
                params["k_kcal_mol"] == uff.GetUFFTorsionParams(mol, *group) / 8
            )

    @pytest.mark.parametrize("smiles", ["C#C", "CC#CC", "[H][H]"])
    def test_no_linear_or_short_graph_torsions(self, smiles):
        result = extract_uff_parameters(molecule(smiles))
        assert result["dihedrals"] == result["dihedral_types"] == ()
        assert result["dihedral_params"] == {}

    def test_renumbered_disconnected_and_exact_types(self):
        mol = molecule("CC.OO.C=C")
        original = indexed_parameters(extract_uff_parameters(mol))
        permutation = list(reversed(range(mol.GetNumAtoms())))
        renamed = extract_uff_parameters(Chem.RenumberAtoms(mol, permutation))
        keys = []
        for group, name in zip(renamed["dihedrals"], renamed["dihedral_types"]):
            mapped = tuple(permutation[i] for i in group)
            if mapped[1] > mapped[2]:
                mapped = mapped[::-1]
            params = renamed["dihedral_params"][name]
            assert params == original[mapped]
            key = (params["k_kcal_mol"], params["n"], params["d"])
            if key not in keys:
                keys.append(key)
            assert name == f"uff_torsion_{keys.index(key)}"
        assert len(renamed["dihedrals"]) == len(original)

    @pytest.mark.parametrize("barrier", [None, -1, math.inf, math.nan])
    def test_invalid_assignment(self, monkeypatch, barrier):
        monkeypatch.setattr(uff, "GetUFFTorsionParams", lambda *args: barrier)
        with pytest.raises(ValueError, match=r"torsion.*\(2, 0, 1, 5\)"):
            extract_uff_parameters(molecule("CC"))

    def test_zero_terms_remain_in_divisor(self, monkeypatch):
        def getter(mol, first, second, third, fourth):
            return 0.0 if first == 2 else 9.0

        monkeypatch.setattr(uff, "GetUFFTorsionParams", getter)
        result = extract_uff_parameters(molecule("CC"))
        assert len(result["dihedrals"]) == 9
        assert len(result["dihedral_params"]) == 2
        for group, params in indexed_parameters(result).items():
            assert params["k_kcal_mol"] == (0 if group[0] == 2 else 1)


def geometric_dihedral(positions, group):
    first, second, third, fourth = positions[list(group)]
    axis = third - second
    axis /= np.linalg.norm(axis)
    left = first - second
    right = fourth - third
    left -= np.dot(left, axis) * axis
    right -= np.dot(right, axis) * axis
    return math.atan2(np.dot(np.cross(axis, left), right), np.dot(left, right))


@pytest.mark.parametrize("smiles", ["OO", "CC", "C=C"])
def test_independent_rigid_rotation_energies_and_derivatives(smiles):
    mol = molecule(smiles)
    mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()))
    result = extract_uff_parameters(mol)
    bond_table = {
        group: result["bond_params"][name]
        for group, name in zip(result["bonds"], result["bond_types"])
    }
    angle_table = {
        group: result["angle_params"][name]
        for group, name in zip(result["angles"], result["angle_types"])
    }
    central_length = bond_table[0, 1]["r0_a"]
    forcefield = uff.UFFGetMoleculeForceField(mol, vdwThresh=0)

    def positions(rotation):
        xyz = np.zeros((mol.GetNumAtoms(), 3))
        xyz[1, 0] = central_length
        for center, partner in [(0, 1), (1, 0)]:
            neighbors = sorted(
                a.GetIdx()
                for a in mol.GetAtomWithIdx(center).GetNeighbors()
                if a.GetIdx() != partner
            )
            for index, outer in enumerate(neighbors):
                length = bond_table[tuple(sorted((center, outer)))]["r0_a"]
                angle = angle_table[
                    min(partner, outer), center, max(partner, outer)
                ]["theta0_rad"]
                azimuth = 2 * math.pi * index / len(neighbors) + (
                    rotation if center == 1 else 0
                )
                xyz[outer] = xyz[center] + [
                    (1 if center == 0 else -1) * length * math.cos(angle),
                    length * math.sin(angle) * math.cos(azimuth),
                    length * math.sin(angle) * math.sin(azimuth),
                ]
        return xyz

    def energies(rotation):
        xyz = positions(rotation)
        extracted = sum(
            0.5
            * params["k_kcal_mol"]
            * (
                1
                + params["d"]
                * math.cos(params["n"] * geometric_dihedral(xyz, group))
            )
            for group, params in indexed_parameters(result).items()
        )
        return np.array(
            [forcefield.CalcEnergy(xyz.ravel().tolist()), extracted]
        )

    reference = energies(0.17)
    for rotation in (0.31, 0.83, 1.47, 2.19):
        differences = energies(rotation) - reference
        assert differences[0] == pytest.approx(differences[1], abs=1e-8)
        step = 1e-5
        derivatives = (
            energies(rotation + step) - energies(rotation - step)
        ) / (2 * step)
        assert derivatives[0] == pytest.approx(derivatives[1], abs=1e-7)
