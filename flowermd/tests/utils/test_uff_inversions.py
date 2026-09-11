import json
import math
from copy import deepcopy

import gmso
import numpy as np
import pytest

from flowermd.internal.uff import extract_uff_parameters
from flowermd.internal.uff_gmso import assign_uff_parameters
from flowermd.tests.utils.test_uff_gmso import groups, inputs

Chem = pytest.importorskip("rdkit.Chem")
uff = pytest.importorskip("rdkit.Chem.rdForceFieldHelpers")


class TestUFFInversions:
    @pytest.mark.parametrize(
        "smiles,expected_k", [("C=O", 50 / 3), ("C=C", 6 / 3)]
    )
    def test_native_constants_three_out_roles_and_legacy_contract(
        self, smiles, expected_k
    ):
        top, mol, mapping = inputs(smiles)
        result, report = assign_uff_parameters(top, mol, atom_map=mapping)
        centers = {group[0] for group in groups(result, "impropers")}
        assert centers
        for center in centers:
            terms = [
                (group, item)
                for group, item in zip(
                    groups(result, "impropers"), result.impropers
                )
                if group[0] == center
            ]
            assert len(terms) == 3
            assert len({group[3] for group, _ in terms}) == 3
            for group, item in terms:
                p = item.improper_type.parameters
                assert p["k"].to_value("kcal/mol") == pytest.approx(expected_k)
                assert tuple(float(p[name]) for name in ("c0", "c1", "c2")) == (
                    1,
                    -1,
                    0,
                )
                assert item.improper_type.tags == {
                    "form": "uff_inversion",
                    "coordinate": "wilson_out_of_plane",
                    "member_convention": "center,plane1,plane2,out",
                }
        assert result.is_fully_typed(group="impropers")
        assert report["assigned_counts"]["impropers"] == result.n_impropers
        assert report["retained_untyped_impropers"] == 0
        assert "backend required" in report["improper_force_backend"]
        assert "impropers" not in extract_uff_parameters(mol)

    def test_phosphine_nonplanar_form(self):
        top, mol, mapping = inputs("P")
        result, _ = assign_uff_parameters(top, mol, atom_map=mapping)
        assert result.n_impropers == 3
        p = result.impropers[0].improper_type.parameters
        target = math.radians(84.4339)
        assert float(p["c1"]) == pytest.approx(-4 * math.cos(target))
        assert float(p["c2"]) == 1
        assert float(
            p["c0"]
            + p["c1"] * math.cos(target)
            + p["c2"] * math.cos(2 * target)
        ) == pytest.approx(0, abs=1e-14)

    def test_mapping_input_preservation_and_serialization(self, tmp_path):
        original, mol, _ = inputs("C=O")
        permutation = list(reversed(range(mol.GetNumAtoms())))
        mol = Chem.RenumberAtoms(mol, permutation)
        before = mol.ToBinary(Chem.PropertyPickleOptions.AllProps)
        existing = groups(original, "impropers")
        result, _ = assign_uff_parameters(
            original, mol, atom_map=dict(enumerate(permutation))
        )
        for center, first, second, out in groups(result, "impropers"):
            assert original.sites[center].element.atomic_number == 6
            assert len({center, first, second, out}) == 4
        assert groups(original, "impropers") == existing
        assert all(item.improper_type is None for item in original.impropers)
        assert mol.ToBinary(Chem.PropertyPickleOptions.AllProps) == before
        filename = tmp_path / "inversions.json"
        result.save(filename)
        # Full Topology.load is not asserted: GMSO 0.17 has independent
        # nested-parameter and improper-restoration bugs in its JSON loader.
        payload = json.loads(filename.read_text())
        assert tuple(
            tuple(item["connection_members"]) for item in payload["impropers"]
        ) == groups(result, "impropers")
        saved_types = {item["id"]: item for item in payload["improper_types"]}
        for saved, improper in zip(payload["impropers"], result.impropers):
            record = deepcopy(saved_types[saved["improper_type"]])
            del record["id"]
            restored = gmso.ImproperType.model_validate(record)
            potential = improper.improper_type
            assert restored.expression == potential.expression
            assert restored.parameters == potential.parameters
            assert restored.tags == potential.tags
            for name in restored.parameters:
                assert (
                    restored.parameters[name].units
                    == potential.parameters[name].units
                )

    def test_other_terms_unchanged_and_ineligible_groups_reported(self):
        top, mol, mapping = inputs("CC.C=O")
        partial, partial_report = assign_uff_parameters(
            top, mol, atom_map=mapping, include_impropers=False
        )
        full, report = assign_uff_parameters(top, mol, atom_map=mapping)
        for kind in ("bonds", "angles", "dihedrals"):
            assert groups(partial, kind) == groups(full, kind)
            for first, second in zip(
                getattr(partial, kind), getattr(full, kind)
            ):
                assert (
                    first.connection_type.parameters
                    == second.connection_type.parameters
                )
        assert report["removed_unassigned_groups"]["impropers"]
        assert partial_report["retained_untyped_impropers"] == top.n_impropers
        assert full.n_impropers == 3

    @pytest.mark.parametrize("value", [None, -1, math.inf, math.nan])
    def test_invalid_eligible_getter_is_indexed(self, monkeypatch, value):
        top, mol, mapping = inputs("C=O")
        monkeypatch.setattr(uff, "GetUFFInversionParams", lambda *args: value)
        with pytest.raises(ValueError, match=r"inversion.*atoms \("):
            assign_uff_parameters(top, mol, atom_map=mapping)
        assert not top.is_typed()

    def test_zero_terms_retained(self, monkeypatch):
        top, mol, mapping = inputs("C=O")
        monkeypatch.setattr(uff, "GetUFFInversionParams", lambda *args: 0)
        result, _ = assign_uff_parameters(top, mol, atom_map=mapping)
        assert result.n_impropers == 3
        assert all(
            item.improper_type.parameters["k"] == 0 for item in result.impropers
        )


@pytest.mark.parametrize("smiles", ["C=O", "P"])
def test_independent_native_uff_energy_and_gradient(smiles):
    top, mol, mapping = inputs(smiles)
    mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()))
    typed, _ = assign_uff_parameters(top, mol, atom_map=mapping)
    native = uff.UFFGetMoleculeForceField(mol, vdwThresh=0)
    bond_terms = [
        (
            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
            uff.GetUFFBondStretchParams(
                mol, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            ),
        )
        for bond in mol.GetBonds()
    ]
    angle_terms = []
    for atom in mol.GetAtoms():
        neighbors = [a.GetIdx() for a in atom.GetNeighbors()]
        for index, first in enumerate(neighbors):
            for third in neighbors[index + 1 :]:
                group = (first, atom.GetIdx(), third)
                angle_terms.append(
                    (group, uff.GetUFFAngleBendParams(mol, *group))
                )

    def native_bonds_angles(xyz):
        total = 0.0
        for (first, second), (k, r0) in bond_terms:
            total += (
                0.5 * k * (np.linalg.norm(xyz[first] - xyz[second]) - r0) ** 2
            )
        for (first, center, third), (k, degrees) in angle_terms:
            left, right = xyz[first] - xyz[center], xyz[third] - xyz[center]
            cosine = (
                np.dot(left, right)
                / np.linalg.norm(left)
                / np.linalg.norm(right)
            )
            theta = math.acos(np.clip(cosine, -1, 1))
            if smiles == "C=O":
                total += k / 9 * (1 - math.cos(3 * theta))
            else:
                target = math.radians(degrees)
                c2 = 1 / (4 * math.sin(target) ** 2)
                c1 = -4 * c2 * math.cos(target)
                c0 = c2 * (2 * math.cos(target) ** 2 + 1)
                total += k * (
                    c0 + c1 * math.cos(theta) + c2 * math.cos(2 * theta)
                )
        return total

    def inversion_energy(xyz):
        total = 0.0
        for group, item in zip(groups(typed, "impropers"), typed.impropers):
            center, first, second, out = xyz[list(group)]
            normal = np.cross(first - center, second - center)
            direction = out - center
            sine = (
                np.dot(normal, direction)
                / np.linalg.norm(normal)
                / np.linalg.norm(direction)
            )
            omega = math.asin(np.clip(sine, -1, 1))
            p = item.improper_type.parameters
            total += float(p["k"].to_value("kcal/mol")) * (
                float(p["c0"])
                + float(p["c1"]) * math.cos(omega)
                + float(p["c2"]) * math.cos(2 * omega)
            )
        return total

    geometries = (
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.2, 0.0, 0.3],
                [-0.6, 0.9, 0.1],
                [-0.6, -0.9, -0.2],
            ]
        )
        if smiles == "C=O"
        else np.array(
            [
                [0.0, 0.0, 0.0],
                [1.4, 0.0, 0.1],
                [0.0, 1.3, -0.2],
                [-0.15, 0.12, 1.45],
            ]
        )
    )
    for perturbation in (0.0, 0.17):
        xyz = geometries.copy()
        xyz[3, 2] += perturbation
        energy = native.CalcEnergy(xyz.ravel().tolist()) - native_bonds_angles(
            xyz
        )
        assert energy == pytest.approx(inversion_energy(xyz), abs=1e-8)
        residual_gradient = np.array(
            native.CalcGrad(xyz.ravel().tolist())
        ).reshape(xyz.shape)
        inversion_gradient = np.zeros_like(xyz)
        step = 1e-6
        for index in np.ndindex(xyz.shape):
            plus, minus = xyz.copy(), xyz.copy()
            plus[index] += step
            minus[index] -= step
            residual_gradient[index] -= (
                native_bonds_angles(plus) - native_bonds_angles(minus)
            ) / (2 * step)
            inversion_gradient[index] = (
                inversion_energy(plus) - inversion_energy(minus)
            ) / (2 * step)
        np.testing.assert_allclose(
            residual_gradient, inversion_gradient, atol=3e-5
        )
