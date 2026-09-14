"""Check donor-indexed graph assembly against independent frozen structures."""

import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

from flowermd.library import assemble_polymer_graph

DATA = json.loads(
    (
        Path(__file__).parents[1]
        / "assets/chemical_assembly/frozen_donors.json"
    ).read_text()
)


def donor_from_record(record):
    graph = Chem.RWMol()
    for row in record["atoms"]:
        atom = Chem.Atom(row["atomic_number"])
        atom.SetFormalCharge(row["formal_charge"])
        atom.SetIsotope(row["isotope"])
        atom.SetAtomMapNum(row["map"])
        atom.SetIsAromatic(row["aromatic"])
        atom.SetNoImplicit(True)
        graph.AddAtom(atom)
    types = {
        1.0: Chem.BondType.SINGLE,
        2.0: Chem.BondType.DOUBLE,
        3.0: Chem.BondType.TRIPLE,
        1.5: Chem.BondType.AROMATIC,
    }
    for bond in record["bonds"]:
        graph.AddBond(*bond["members"], types[bond["order"]])
    molecule = graph.GetMol()
    Chem.SanitizeMol(molecule)
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    for i, point in enumerate(record["positions_angstrom"]):
        conformer.SetAtomPosition(i, point)
    molecule.AddConformer(conformer)
    return molecule


def intended_points(donor, report, center, neighbors):
    origins = report["atom_origins"]
    repeat, local = origins[center]["repeat"], origins[center]["donor_index"]
    xyz = donor.GetConformer().GetPositions()
    caps = {
        a.GetAtomMapNum(): a.GetIdx()
        for a in donor.GetAtoms()
        if a.GetAtomMapNum()
    }
    points = []
    for neighbor in neighbors:
        origin = origins[neighbor]
        if origin["repeat"] == repeat:
            local_neighbor = origin["donor_index"]
        else:
            link = next(
                row
                for row in report["junctions"]
                if set(row["atoms"]) == {center, neighbor}
            )
            pair = report["connection_pairs"][link["pair_index"]]
            local_neighbor = caps[pair[0 if repeat < origin["repeat"] else 1]]
        points.append(xyz[local_neighbor] - xyz[local])
    points = np.asarray(points)
    points[:, 2] *= -1 if report["reflections"][repeat] else 1
    return points


def volume_phase(points):
    edges = points[:3] - points[3]
    volume = np.linalg.det(edges) / np.prod(np.linalg.norm(edges, axis=1))
    u, v, w = (
        points[0] - points[1],
        points[2] - points[1],
        points[3] - points[2],
    )
    a, b = np.cross(u, -v), np.cross(w, -v)
    phase = np.arctan2(np.linalg.norm(v) * np.dot(a, w), np.dot(a, b)) % (
        2 * np.pi
    )
    return volume, phase


@pytest.mark.parametrize(
    "reference",
    DATA["references"],
    ids=lambda r: f"{r['chemistry']}-{r['degree']}",
)
def test_all_frozen_graphs_and_donor_targets(reference):
    record = DATA["donors"][reference["chemistry"]]
    donor = donor_from_record(record)
    molecule, report = assemble_polymer_graph(
        donor,
        connection_pairs=record["connection_pairs"],
        repeats=reference["degree"],
        reflections=reference["reflections"],
    )
    assert molecule.GetNumConformers() == 0
    assert molecule.GetNumAtoms() == len(reference["atoms"])
    assert len(Chem.GetMolFrags(molecule)) == 1
    assert all(
        not a.GetNumImplicitHs() and not a.GetAtomMapNum()
        for a in molecule.GetAtoms()
    )
    origins = [
        (row["repeat"], row["donor_index"]) for row in report["atom_origins"]
    ]
    assert len(set(origins)) == molecule.GetNumAtoms()
    index = {origin: i for i, origin in enumerate(origins)}
    mapping = [index[tuple(origin)] for origin in reference["frozen_origins"]]
    assert sorted(mapping) == list(range(molecule.GetNumAtoms()))
    for old, row in enumerate(reference["atoms"]):
        atom = molecule.GetAtomWithIdx(mapping[old])
        assert (
            atom.GetAtomicNum(),
            atom.GetFormalCharge(),
            atom.GetIsotope(),
        ) == (row["atomic_number"], row["formal_charge"], row["isotope"])
    actual = {
        tuple(
            sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
        ): b.GetBondTypeAsDouble()
        for b in molecule.GetBonds()
    }
    expected = {
        tuple(sorted(mapping[i] for i in b["members"])): b["order"]
        for b in reference["bonds"]
    }
    assert actual == expected
    for link in reference["junction_bond_orders"]:
        edge = tuple(sorted(mapping[i] for i in link["atoms"]))
        native = next(
            row
            for row in report["junctions"]
            if tuple(sorted(row["atoms"])) == edge
        )
        assert native["input_bond_order"] == link["input_bond_order"] == 1
        assert native["bond_order"] == link["bond_order"]
    removed = {(r["repeat"], r["donor_index"]) for r in report["removed_caps"]}
    all_origins = {
        (r, i)
        for r in range(reference["degree"])
        for i in range(donor.GetNumAtoms())
    }
    assert set(origins) == all_origins - removed
    caps = {
        a.GetAtomMapNum(): (a.GetIdx(), a.GetNeighbors()[0].GetIdx())
        for a in donor.GetAtoms()
        if a.GetAtomMapNum()
    }
    expected_removed = []
    for junction in range(reference["degree"] - 1):
        expected_pairs = [
            (index[junction, caps[o][1]], index[junction + 1, caps[i][1]])
            for o, i in record["connection_pairs"]
        ]
        rows = [
            row for row in report["junctions"] if row["junction"] == junction
        ]
        assert [tuple(row["atoms"]) for row in rows] == expected_pairs
        assert [row["pair_index"] for row in rows] == list(
            range(len(expected_pairs))
        )
        for outgoing, incoming in record["connection_pairs"]:
            for repeat, label, role in (
                (junction, outgoing, "outgoing"),
                (junction + 1, incoming, "incoming"),
            ):
                expected_removed.append(
                    dict(
                        repeat=repeat,
                        donor_index=caps[label][0],
                        cap_map=label,
                        role=role,
                        junction=junction,
                    )
                )
        for o, i in record["connection_pairs"]:
            assert (junction, caps[o][0]) in removed
            assert (junction + 1, caps[i][0]) in removed
    expected_removed.sort(key=lambda row: (row["repeat"], row["donor_index"]))
    assert report["removed_caps"] == expected_removed
    expected_terminal = [
        dict(
            repeat=repeat,
            donor_index=cap,
            cap_map=label,
            final_index=index[repeat, cap],
            anchor_index=index[repeat, anchor],
        )
        for repeat in range(reference["degree"])
        for label, (cap, anchor) in caps.items()
        if (repeat, cap) in index
    ]
    assert report["terminal_caps"] == expected_terminal
    for i, origin in enumerate(origins):
        assert report["retained_donor_indices"][origin[0]][origin[1]] == i
    assert len(report["stereo_targets"]) == len(
        reference["stereo_reference"]["centers"]
    )
    by_center = {row["center_index"]: row for row in report["stereo_targets"]}
    for center, target in by_center.items():
        volume, phase = volume_phase(
            intended_points(donor, report, center, target["neighbors"])
        )
        assert target["reference_volume"] == pytest.approx(volume, abs=2e-12)
        assert target["phi0"] == pytest.approx(phase, abs=2e-12)
        assert target["expected_sign"] == (1 if volume > 0 else -1)
        assert target["source"] == "independent_labelled_donor_cap_geometry"
    for frozen in reference["stereo_reference"]["centers"]:
        center = mapping[frozen["center_index"]]
        neighbors = [mapping[n] for n in frozen["neighbors"]]
        assert set(by_center[center]["neighbors"]) == set(neighbors)
        volume, phase = volume_phase(
            intended_points(donor, report, center, neighbors)
        )
        assert volume == pytest.approx(frozen["reference_volume"], abs=2e-12)
        assert phase == pytest.approx(frozen["phi0"], abs=2e-12)
        assert by_center[center]["stereo_label"] == frozen["stereo_label"]


def embedded(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomicNum(1)
            atom.SetNoImplicit(True)
    Chem.SanitizeMol(molecule)
    molecule = Chem.AddHs(molecule)
    assert AllChem.EmbedMolecule(molecule, randomSeed=77) == 0
    return molecule


@pytest.mark.parametrize(
    "smiles,pairs",
    [
        ("[*:1]C([2H])(F)C[*:2]", [(1, 2)]),
        ("[*:1]CC([*:3])CC([*:4])C[*:2]", [(1, 2), (3, 4)]),
    ],
)
def test_nonpreset_permutation_maps_reflections_and_ownership(smiles, pairs):
    donor = embedded(smiles)
    sequence = [0, 1, 0]
    before = donor.ToBinary(Chem.PropertyPickleOptions.AllProps)
    molecule, report = assemble_polymer_graph(
        donor, connection_pairs=pairs, repeats=3, reflections=sequence
    )
    assert donor.ToBinary(Chem.PropertyPickleOptions.AllProps) == before
    permutation = list(reversed(range(donor.GetNumAtoms())))
    permuted = Chem.RenumberAtoms(donor, permutation)
    for atom in permuted.GetAtoms():
        if atom.GetAtomMapNum():
            atom.SetAtomMapNum(atom.GetAtomMapNum() + 10)
    other, other_report = assemble_polymer_graph(
        permuted,
        connection_pairs=[(a + 10, b + 10) for a, b in pairs],
        repeats=3,
        reflections=sequence,
    )
    native_indices = {
        (row["repeat"], row["donor_index"]): i
        for i, row in enumerate(report["atom_origins"])
    }
    mapped = [
        native_indices[row["repeat"], permutation[row["donor_index"]]]
        for row in other_report["atom_origins"]
    ]
    assert {
        tuple(
            sorted((mapped[b.GetBeginAtomIdx()], mapped[b.GetEndAtomIdx()]))
        ): b.GetBondTypeAsDouble()
        for b in other.GetBonds()
    } == {
        tuple(
            sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
        ): b.GetBondTypeAsDouble()
        for b in molecule.GetBonds()
    }
    assert Chem.MolToSmiles(other) == Chem.MolToSmiles(molecule)
    assert rdMolDescriptors.CalcMolFormula(
        other
    ) == rdMolDescriptors.CalcMolFormula(molecule)
    assert sum(a.GetIsotope() == 2 for a in molecule.GetAtoms()) == 3 * sum(
        a.GetIsotope() == 2 for a in donor.GetAtoms()
    )
    reflected = Chem.Mol(donor)
    xyz = reflected.GetConformer().GetPositions()
    xyz[:, 2] *= -1
    for i, point in enumerate(xyz):
        reflected.GetConformer().SetAtomPosition(i, point)
    same, _ = assemble_polymer_graph(
        reflected,
        connection_pairs=pairs,
        repeats=3,
        reflections=[1 - value for value in sequence],
    )
    assert Chem.MolToSmiles(same) == Chem.MolToSmiles(molecule)
    report["donor_positions_angstrom"][0][0] = 999
    report["connection_pairs"][0][0] = 999
    sequence[0] = 1
    assert donor.ToBinary(Chem.PropertyPickleOptions.AllProps) == before
    assert other_report["reflections"] == [0, 1, 0]


@pytest.mark.parametrize(
    "options",
    [
        {"repeats": True},
        {"repeats": 0},
        {"reflections": [0]},
        {"reflections": [0, 2]},
        {"cap_policy": "methyl"},
        {"connection_pairs": [(1, 1)]},
        {"connection_pairs": []},
        {"connection_pairs": [(1, 3)]},
    ],
)
def test_invalid_contract(options):
    donor = embedded("[*:1]CC[*:2]")
    args = dict(connection_pairs=[(1, 2)], repeats=2, reflections=[0, 1])
    args.update(options)
    with pytest.raises(ValueError):
        assemble_polymer_graph(donor, **args)


@pytest.mark.parametrize(
    "change",
    [
        "implicit",
        "disconnected",
        "no_conformer",
        "multiple_conformers",
        "nonfinite",
        "duplicate_map",
        "heavy_map",
        "valence",
        "coincident_cap",
        "planar",
    ],
)
def test_invalid_donor_geometry_and_metadata(change):
    donor = donor_from_record(DATA["donors"]["ps"])
    if change == "implicit":
        donor = Chem.MolFromSmiles("CC")
    elif change == "disconnected":
        donor = Chem.CombineMols(donor, donor)
    elif change == "no_conformer":
        donor.RemoveAllConformers()
    elif change == "multiple_conformers":
        donor.AddConformer(Chem.Conformer(donor.GetConformer()), assignId=True)
    elif change == "nonfinite":
        donor.GetConformer().SetAtomPosition(0, (float("nan"), 0, 0))
    elif change == "duplicate_map":
        caps = [a for a in donor.GetAtoms() if a.GetAtomMapNum()]
        caps[1].SetAtomMapNum(caps[0].GetAtomMapNum())
    elif change == "heavy_map":
        next(
            a for a in donor.GetAtoms() if a.GetAtomicNum() == 6
        ).SetAtomMapNum(99)
    elif change == "valence":
        next(a for a in donor.GetAtoms() if a.GetAtomicNum() == 6).SetAtomicNum(
            9
        )
    elif change == "coincident_cap":
        cap = next(a for a in donor.GetAtoms() if a.GetAtomMapNum())
        donor.GetConformer().SetAtomPosition(
            cap.GetIdx(),
            donor.GetConformer().GetAtomPosition(
                cap.GetNeighbors()[0].GetIdx()
            ),
        )
    else:
        for i, point in enumerate(donor.GetConformer().GetPositions()):
            donor.GetConformer().SetAtomPosition(i, (point[0], point[1], 0))
    before = donor.ToBinary(Chem.PropertyPickleOptions.AllProps)
    with pytest.raises(ValueError):
        assemble_polymer_graph(
            donor,
            connection_pairs=DATA["donors"]["ps"]["connection_pairs"],
            repeats=2,
            reflections=[0, 1],
        )
    assert donor.ToBinary(Chem.PropertyPickleOptions.AllProps) == before


def test_single_repeat_retains_isotope_caps_and_has_no_geometry():
    donor = embedded("[*:1]C(F)(Cl)C[*:2]")
    cap = next(a for a in donor.GetAtoms() if a.GetAtomMapNum() == 1)
    cap.SetIsotope(2)
    result, report = assemble_polymer_graph(
        donor, connection_pairs=[(1, 2)], repeats=1, reflections=[0]
    )
    assert result.GetNumAtoms() == donor.GetNumAtoms()
    assert result.GetAtomWithIdx(cap.GetIdx()).GetIsotope() == 2
    assert result.GetNumConformers() == 0
    assert report["removed_caps"] == report["junctions"] == []
    assert len(report["terminal_caps"]) == 2
    assert report["retained_donor_indices"] == [
        dict(enumerate(range(donor.GetNumAtoms())))
    ]
    assert (
        len(report["donor_graph_sha256"])
        == len(report["donor_conformer_sha256"])
        == 64
    )


def test_duplicate_resolved_junctions_rejected():
    donor = embedded("C([*:1])([*:3])C([*:2])[*:4]")
    with pytest.raises(ValueError, match="duplicate junction"):
        assemble_polymer_graph(
            donor,
            connection_pairs=[(1, 2), (3, 4)],
            repeats=2,
            reflections=[0, 0],
        )


def test_formal_charge_and_input_sequences_are_preserved():
    donor = embedded("[*:1]C[N+](C)(C)C[*:2]")
    pairs = [[1, 2]]
    sequence = np.array([0, 1, 0])
    molecule, report = assemble_polymer_graph(
        donor, connection_pairs=pairs, repeats=3, reflections=sequence
    )
    assert sum(a.GetFormalCharge() for a in molecule.GetAtoms()) == 3
    assert report["connection_pairs"] == pairs
    report["connection_pairs"][0][0] = 55
    report["reflections"][0] = 1
    assert pairs == [[1, 2]]
    np.testing.assert_array_equal(sequence, [0, 1, 0])
