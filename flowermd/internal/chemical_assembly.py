"""Assemble finite chemical graphs and retain explicit donor correspondences."""

import hashlib
from numbers import Integral

import numpy as np


def assemble_graph(donor, connection_pairs, repeats, reflections, cap_policy):
    """Return an ordinary RDKit graph and owned donor provenance."""
    from rdkit import Chem, rdBase

    if not isinstance(donor, Chem.Mol):
        raise ValueError("donor must be an explicit-H RDKit Mol")
    if (
        isinstance(repeats, bool)
        or not isinstance(repeats, Integral)
        or repeats < 1
    ):
        raise ValueError("repeats must be a positive integer")
    if not isinstance(cap_policy, str) or cap_policy != "hydrogen":
        raise ValueError("only the hydrogen cap policy is supported")
    try:
        sequence = tuple(reflections)
        pairs = tuple(tuple(pair) for pair in connection_pairs)
    except TypeError as error:
        raise ValueError(
            "reflections and connection_pairs must be sequences"
        ) from error
    if len(sequence) != repeats or any(
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or value not in (0, 1)
        for value in sequence
    ):
        raise ValueError(
            "reflections requires exactly one integer 0 or 1 per repeat"
        )
    if not pairs or any(len(pair) != 2 for pair in pairs):
        raise ValueError("connection_pairs must contain directed cap-map pairs")
    labels = tuple(label for pair in pairs for label in pair)
    if any(
        isinstance(label, bool) or not isinstance(label, Integral) or label <= 0
        for label in labels
    ) or len(set(labels)) != len(labels):
        raise ValueError("each positive cap map must occur exactly once")
    molecule = Chem.Mol(donor)
    if not molecule.GetNumAtoms() or len(Chem.GetMolFrags(molecule)) != 1:
        raise ValueError("donor must be nonempty and connected")
    Chem.SanitizeMol(molecule)
    if any(
        a.GetAtomicNum() == 0 or a.GetNumImplicitHs() or a.GetNumExplicitHs()
        for a in molecule.GetAtoms()
    ):
        raise ValueError(
            "donor requires explicit hydrogen atoms and no dummy atoms"
        )
    if molecule.GetNumConformers() != 1:
        raise ValueError(
            "donor requires exactly one finite undistorted conformer"
        )
    xyz = np.array(
        molecule.GetConformer().GetPositions(), dtype=float, copy=True
    )
    if xyz.shape != (molecule.GetNumAtoms(), 3) or not np.isfinite(xyz).all():
        raise ValueError("donor requires finite coordinates for every atom")
    caps = {}
    for atom in molecule.GetAtoms():
        label = atom.GetAtomMapNum()
        if not label:
            continue
        if label in caps or atom.GetAtomicNum() != 1 or atom.GetDegree() != 1:
            raise ValueError(
                "unique cap maps must label singly bonded hydrogen atoms"
            )
        anchor = atom.GetNeighbors()[0].GetIdx()
        if (
            molecule.GetBondBetweenAtoms(atom.GetIdx(), anchor).GetBondType()
            != Chem.BondType.SINGLE
        ):
            raise ValueError("cap hydrogen requires one single bond")
        if (
            np.linalg.norm(xyz[atom.GetIdx()] - xyz[anchor])
            <= np.finfo(float).tiny
        ):
            raise ValueError(
                "cap hydrogen and anchor coordinates must be distinct"
            )
        caps[label] = (atom.GetIdx(), anchor)
    if set(caps) != set(labels):
        raise ValueError(
            "connection_pairs must cover every donor cap map exactly once"
        )
    anchors = [(caps[out][1], caps[inc][1]) for out, inc in pairs]
    if len(set(anchors)) != len(anchors):
        raise ValueError("connection_pairs resolve to duplicate junction bonds")
    donor_hash = hashlib.sha256(
        donor.ToBinary(Chem.PropertyPickleOptions.AllProps)
    ).hexdigest()
    molecule.RemoveAllConformers()
    graph_hash = hashlib.sha256(
        molecule.ToBinary(Chem.PropertyPickleOptions.AllProps)
    ).hexdigest()
    combined = Chem.Mol(molecule)
    for _ in range(repeats - 1):
        combined = Chem.CombineMols(combined, molecule)
    editable = Chem.RWMol(combined)
    size = molecule.GetNumAtoms()
    removed = {}
    junctions = []
    cross_caps = {}
    for junction in range(repeats - 1):
        for pair_index, (outgoing, incoming) in enumerate(pairs):
            out_cap, out_anchor = caps[outgoing]
            in_cap, in_anchor = caps[incoming]
            left, right = (
                junction * size + out_anchor,
                (junction + 1) * size + in_anchor,
            )
            editable.AddBond(left, right, Chem.BondType.SINGLE)
            junctions.append((junction, pair_index, left, right))
            cross_caps[left, right] = out_cap
            cross_caps[right, left] = in_cap
            for repeat, cap, label, role in (
                (junction, out_cap, outgoing, "outgoing"),
                (junction + 1, in_cap, incoming, "incoming"),
            ):
                removed[repeat * size + cap] = dict(
                    repeat=repeat,
                    donor_index=cap,
                    cap_map=int(label),
                    role=role,
                    junction=junction,
                )
    keep = [index for index in range(repeats * size) if index not in removed]
    old_to_new = {old: new for new, old in enumerate(keep)}
    for index in sorted(removed, reverse=True):
        editable.RemoveAtom(index)
    result = editable.GetMol()
    for atom in result.GetAtoms():
        atom.SetAtomMapNum(0)
        if atom.GetChiralTag() in (
            Chem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
        ):
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        if atom.HasProp("_CIPCode"):
            atom.ClearProp("_CIPCode")
    Chem.SanitizeMol(result)
    origins = [dict(repeat=old // size, donor_index=old % size) for old in keep]
    reverse = [
        {
            int(old % size): new
            for old, new in old_to_new.items()
            if old // size == repeat
        }
        for repeat in range(repeats)
    ]
    targets = _donor_targets(result, keep, xyz, size, sequence, cross_caps)
    terminal = []
    for repeat in range(repeats):
        for label, (cap, anchor) in caps.items():
            old = repeat * size + cap
            if old in old_to_new:
                terminal.append(
                    dict(
                        repeat=repeat,
                        donor_index=cap,
                        cap_map=int(label),
                        final_index=old_to_new[old],
                        anchor_index=old_to_new[repeat * size + anchor],
                    )
                )
    return result, {
        "source": "independent_labelled_donor_cap_geometry",
        "rdkit_version": rdBase.rdkitVersion,
        "donor_sha256": donor_hash,
        "donor_graph_sha256": graph_hash,
        "donor_conformer_sha256": hashlib.sha256(
            np.asarray(xyz, dtype="<f8").tobytes()
        ).hexdigest(),
        "donor_positions_angstrom": xyz.tolist(),
        "cap_vectors_angstrom": {
            int(label): (xyz[cap] - xyz[anchor]).tolist()
            for label, (cap, anchor) in caps.items()
        },
        "connection_pairs": [list(map(int, pair)) for pair in pairs],
        "reflections": list(map(int, sequence)),
        "repeats": int(repeats),
        "cap_policy": cap_policy,
        "atom_origins": origins,
        "retained_donor_indices": reverse,
        "removed_caps": [removed[index] for index in sorted(removed)],
        "terminal_caps": terminal,
        "junctions": [
            dict(
                junction=junction,
                pair_index=pair_index,
                atoms=[old_to_new[left], old_to_new[right]],
                input_bond_order=1.0,
                bond_order=result.GetBondBetweenAtoms(
                    old_to_new[left], old_to_new[right]
                ).GetBondTypeAsDouble(),
            )
            for junction, pair_index, left, right in junctions
        ],
        "stereo_targets": targets,
    }


def _donor_targets(molecule, keep, xyz, size, sequence, cross_caps):
    """Use four donor substituents with the frozen signed-volume convention."""
    from rdkit import Chem

    possible = [
        int(info.centeredOn)
        for info in Chem.FindPotentialStereo(
            molecule, cleanIt=False, flagPossible=True
        )
        if info.type == Chem.StereoType.Atom_Tetrahedral
    ]
    targets = []
    for center in possible:
        atom = molecule.GetAtomWithIdx(center)
        neighbors = sorted(
            neighbor.GetIdx() for neighbor in atom.GetNeighbors()
        )
        if len(neighbors) != 4:
            raise ValueError(
                "potential tetrahedral centers require four explicit neighbors"
            )
        old_center = keep[center]
        repeat, donor_center = divmod(old_center, size)
        points = []
        for neighbor in neighbors:
            old_neighbor = keep[neighbor]
            donor_neighbor = (
                old_neighbor % size
                if old_neighbor // size == repeat
                else cross_caps[old_center, old_neighbor]
            )
            points.append(xyz[donor_neighbor] - xyz[donor_center])
        points = np.array(points)
        if sequence[repeat]:
            points[:, 2] *= -1
        edges = points[:3] - points[3]
        denominator = float(np.prod(np.linalg.norm(edges, axis=1)))
        volume = (
            float(np.linalg.det(edges) / denominator)
            if denominator > np.finfo(float).tiny
            else 0.0
        )
        if not np.isfinite(volume) or abs(volume) < 0.05:
            raise ValueError(
                "donor tetrahedral geometry is nonfinite or near planar"
            )
        left, middle, right = (
            points[0] - points[1],
            points[2] - points[1],
            points[3] - points[2],
        )
        first, second = np.cross(left, -middle), np.cross(right, -middle)
        x, y = (
            float(first @ second),
            float(np.linalg.norm(middle) * (first @ right)),
        )
        if x * x + y * y <= np.finfo(float).tiny:
            raise ValueError(
                "donor tetrahedral geometry has an undefined dihedral"
            )
        phi0 = round(float(np.arctan2(y, x) % (2 * np.pi)), 12)
        probe = Chem.RWMol()
        probe.AddAtom(Chem.Atom("C"))
        for symbol in ("F", "Cl", "Br", "I"):
            index = probe.AddAtom(Chem.Atom(symbol))
            probe.AddBond(0, index, Chem.BondType.SINGLE)
        Chem.SanitizeMol(probe)
        conformer = Chem.Conformer(5)
        conformer.SetAtomPosition(0, (0.0, 0.0, 0.0))
        for index, neighbor in enumerate(atom.GetNeighbors(), 1):
            conformer.SetAtomPosition(
                index, points[neighbors.index(neighbor.GetIdx())]
            )
        probe.AddConformer(conformer)
        Chem.AssignAtomChiralTagsFromStructure(probe, replaceExistingTags=True)
        atom.SetChiralTag(probe.GetAtomWithIdx(0).GetChiralTag())
        targets.append(
            dict(
                center_index=center,
                neighbors=neighbors,
                expected_sign=1 if volume > 0 else -1,
                reference_volume=volume,
                phi0=phi0,
                source="independent_labelled_donor_cap_geometry",
            )
        )
    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    final_possible = {
        int(info.centeredOn)
        for info in Chem.FindPotentialStereo(
            molecule, cleanIt=False, flagPossible=True
        )
        if info.type == Chem.StereoType.Atom_Tetrahedral
    }
    if final_possible != set(possible):
        raise ValueError(
            "donor target assignment changed potential tetrahedral coverage"
        )
    for target in targets:
        atom = molecule.GetAtomWithIdx(target["center_index"])
        target["stereo_label"] = (
            atom.GetProp("_CIPCode")
            if atom.HasProp("_CIPCode")
            else "unspecified"
        )
    return targets
