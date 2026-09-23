"""Build polymer repeat units from SMILES with marked attachment points.

A repeat unit is written as a SMILES string in which the two bonds to the
neighbouring repeats are marked with dummy atoms ``[*:1]`` (head) and
``[*:2]`` (tail), for example polystyrene ``c1ccc([C@H](C[*:2])[*:1])cc1``.
Writing the repeat this way keeps a stereocenter's chirality tag valid: with
hydrogens in place of the dummies the center would carry two hydrogens and
RDKit would drop the tag.

The dummies are embedded as heavy placeholder atoms so the 3D geometry
honours the chirality, then turned into hydrogens at a C-H bond length. mBuild's
`Polymer` recipe removes exactly those two hydrogens when it forms the chain.

Requires RDKit (imported lazily).
"""

import numpy as np

PLACEHOLDER_ATOMIC_NUMBER = 35  # bromine: heavy, monovalent, embeds cleanly
CH_BOND_LENGTH_NM = 0.109


def _embed_marked(smiles, labels, seed, name):
    """Embed a marked SMILES with heavy placeholders, return (compound, ports).

    ``ports`` maps each attachment label to the particle index of the
    placeholder, which is converted to a hydrogen at a C-H bond length.
    """
    from mbuild.conversion import from_rdkit
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse the SMILES {smiles!r}.")
    editable = Chem.RWMol(mol)
    ports = {}
    for atom in editable.GetAtoms():
        if atom.GetAtomicNum() == 0:
            label = atom.GetAtomMapNum()
            if label not in labels or label in ports:
                raise ValueError(
                    f"Marked SMILES needs exactly one of each {sorted(labels)}."
                )
            ports[label] = atom.GetIdx()
            atom.SetAtomicNum(PLACEHOLDER_ATOMIC_NUMBER)
            atom.SetAtomMapNum(0)
    if set(ports) != set(labels):
        raise ValueError(
            f"Marked SMILES needs exactly one of each {sorted(labels)}."
        )
    placeholder = editable.GetMol()
    Chem.SanitizeMol(placeholder)
    compound = from_rdkit(placeholder, smiles_seed=seed)
    compound.name = name
    particles = list(compound.particles())
    for index in ports.values():
        port = particles[index]
        heavy = next(iter(port.direct_bonds()))
        direction = np.asarray(port.pos) - np.asarray(heavy.pos)
        port.pos = (
            np.asarray(heavy.pos)
            + direction / np.linalg.norm(direction) * CH_BOND_LENGTH_NM
        )
        port.name = "H"
        port.element = "H"
    return compound, ports


def ladder_monomer_from_marked_smiles(smiles, seed=0, name="monomer"):
    """Return ``(compound, ports)`` for a two-bond (ladder) repeat unit.

    The repeat carries four marks: ``[*:1]`` and ``[*:2]`` on the atoms that
    bond to the next repeat, ``[*:3]`` and ``[*:4]`` on the atoms that bond
    to the previous one, paired as 1 with 3 and 2 with 4. ``ports`` maps each
    label to the index of the hydrogen standing in for that bond.
    """
    return _embed_marked(smiles, (1, 2, 3, 4), seed, name)


def monomer_from_marked_smiles(smiles, seed=0, name="monomer"):
    """Return ``(compound, bond_indices)`` for a marked repeat-unit SMILES.

    Parameters
    ----------
    smiles : str, required
        Repeat-unit SMILES with ``[*:1]`` at the head attachment and
        ``[*:2]`` at the tail attachment.
    seed : int, default 0
        RDKit embedding seed.
    name : str, default "monomer"
        Name given to the returned compound.

    Returns
    -------
    compound : mbuild.Compound
        Explicit-hydrogen repeat unit with the two attachment positions
        occupied by hydrogens.
    bond_indices : list of int
        Indices (in ``compound.particles()`` order) of the head and tail
        hydrogens, for `flowermd.base.Polymer`'s ``bond_indices``.

    """
    compound, ports = _embed_marked(smiles, (1, 2), seed, name)
    return compound, [ports[1], ports[2]]
