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
            if label not in (1, 2) or label in ports:
                raise ValueError(
                    "Marked SMILES needs exactly one [*:1] and one [*:2]."
                )
            ports[label] = atom.GetIdx()
            atom.SetAtomicNum(PLACEHOLDER_ATOMIC_NUMBER)
            atom.SetAtomMapNum(0)
    if set(ports) != {1, 2}:
        raise ValueError("Marked SMILES needs exactly one [*:1] and one [*:2].")
    placeholder = editable.GetMol()
    Chem.SanitizeMol(placeholder)
    compound = from_rdkit(placeholder, smiles_seed=seed)
    compound.name = name
    particles = list(compound.particles())
    for label in (1, 2):
        port = particles[ports[label]]
        heavy = next(iter(port.direct_bonds()))
        direction = np.asarray(port.pos) - np.asarray(heavy.pos)
        port.pos = (
            np.asarray(heavy.pos)
            + direction / np.linalg.norm(direction) * CH_BOND_LENGTH_NM
        )
        port.name = "H"
        port.element = "H"
    return compound, [ports[1], ports[2]]
