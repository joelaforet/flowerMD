"""Partial charges for the all-atom DPD force field.

Charges are assigned per molecule in elementary charges and cached per
distinct chemical graph, so a melt of identical chains costs one pass. Three
methods are available:

``"formal"``
    The formal charge of each atom, inferred from its explicit valence (see
    `flowermd.internal.all_atom_parameters.assign_formal_charges`). No
    polarization, but exact net charges and no dependencies beyond RDKit.
``"gasteiger"``
    RDKit's Gasteiger-Marsili charges, shifted so each molecule's total is
    its formal charge.
``"nagl"``
    AM1-BCC-like charges from an OpenFF NAGL graph neural network, the
    charges a Sage hand-off would use. Requires openff-toolkit and
    openff-nagl.

A single-atom molecule (a counterion) always gets its formal charge.
"""

import numpy as np

from flowermd.internal.all_atom_parameters import (
    compound_to_rdkit,
    molecular_compounds,
    molecule_graph_key,
)

CHARGE_METHODS = ("formal", "gasteiger", "nagl")
DEFAULT_NAGL_MODEL = "openff-gnn-am1bcc-1.0.0.pt"


def _formal(mol):
    return np.asarray(
        [atom.GetFormalCharge() for atom in mol.GetAtoms()], dtype=float
    )


def _gasteiger(mol):
    from rdkit.Chem import rdPartialCharges

    rdPartialCharges.ComputeGasteigerCharges(mol)
    charges = np.asarray(
        [atom.GetDoubleProp("_GasteigerCharge") for atom in mol.GetAtoms()]
    )
    if not np.all(np.isfinite(charges)):
        raise ValueError(
            "RDKit has no Gasteiger parameters for some atoms in this "
            "molecule; use charges='formal' or 'nagl'."
        )
    return charges


def _nagl(mol, model):
    from openff.toolkit import Molecule
    from openff.toolkit.utils.nagl_wrapper import NAGLToolkitWrapper
    from openff.units import unit

    molecule = Molecule.from_rdkit(
        mol, allow_undefined_stereo=True, hydrogens_are_explicit=True
    )
    molecule.assign_partial_charges(
        partial_charge_method=model, toolkit_registry=NAGLToolkitWrapper()
    )
    return np.asarray(molecule.partial_charges.m_as(unit.elementary_charge))


def _charges_one(molecule, method, nagl_model):
    mol = compound_to_rdkit(molecule)
    formal = _formal(mol)
    if method == "formal" or mol.GetNumAtoms() == 1:
        return formal
    if method == "gasteiger":
        charges = _gasteiger(mol)
    else:
        charges = _nagl(mol, nagl_model)
    # Put any rounding residue back so the molecule carries exactly its
    # formal net charge; PPPM needs an exactly neutral system.
    return charges + (formal.sum() - charges.sum()) / len(charges)


def assign_partial_charges(compound, method, nagl_model=DEFAULT_NAGL_MODEL):
    """Return one partial charge per particle of `compound`, in e.

    Parameters
    ----------
    compound : mbuild.Compound, required
        All-atom compound; every disconnected child is one molecule.
    method : {"formal", "gasteiger", "nagl"}, required
        How to assign charges, see the module docstring.
    nagl_model : str, default "openff-gnn-am1bcc-1.0.0.pt"
        NAGL model file when ``method="nagl"``.

    Returns
    -------
    numpy.ndarray, shape (compound.n_particles,)

    """
    if method not in CHARGE_METHODS:
        raise ValueError(f"method must be one of {CHARGE_METHODS}.")
    cache = {}
    charges = []
    for molecule in molecular_compounds(compound):
        key = molecule_graph_key(molecule)
        if key not in cache:
            cache[key] = _charges_one(molecule, method, nagl_model)
        charges.append(cache[key])
    return np.concatenate(charges)
