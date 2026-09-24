"""Polymer and CoPolymer example classes."""

import os
import random

import mbuild as mb
import numpy as np
from mbuild.coordinate_transform import z_axis_transform

from flowermd import CoPolymer, Molecule, Polymer
from flowermd.assets import MON_DIR
from flowermd.internal import check_return_iterable
from flowermd.internal.monomers import (
    ladder_monomer_from_marked_smiles,
    monomer_from_marked_smiles,
)


class PolyEthylene(Polymer):
    """Create a Poly(ethylene) chain.

    Parameters
    ----------
    lengths : int, required
        The number of monomer repeat units in the chain.
    num_mols : int, required
        The number of chains to create.
    name : str, default 'polyethylene'
        The name of the polymer. Setting the name is
        important for using the `speedup_by_moltag=True`
        parameter with polydisperse systems, or other
        mixtures. This helps improve performance
        for large systems.

    """

    def __init__(self, lengths, num_mols, name="polyethylene", **kwargs):
        smiles = "CC"
        bond_indices = [2, 6]
        bond_length = 0.145
        bond_orientation = [None, None]
        super(PolyEthylene, self).__init__(
            lengths=lengths,
            num_mols=num_mols,
            smiles=smiles,
            name=name,
            bond_indices=bond_indices,
            bond_length=bond_length,
            bond_orientation=bond_orientation,
            **kwargs,
        )


class PPS(Polymer):
    """Create a Poly(phenylene-sulfide) (PPS) chain.

    Parameters
    ----------
    lengths : int, required
        The number of monomer repeat units in the chain.
    num_mols : int, required
        The number of chains to create.
    name : str, default 'pps'
        The name of the polymer. Setting the name is
        important for using the `speedup_by_moltag=True`
        parameter with polydisperse systems, or other
        mixtures. This helps improve performance
        for large systems.

    """

    def __init__(self, lengths, num_mols, name="pps", **kwargs):
        smiles = "c1ccc(S)cc1"
        file = None
        bond_indices = [7, 10]
        bond_length = 0.176
        bond_orientation = [[0, 0, 1], [0, 0, -1]]
        super(PPS, self).__init__(
            lengths=lengths,
            num_mols=num_mols,
            smiles=smiles,
            name=name,
            file=file,
            bond_indices=bond_indices,
            bond_length=bond_length,
            bond_orientation=bond_orientation,
            **kwargs,
        )

    def _load(self):
        monomer = mb.load(self.smiles, smiles=True)
        # Need to align monomer along zx plane due to orientation of S-H bond
        z_axis_transform(
            monomer, point_on_z_axis=monomer[7], point_on_zx_plane=monomer[4]
        )
        return monomer


class PEEK(Polymer):
    """Create a Poly(ether-ether-ketone) (PEEK) chain.

    Parameters
    ----------
    lengths : int, required
        The number of monomer repeat units in the chain.
    num_mols : int, required
        The number of chains to create.
    name : str, default 'peek'
        The name of the polymer. Setting the name is
        important for using the `speedup_by_moltag=True`
        parameter with polydisperse systems, or other
        mixtures. This helps improve performance
        for large systems.

    """

    def __init__(self, lengths, num_mols, name="peek", **kwargs):
        smiles = "Oc1ccc(Oc2ccc(C(=O)c3ccccc3)cc2)cc1"
        file = os.path.join(MON_DIR, "peek.mol2")
        bond_indices = [35, 34]
        bond_length = 0.1376
        bond_orientation = [[-1, 0, 0], [1, 0, 0]]
        super(PEEK, self).__init__(
            lengths=lengths,
            num_mols=num_mols,
            smiles=smiles,
            name=name,
            file=file,
            bond_indices=bond_indices,
            bond_length=bond_length,
            bond_orientation=bond_orientation,
            **kwargs,
        )


class PEKK(CoPolymer):
    """Create a Poly(ether-ketone-ketone) (PEKK) chain.

    Creates a polymer chain with two different monomer types,
    represented by the para (T) and meta (I) isomeric forms of PEKK.

    Parameters
    ----------
    lengths : int, required
        The number of monomer repeat units in the chain.
    num_mols : int, required
        The number of chains to create.
    sequence : str, default None
        Manually define the sequence of para (T) and meta (I) monomers.
        Leave as None if generating random sequences.
        Example: sequence = "TTITTITTI"
    TI_ratio : float, required
        The ratio of meta to para isomers in the chain.

    """

    def __init__(
        self,
        lengths,
        num_mols,
        force_field=None,
        sequence=None,
        name="pekk",
        TI_ratio=0.50,
        seed=24,
    ):
        super(PEKK, self).__init__(
            monomer_A=PEKK_meta,
            monomer_B=PEKK_para,
            lengths=lengths,
            name=name,
            num_mols=num_mols,
            force_field=force_field,
            sequence=sequence,
            AB_ratio=TI_ratio,
            seed=seed,
        )


class PEKK_para(Polymer):
    """Create a Poly(ether-ketone-ketone) (PEKK) chain.

    The bonding positions of consecutive ketone groups
    takes place on the para site of the phenyl ring.

    Parameters
    ----------
    lengths : int, required
        The number of monomer repeat units in the chain.
    num_mols : int, required
        The number of chains to create.
    name : str, default 'pekk_para'
        The name of the polymer. Setting the name is
        important for using the `speedup_by_moltag=True`
        parameter with polydisperse systems, or other
        mixtures. This helps improve performance
        for large systems.

    """

    def __init__(self, lengths, num_mols, name="pekk_para"):
        smiles = "c1ccc(Oc2ccc(C(=O)c3ccc(C(=O))cc3)cc2)cc1"
        file = os.path.join(MON_DIR, "pekk_para.mol2")
        bond_indices = [35, 36]
        bond_length = 0.148
        bond_orientation = [[0, 0, -1], [0, 0, 1]]
        super(PEKK_para, self).__init__(
            lengths=lengths,
            num_mols=num_mols,
            smiles=smiles,
            name=name,
            file=file,
            bond_indices=bond_indices,
            bond_length=bond_length,
            bond_orientation=bond_orientation,
        )


class PEKK_meta(Polymer):
    """Create a Poly(ether-ketone-ketone) (PEKK) chain.

    The bonding positions of consecutive ketone groups
    takes place on the meta site of the phenyl ring.

    Parameters
    ----------
    lengths : int, required
        The number of monomer repeat units in the chain.
    num_mols : int, required
        The number of chains to create.
    name : str, default 'pekk_meta'
        The name of the polymer. Setting the name is
        important for using the `speedup_by_moltag=True`
        parameter with polydisperse systems, or other
        mixtures. This helps improve performance
        for large systems.

    """

    def __init__(self, lengths, num_mols, name="pekk_meta"):
        smiles = "c1cc(Oc2ccc(C(=O)c3cc(C(=O))ccc3)cc2)ccc1"
        file = os.path.join(MON_DIR, "pekk_meta.mol2")
        bond_indices = [35, 36]
        bond_length = 0.148
        bond_orientation = [[0, 0, -1], [0, 0, 1]]
        super(PEKK_meta, self).__init__(
            lengths=lengths,
            num_mols=num_mols,
            smiles=smiles,
            name=name,
            file=file,
            bond_indices=bond_indices,
            bond_length=bond_length,
            bond_orientation=bond_orientation,
        )


class LJChain(Polymer):
    """Create a coarse-grained bead-spring polymer chain.

    Parameters
    ----------
    lengths : int, required
        The number of times to repeat bead_sequence in a single chain.
    bead_sequence : list; default ["A"]
        The sequence of bead types in the chain.
    bond_length : dict; optional; default {"A-A": 1.0}
        The bond length between connected beads (units: nm).
    bead_mass : dict; default {"A": 1.0}
        The mass of the bead types.
    name : str, default 'lj_chain'
        The name of the polymer. Setting the name is
        important for using the `speedup_by_moltag=True`
        parameter with polydisperse systems, or other
        mixtures. This helps improve performance
        for large systems.

    """

    def __init__(
        self,
        lengths,
        num_mols,
        bead_sequence=["A"],
        bead_mass={"A": 1.0},
        bond_lengths={"A-A": 1.0},
        name="lj_chain",
    ):
        self.bead_sequence = bead_sequence
        self.bead_mass = bead_mass
        self.bond_lengths = bond_lengths
        super(LJChain, self).__init__(
            lengths=lengths, num_mols=num_mols, name=name
        )

    def _build(self, length):
        chain = mb.Compound()
        last_bead = None
        for i in range(length):
            for idx, bead_type in enumerate(self.bead_sequence):
                mass = self.bead_mass.get(bead_type, None)
                if not mass:
                    raise ValueError(
                        f"The bead mass for {bead_type} was not given "
                        "in the bead_mass dict."
                    )
                next_bead = mb.Compound(mass=mass, name=bead_type, charge=0)
                chain.add(next_bead)
                if last_bead:
                    bead_pair = "-".join([last_bead.name, next_bead.name])
                    bond_length = self.bond_lengths.get(bead_pair, None)
                    if not bond_length:
                        bead_pair_rev = "-".join(
                            [next_bead.name, last_bead.name]
                        )
                        bond_length = self.bond_lengths.get(bead_pair_rev, None)
                        if not bond_length:
                            raise ValueError(
                                "The bond length for pair "
                                f"{bead_pair} or {bead_pair_rev} "
                                "is not found in the bond_lengths dict."
                            )
                    new_pos = last_bead.xyz[0] + (0, 0, bond_length)
                    next_bead.translate_to(new_pos)
                    chain.add_bond([next_bead, last_bead])
                last_bead = next_bead
            chain.name = f"{self.name}_{length}mer"
        return chain


class EllipsoidChain(Polymer):
    """Create an ellipsoid polymer chain.

    This is a coarse-grained molecule where each monomer is modeled
    as an anisotropic bead (i.e. ellipsoid).

    Notes
    -----
    In order to form chains of connected ellipsoids, "ghost"
    particles of types "A" and "B" are used.

    This is meant to be used with
    `flowermd.library.forcefields.EllipsoidForcefield`
    and requires using `flowermd.utils.constraints.set_bond_constraints` to set up
    the fixed bonds correctly in HOOMD-Blue.

    Parameters
    ----------
    lengths : int, required
        The number of monomer repeat units in the chain.
    num_mols : int, required
        The number of chains to create.
    lpar : float, required
        The semi-axis length of the ellipsoid bead along its major axis.
    bead_mass : float, required
        The mass of the ellipsoid bead.
    name : str, default 'ellipsoid_chain'
        The name of the polymer. Setting the name is
        important for using the `speedup_by_moltag=True`
        parameter with polydisperse systems, or other
        mixtures. This helps improve performance
        for large systems.
    """

    def __init__(
        self,
        lengths,
        num_mols,
        lpar,
        bead_mass,
        bond_L=0.1,
        name="ellipsoid_chain",
    ):
        self.bead_mass = bead_mass
        self.lpar = lpar
        self.bond_L = bond_L
        # get the indices of the particles in a rigid body
        self.bead_constituents_types = ["X", "A", "T", "T"]
        super(EllipsoidChain, self).__init__(
            lengths=lengths, num_mols=num_mols, name=name
        )

    def _build(self, length):
        # Build up ellipsoid bead
        bead = mb.Compound(name="ellipsoid")
        center = mb.Compound(pos=(0, 0, 0), name="X", mass=self.bead_mass / 4)
        head = mb.Compound(
            pos=(0, 0, self.lpar + (self.bond_L / 2)),
            name="A",
            mass=self.bead_mass / 4,
        )
        tether_head = mb.Compound(
            pos=(0, 0, self.lpar), name="T", mass=self.bead_mass / 4
        )
        tether_tail = mb.Compound(
            pos=(0, 0, -self.lpar), name="T", mass=self.bead_mass / 4
        )
        bead.add([center, head, tether_head, tether_tail])
        bead.add_bond([center, head])

        chain = mb.Compound()
        last_bead = None
        for i in range(length):
            translate_by = np.array([0, 0, (i * self.lpar * 2) + self.bond_L])
            this_bead = mb.clone(bead)
            this_bead.translate(by=translate_by)
            chain.add(this_bead)
            if last_bead:
                chain.add_bond([this_bead.children[0], last_bead.children[1]])
                chain.add_bond([this_bead.children[3], last_bead.children[2]])
            last_bead = this_bead
        chain.name = f"{self.name}_{length}mer"

        return chain


class EllipsoidChainRand(Polymer):
    """Create an ellipsoid polymer chain in a random walk configuration, considering density and box size.

    This is a coarse-grained molecule where each monomer is modeled
    as an anisotropic bead (i.e. ellipsoid).

    Notes
    -----
    In order to form chains of connected ellipsoids, "ghost"
    particles of types "A" are used.

    This is meant to be used with
    `flowermd.library.forcefields.EllipsoidFF_DPD`
    and requires using `flowermd.utils.constraints.set_bond_constraints` to set up
    the fixed bonds correctly in HOOMD-Blue.

    Parameters
    ----------
    lengths : int, required
        The number of monomer repeat units in the chain.
    num_mols : int, required
        The number of chains to create.
    lpar : float, required
        The semi-axis length of the ellipsoid bead along its major axis.
    bead_mass : float, required
        The mass of the ellipsoid bead.
    density : float, required
        Number density used to calculate box lengths for random walk position range.
    name : str, default 'ellipsoid_chain'
        The name of the polymer. Setting the name is
        important for using the `speedup_by_moltag=True`
        parameter with polydisperse systems, or other
        mixtures. This helps improve performance
        for large systems.
    """

    def __init__(
        self,
        lengths,
        num_mols,
        lpar,
        bead_mass,
        density,
        bond_L=0.1,  # T-T bond length
        name="ellipsoid_chain",
    ):
        self.bead_mass = bead_mass
        self.lpar = lpar
        self.bond_L = bond_L
        self.density = density
        N = lengths * num_mols
        L = np.cbrt(N / self.density)
        self.L = L
        self.box = mb.Box(lengths=np.array([L] * 3))
        self.bead_constituents_types = ["X", "A", "T", "T"]
        super(EllipsoidChainRand, self).__init__(
            lengths=lengths, num_mols=num_mols, name=name
        )

    def _build(self, length):
        bead = mb.Compound(name="ellipsoid")
        center = mb.Compound(pos=(0, 0, 0), name="X", mass=self.bead_mass / 4)
        head = mb.Compound(
            pos=(0, 0, self.lpar + (self.bond_L / 2)),
            name="A",
            mass=self.bead_mass / 4,
        )
        tether_head = mb.Compound(
            pos=(0, 0, self.lpar), name="T", mass=self.bead_mass / 4
        )
        tether_tail = mb.Compound(
            pos=(0, 0, -self.lpar), name="T", mass=self.bead_mass / 4
        )
        bead.add([center, head, tether_head, tether_tail])
        bead.add_bond([center, head])
        chain = mb.Compound()
        last_bead = None
        rand_range = (self.L / 2) - (
            self.lpar + (self.bond_L / 2)
        )  # reducing step size for random walk
        for i in range(length):
            translate_by = np.random.uniform(low=-1, high=1, size=(3,))
            translate_by /= np.linalg.norm(translate_by) * self.bond_L
            this_bead = mb.clone(bead)

            if last_bead:
                chain.add_bond([this_bead.children[0], last_bead.children[1]])
                chain.add_bond([this_bead.children[3], last_bead.children[2]])
                this_bead.translate(
                    by=self.pbc(
                        translate_by + last_bead.pos,
                        pos_range=([rand_range] * 3),
                    )
                )
            else:
                translate_by = np.random.uniform(
                    low=-rand_range, high=rand_range, size=(3,)
                )
                this_bead.translate(
                    by=self.pbc(translate_by, pos_range=([rand_range] * 3))
                )
            chain.add(this_bead)
            last_bead = this_bead
        chain.name = f"{self.name}_{length}mer"
        return chain

    def pbc(self, d, pos_range):
        """Periodic boundary conditions for a reduced box considering position of A beads."""
        for i in range(3):
            while d[i] > pos_range[i] or d[i] < -(pos_range[i]):
                if d[i] < -pos_range[i]:
                    d[i] += pos_range[i]
                if d[i] > pos_range[i]:
                    d[i] -= pos_range[i]
        return d


class MarkedSmilesPolymer(Polymer):
    """A polymer whose repeat unit is a SMILES with marked attachment points.

    Subclasses set ``smiles`` (with ``[*:1]`` at the head and ``[*:2]`` at
    the tail), ``bond_length`` (nm) and, for reference, ``reference_density``
    in g/cm**3. The repeat unit is embedded with RDKit so chirality tags are
    honoured, and the two attachment hydrogens become the bonding sites. See
    `flowermd.internal.monomers.monomer_from_marked_smiles`.

    Parameters
    ----------
    lengths : int or list, required
        Repeat units per chain.
    num_mols : int or list, required
        Chains per length.
    seed : int, default 0
        RDKit embedding seed for the repeat unit.
    name : str, optional
        Compound name; defaults to the class attribute ``default_name``.

    """

    smiles = None
    bond_length = 0.154
    reference_density = None
    default_name = "polymer"

    def __init__(self, lengths, num_mols, seed=0, name=None, **kwargs):
        if self.smiles is None:
            raise NotImplementedError("Subclasses must define `smiles`.")
        monomer, bond_indices = monomer_from_marked_smiles(
            self.smiles, seed=seed, name=name or self.default_name
        )
        self.marked_smiles = self.smiles
        super(MarkedSmilesPolymer, self).__init__(
            lengths=lengths,
            num_mols=num_mols,
            compound=monomer,
            bond_indices=bond_indices,
            bond_length=self.bond_length,
            bond_orientation=[None, None],
            name=name or self.default_name,
            **kwargs,
        )


class PET(MarkedSmilesPolymer):
    """Poly(ethylene terephthalate). Amorphous density about 1.33 g/cm**3."""

    smiles = "O=C(OCCO[*:2])c1ccc(C(=O)[*:1])cc1"
    bond_length = 0.134
    reference_density = 1.33
    default_name = "pet"


class Polycarbonate(MarkedSmilesPolymer):
    """Bisphenol-A polycarbonate. Amorphous density about 1.20 g/cm**3."""

    smiles = "CC(C)(c1ccc(OC(=O)O[*:2])cc1)c1ccc([*:1])cc1"
    bond_length = 0.136
    reference_density = 1.20
    default_name = "pc"


class PEI(MarkedSmilesPolymer):
    """Polyetherimide (Ultem type). Amorphous density about 1.27 g/cm**3."""

    smiles = (
        "CC(C)(c1ccc(Oc2ccc3c(c2)C(=O)N(c2cccc([*:2])c2)C3=O)cc1)"
        "c1ccc(Oc2ccc3c(c2)C(=O)N([*:1])C3=O)cc1"
    )
    bond_length = 0.140
    reference_density = 1.27
    default_name = "pei"


class P3HT(MarkedSmilesPolymer):
    """Poly(3-hexylthiophene), head-to-tail. Density about 1.09 g/cm**3."""

    smiles = "[*:1]c1sc([*:2])c(CCCCCC)c1"
    bond_length = 0.145
    reference_density = 1.094
    default_name = "p3ht"


class PES(MarkedSmilesPolymer):
    """Poly(ether sulfone), [-O-C6H4-SO2-C6H4-]n. Density about 1.37 g/cm**3."""

    smiles = "O=S(=O)(c1ccc(O[*:1])cc1)c1ccc([*:2])cc1"
    bond_length = 0.138
    reference_density = 1.37
    default_name = "pes"


class _PolystyreneR(MarkedSmilesPolymer):
    smiles = "c1ccc([C@H](C[*:2])[*:1])cc1"
    default_name = "ps"


class _PolystyreneS(MarkedSmilesPolymer):
    smiles = "c1ccc([C@@H](C[*:2])[*:1])cc1"
    default_name = "ps"


class _PMMAR(MarkedSmilesPolymer):
    smiles = "COC(=O)[C@](C)(C[*:1])[*:2]"
    default_name = "pmma"


class _PMMAS(MarkedSmilesPolymer):
    smiles = "COC(=O)[C@@](C)(C[*:1])[*:2]"
    default_name = "pmma"


TACTICITY_SEQUENCES = {"isotactic": "A", "syndiotactic": "AB", "atactic": None}


class _TacticCoPolymer(CoPolymer):
    """Two enantiomeric repeat units combined into a chain of set tacticity."""

    monomer_R = None
    monomer_S = None
    reference_density = None
    default_name = "polymer"

    def _build(self, length, sequence):
        # mBuild's recipe wants exactly the monomers that appear in the
        # sequence, so an isotactic ("A") or a short random sequence that
        # happens to use one hand must add only that monomer.
        chain = mb.lib.recipes.Polymer()
        for label, monomer in (
            ("A", self.monomer_A),
            ("B", self.monomer_B),
        ):
            if label in sequence:
                chain.add_monomer(
                    monomer.monomer,
                    indices=monomer.bond_indices,
                    orientation=monomer.bond_orientation,
                    separation=monomer.bond_length,
                )
        chain.build(n=length, sequence=sequence)
        return chain

    def _generate(self):
        # `lengths` counts repeat units. Build each chain from an explicit
        # sequence of that many letters, so a syndiotactic chain is not
        # `length` copies of "AB".
        rng = random.Random(self.seed)
        for idx, length in enumerate(self.lengths):
            for _ in range(self.n_mols[idx]):
                if self.tacticity == "atactic":
                    sequence = "".join(rng.choice("AB") for _ in range(length))
                elif self.tacticity == "isotactic":
                    sequence = "A" * length
                else:
                    sequence = ("AB" * length)[:length]
                self._A_count += sequence.count("A")
                self._B_count += sequence.count("B")
                mol = self._build(length=1, sequence=sequence)
                mol.name = f"{self.name}_{length}mer_{sequence}"
                self._molecules.append(mol)

    def __init__(
        self,
        lengths,
        num_mols,
        tacticity="atactic",
        seed=24,
        name=None,
        **kwargs,
    ):
        if tacticity not in TACTICITY_SEQUENCES:
            raise ValueError(
                f"tacticity must be one of {sorted(TACTICITY_SEQUENCES)}."
            )
        self.tacticity = tacticity
        super(_TacticCoPolymer, self).__init__(
            monomer_A=self.monomer_R,
            monomer_B=self.monomer_S,
            lengths=lengths,
            num_mols=num_mols,
            name=name or self.default_name,
            sequence=TACTICITY_SEQUENCES[tacticity],
            AB_ratio=0.5,
            seed=seed,
            **kwargs,
        )


class PolyStyrene(_TacticCoPolymer):
    """Polystyrene with chosen tacticity. Amorphous density about 1.04 g/cm**3.

    The two enantiomeric repeat units are ``c1ccc([C@H](C[*:2])[*:1])cc1``
    and its mirror image; ``tacticity`` selects the sequence: ``"atactic"``
    (random, seeded), ``"isotactic"`` (all one hand) or ``"syndiotactic"``
    (alternating). Each backbone CH is a stereocenter that the all-atom
    DPD stereochemistry guard records and protects.

    Parameters
    ----------
    lengths : int or list, required
    num_mols : int or list, required
    tacticity : {"atactic", "isotactic", "syndiotactic"}, default "atactic"
    seed : int, default 24
        Seed for the random atactic sequence.
    name : str, default "ps"

    """

    monomer_R = _PolystyreneR
    monomer_S = _PolystyreneS
    reference_density = 1.04
    default_name = "ps"


class PMMA(_TacticCoPolymer):
    """Poly(methyl methacrylate) with chosen tacticity. Density about 1.18 g/cm**3.

    Repeat units ``COC(=O)[C@](C)(C[*:1])[*:2]`` and its mirror image; see
    `PolyStyrene` for the tacticity options. The quaternary backbone carbon
    is the stereocenter.

    Parameters
    ----------
    lengths : int or list, required
    num_mols : int or list, required
    tacticity : {"atactic", "isotactic", "syndiotactic"}, default "atactic"
    seed : int, default 24
    name : str, default "pmma"

    """

    monomer_R = _PMMAR
    monomer_S = _PMMAS
    reference_density = 1.18
    default_name = "pmma"


class PEAAIonomer(Molecule):
    """Precise polyethylene-acrylate ionomer chains in the carboxylate form.

    Each chain is ``lengths`` copies of ``pattern``, a string of
    three-carbon backbone blocks: ``"E"`` is ``-CH2-CH2-CH2-`` and ``"A"`` is
    ``-CH2-CH(CH2COO-)-CH2-``. For example ``pattern="EEAEE"`` places one
    carboxylate every 15 backbone carbons. Chain ends are capped with
    hydrogens. Every acid block's CH is a stereocenter; ``tacticity``
    chooses its handedness along the chain, and the all-atom DPD
    stereochemistry guard protects it. Use `counterions` for the
    neutralizing ions.

    Parameters
    ----------
    lengths : int or list, required
        Copies of ``pattern`` per chain.
    num_mols : int or list, required
        Chains per length.
    pattern : str, default "EEAEE"
        Block sequence of one pattern repeat, from ``"E"`` and ``"A"``.
    tacticity : {"atactic", "isotactic"}, default "atactic"
        ``"atactic"`` draws each acid block's handedness from a seeded
        random sequence.
    seed : int, default 24
        Seed for the atactic sequence.
    name : str, default "peaa"

    """

    ethylene_smiles = "[*:1]CCC[*:2]"
    acid_smiles_R = "[*:1]C[C@H](CC(=O)[O-])C[*:2]"
    acid_smiles_S = "[*:1]C[C@@H](CC(=O)[O-])C[*:2]"
    bond_length = 0.154
    default_name = "peaa"

    def __init__(
        self,
        lengths,
        num_mols,
        pattern="EEAEE",
        tacticity="atactic",
        seed=24,
        name=None,
        **kwargs,
    ):
        if not pattern or set(pattern) - {"E", "A"}:
            raise ValueError("pattern must be a string of 'E' and 'A'.")
        if tacticity not in ("atactic", "isotactic"):
            raise ValueError("tacticity must be 'atactic' or 'isotactic'.")
        self.lengths = check_return_iterable(lengths)
        num_mols = check_return_iterable(num_mols)
        if len(num_mols) != len(self.lengths):
            raise ValueError("Number of molecules and lengths must be equal.")
        self.pattern = pattern
        self.tacticity = tacticity
        self.seed = seed
        self.sequences = []
        self._blocks = {
            "E": monomer_from_marked_smiles(self.ethylene_smiles, name="E"),
            "R": monomer_from_marked_smiles(self.acid_smiles_R, name="A"),
            "S": monomer_from_marked_smiles(self.acid_smiles_S, name="A"),
        }
        super(PEAAIonomer, self).__init__(
            num_mols=num_mols, name=name or self.default_name, **kwargs
        )

    @property
    def n_carboxylates(self):
        """Carboxylate groups over all chains."""
        return sum(
            sequence.count("R") + sequence.count("S")
            for sequence in self.sequences
        )

    def counterions(self, smiles="[Na+]", name="counterion"):
        """Return a `flowermd.base.Molecule` of neutralizing counterions.

        Parameters
        ----------
        smiles : str, default "[Na+]"
            SMILES of a monatomic cation such as ``"[Na+]"`` or ``"[Zn+2]"``.
        name : str, default "counterion"

        """
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() != 1:
            raise ValueError("counterions needs a monatomic ion SMILES.")
        charge = mol.GetAtomWithIdx(0).GetFormalCharge()
        if charge <= 0 or self.n_carboxylates % charge:
            raise ValueError(
                f"{smiles} cannot neutralize {self.n_carboxylates} "
                "carboxylates."
            )
        return Molecule(
            num_mols=self.n_carboxylates // charge, smiles=smiles, name=name
        )

    def _load(self):
        return None

    def _build(self, sequence):
        # mBuild's recipe wants exactly the monomers used in the sequence
        # and names them A, B, C in the order they were added.
        chain = mb.lib.recipes.Polymer()
        used = [label for label in ("E", "R", "S") if label in sequence]
        for label in used:
            monomer, indices = self._blocks[label]
            chain.add_monomer(
                mb.clone(monomer), indices=indices, separation=self.bond_length
            )
        letters = {label: "ABC"[i] for i, label in enumerate(used)}
        chain.build(n=1, sequence="".join(letters[s] for s in sequence))
        return chain

    def _generate(self):
        rng = random.Random(self.seed)
        for idx, length in enumerate(self.lengths):
            for _ in range(self.n_mols[idx]):
                sequence = ""
                for block in self.pattern * length:
                    if block == "E":
                        sequence += "E"
                    elif self.tacticity == "isotactic":
                        sequence += "R"
                    else:
                        sequence += rng.choice("RS")
                self.sequences.append(sequence)
                mol = self._build(sequence)
                mol.name = f"{self.name}_{len(sequence)}block"
                self._molecules.append(mol)


class LadderPolymer(Molecule):
    """A polymer whose repeat units join through two bonds (a ladder polymer).

    mBuild's `Polymer` recipe joins repeats through one bond, so ladder
    polymers such as PIM-1 are assembled here directly: the repeat unit is
    embedded once from a SMILES with four marked attachment points, cloned
    per repeat, and each clone is placed by a rigid fit that puts its two
    inbound attachment atoms where the previous repeat's outbound
    placeholders sit (and the reverse), which gives both junction bonds
    near their built length. Chain ends are capped with hydrogens. This is
    a purpose-built assembler for the PhantomWalk study; see the class
    docstring of `PIM1`.

    Subclasses set ``smiles`` with ``[*:1]``/``[*:2]`` on the outbound atoms
    and ``[*:3]``/``[*:4]`` on the inbound atoms (1 pairs with 3, 2 with 4).

    Parameters
    ----------
    lengths : int or list, required
        Repeat units per chain.
    num_mols : int or list, required
        Chains per length.
    seed : int, default 0
        RDKit embedding seed for the repeat unit.
    name : str, optional

    The class attribute ``junction_length`` (nm) is the target for the two
    junction bonds. Because the two attachment sites on each side are a
    rigid pair, the four-point fit is a compromise and the junction bonds
    come out shorter than the target (about 0.09 nm for PIM-1); the DPD
    stage, with bonded terms scaled by 30, pulls them to the force-field
    length within the first steps, like the stretched repeat junctions of
    the lattice placement.

    """

    smiles = None
    reference_density = None
    default_name = "ladder"
    junction_length = 0.136  # nm, aromatic C-O for PIM-1's dioxane links

    def __init__(self, lengths, num_mols, seed=0, name=None, **kwargs):
        if self.smiles is None:
            raise NotImplementedError("Subclasses must define `smiles`.")
        self.lengths = check_return_iterable(lengths)
        num_mols = check_return_iterable(num_mols)
        if len(num_mols) != len(self.lengths):
            raise ValueError("Number of molecules and lengths must be equal.")
        self.seed = seed
        self._repeat, self._ports = ladder_monomer_from_marked_smiles(
            self.smiles, seed=seed, name=name or self.default_name
        )
        super(LadderPolymer, self).__init__(
            num_mols=num_mols, name=name or self.default_name, **kwargs
        )

    def _load(self):
        return None

    def _build(self, length):
        template = self._repeat
        t_parts = list(template.particles())
        t_xyz = np.asarray(template.xyz, dtype=float)
        out_ports = [self._ports[1], self._ports[2]]
        in_ports = [self._ports[3], self._ports[4]]
        anchor = {
            k: t_parts.index(next(iter(t_parts[k].direct_bonds())))
            for k in out_ports + in_ports
        }
        chain = mb.Compound(name=f"{self.name}_{length}mer")
        repeats = []
        for k in range(length):
            repeat = mb.clone(template)
            repeat.name = self.name
            xyz = t_xyz.copy()
            if k:
                previous = np.asarray(repeats[-1].xyz, dtype=float)
                # Fit this repeat's (inbound anchor, inbound placeholder) pairs
                # onto the previous repeat's (outbound placeholder, outbound
                # anchor) pairs: the anchors land where the placeholders were.
                # Placeholders sit at a C-H length; extend them to the
                # junction bond length before fitting so the new bonds
                # come out at that length.
                L = self.junction_length

                def extend(points, anchor_i, port_i):
                    d = points[port_i] - points[anchor_i]
                    return points[anchor_i] + d / np.linalg.norm(d) * L

                source = np.array(
                    [
                        xyz[anchor[in_ports[0]]],
                        extend(xyz, anchor[in_ports[0]], in_ports[0]),
                        xyz[anchor[in_ports[1]]],
                        extend(xyz, anchor[in_ports[1]], in_ports[1]),
                    ]
                )
                target = np.array(
                    [
                        extend(previous, anchor[out_ports[0]], out_ports[0]),
                        previous[anchor[out_ports[0]]],
                        extend(previous, anchor[out_ports[1]], out_ports[1]),
                        previous[anchor[out_ports[1]]],
                    ]
                )
                rotation, translation = _kabsch(source, target)
                xyz = xyz @ rotation.T + translation
            repeat.xyz = xyz
            chain.add(repeat)
            repeats.append(repeat)
        # junction bonds, then remove the placeholders they replace
        to_remove = []
        for k in range(length - 1):
            a_parts = list(repeats[k].particles())
            b_parts = list(repeats[k + 1].particles())
            for out_port, in_port in zip(out_ports, in_ports):
                chain.add_bond(
                    (a_parts[anchor[out_port]], b_parts[anchor[in_port]])
                )
                to_remove += [a_parts[out_port], b_parts[in_port]]
        chain.remove(to_remove)
        return chain

    def _generate(self):
        for idx, length in enumerate(self.lengths):
            for _ in range(self.n_mols[idx]):
                self._molecules.append(self._build(length))


def _kabsch(source, target):
    """Proper rigid transform (R, t) minimizing |R source + t - target|."""
    s_center = source.mean(axis=0)
    t_center = target.mean(axis=0)
    h = (source - s_center).T @ (target - t_center)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return rotation, t_center - s_center @ rotation.T


class PIM1(LadderPolymer):
    """PIM-1, the archetypal polymer of intrinsic microporosity. Bulk density about 1.06 g/cm**3.

    A ladder polymer: each spirobisindane-dioxane repeat joins the next
    through two C-O bonds, so it cannot be built with the one-bond
    `flowermd.base.Polymer` recipe and uses `LadderPolymer` instead. The
    repeat SMILES is the Abbott, Hart and Colina 2013 structure with
    ``[*:1]``/``[*:2]`` on the dinitrile-ring carbons and ``[*:3]``/``[*:4]``
    on the catechol oxygens. Included for the PhantomWalk benchmark set;
    the assembler is intentionally specific to two-bond junctions.

    """

    smiles = (
        "CC1(C)CC2(CC(C)(C)c3cc(O[*:4])c(O[*:3])cc32)c2cc3c(cc21)"
        "Oc1c(C#N)c([*:1])c([*:2])c(C#N)c1O3"
    )
    reference_density = 1.06
    default_name = "pim1"
