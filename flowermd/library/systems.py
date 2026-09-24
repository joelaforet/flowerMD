"""Examples for the Systems class."""

import warnings

import mbuild as mb
import numpy as np
import unyt as u
from scipy.spatial.distance import pdist

from flowermd.base.system import System
from flowermd.internal.placement import (
    random_rotation,
    random_walk_conformation,
    repeat_units,
    serpentine_lattice_sites,
    target_box_lengths,
)


class SingleChainSystem(System):
    """Builds a vacuum box around a single chain.

    The box lengths are chosen so they are at least as long as the largest particle distance.
    The maximum distance of the chain is calculated using scipy.spatial.distance.pdist().
    This distance multiplied by a buffer defines the box dimensions. The chain is centered in the box.

    Parameters
    ----------
    buffer : float, default 1.05
        A factor to multiply box dimensions. Must be greater than 1 so that the particles are inside the box.

    """

    def __init__(self, molecules, base_units=dict(), buffer=1.05):
        self.buffer = buffer
        super(SingleChainSystem, self).__init__(
            molecules=molecules, base_units=base_units
        )

    def _build_system(self):
        if len(self.all_molecules) > 1:
            raise ValueError(
                "This system class only works for systems contianing a single molecule."
            )
        chain = self.all_molecules[0]
        eucl_dist = pdist(self.all_molecules[0].xyz)
        chain_length = np.max(eucl_dist)
        box = mb.Box(lengths=np.array([chain_length] * 3) * self.buffer)
        comp = mb.Compound()
        comp.add(chain)
        comp.box = box
        chain.translate_to((box.Lx / 2, box.Ly / 2, box.Lz / 2))
        return comp


class mbuildSystem(System):
    """Builds a system using mbuild box and mbuild positions.

    The box lengths and positions are read from the input mbuild compound. This is intended to be used with mbuild intialization methods,
    like translating polymer contiuents within the box, or a random walk cuboid constraint in mbuild 2.0.

    """

    def __init__(self, molecules, base_units=dict()):
        self.box_temp = molecules.box
        super(mbuildSystem, self).__init__(
            molecules=molecules, base_units=base_units
        )

    def _build_system(self):
        chain = self.all_molecules
        comp = mb.Compound()
        comp.add(chain)
        comp.box = self.box_temp
        return comp


def _as_density(density):
    if isinstance(density, u.array.unyt_quantity):
        return density
    warnings.warn(
        "Units for density were not given, assuming units of g/cm**3."
    )
    return density * u.Unit("g") / u.Unit("cm**3")


class AllAtomRandomWalk(System):
    """Place all-atom chains as random coils that keep their stereochemistry.

    The box is sized for the target density, so the chains overlap heavily;
    this is the starting point the all-atom PhantomWalk DPD stage is designed
    to relax. Each chain starts from its built geometry, is given a uniformly
    random orientation and a random position in the box, and then every bond
    between two repeat units (the children of the chain built by
    `flowermd.base.Polymer`) is turned to a uniformly random torsion. Only
    those torsions change: every bond length, bond angle and stereocenter of
    the built chain is kept, including centers whose substituents span two
    repeats (polystyrene, PMMA). A junction that is part of a ring, such as
    the two-bond junction of a ladder polymer, keeps its built geometry.
    There is no self-avoidance, within or between chains.

    Parameters
    ----------
    molecules : flowermd.base.Molecule or list, required
        Chains to place.
    density : float or unyt.unyt_quantity, required
        Target mass density (g/cm**3 assumed if unitless) or number density.
    seed : int, default 1234
        Seed for the orientations, start positions and torsions.
    base_units : dict, default {}

    Attributes
    ----------
    target_box : numpy.ndarray
        Box lengths in nm.

    """

    def __init__(self, molecules, density, seed=1234, base_units=dict()):
        self.density = _as_density(density)
        self.seed = seed
        super(AllAtomRandomWalk, self).__init__(
            molecules=molecules, base_units=base_units
        )

    def _build_system(self):
        self.target_box = target_box_lengths(
            self.density, self.mass, self.n_particles
        )
        rng = np.random.default_rng(self.seed)
        for chain in self.all_molecules:
            particles = list(chain.particles())
            index = {p: i for i, p in enumerate(particles)}
            units = [
                [index[p] for p in repeat.particles()]
                for repeat in repeat_units(chain)
            ]
            bonds = [(index[a], index[b]) for a, b in chain.bonds()]
            start = rng.uniform(0.0, self.target_box)
            chain.xyz = random_walk_conformation(
                chain.xyz, units, bonds, start, rng
            )
        compound = mb.Compound()
        compound.add(self.all_molecules)
        compound.box = mb.Box(lengths=self.target_box)
        return compound


class AllAtomLattice(System):
    """Place whole all-atom chains on a lattice.

    The box is sized for the target density and divided into a near-cubic
    grid with one site per chain. Each chain keeps its built conformation,
    is centered on its site with a uniformly random rotation, and the whole
    lattice gets one random shift. Chains move as rigid bodies, so every
    bond, angle, torsion and stereocenter of the built chain is kept.

    Parameters
    ----------
    molecules : flowermd.base.Molecule or list, required
        Chains to place.
    density : float or unyt.unyt_quantity, required
        Target mass density (g/cm**3 assumed if unitless) or number density.
    seed : int, default 1234
        Seed for the rotations and the global shift.
    base_units : dict, default {}

    Attributes
    ----------
    target_box : numpy.ndarray
        Box lengths in nm.
    grid_shape : numpy.ndarray
        Number of sites along x, y and z.

    """

    def __init__(self, molecules, density, seed=1234, base_units=dict()):
        self.density = _as_density(density)
        self.seed = seed
        super(AllAtomLattice, self).__init__(
            molecules=molecules, base_units=base_units
        )

    def _build_system(self):
        self.target_box = target_box_lengths(
            self.density, self.mass, self.n_particles
        )
        units = list(self.all_molecules)
        sites, self.grid_shape = serpentine_lattice_sites(
            len(units), self.target_box
        )
        rng = np.random.default_rng(self.seed)
        shift = rng.uniform(0.0, self.target_box)
        for unit, site in zip(units, sites):
            xyz = np.asarray(unit.xyz, dtype=float)
            center = xyz.mean(axis=0)
            unit.xyz = (xyz - center) @ random_rotation(rng).T + site + shift
        compound = mb.Compound()
        compound.add(self.all_molecules)
        compound.box = mb.Box(lengths=self.target_box)
        return compound
