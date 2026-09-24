import numpy as np
import pytest
import unyt as u

from flowermd import Simulation
from flowermd.internal.placement import (
    random_walk_conformation,
    serpentine_lattice_sites,
)
from flowermd.library import (
    AllAtomDPD,
    AllAtomLattice,
    AllAtomRandomWalk,
    PolyEthylene,
)
from flowermd.tests import BaseTest
from flowermd.utils import get_target_box_mass_density

DENSITY = 0.5 * u.g / u.cm**3


def _bond_lengths(compound, within_children_only=False):
    """Sorted bond lengths (nm); optionally only bonds inside one child."""
    owner = {}
    for child in compound.children:
        for particle in child.particles():
            owner[particle] = child
    lengths = []
    for a, b in compound.bonds():
        if within_children_only and owner.get(a) is not owner.get(b):
            continue
        lengths.append(np.linalg.norm(np.asarray(a.pos) - np.asarray(b.pos)))
    return np.sort(lengths)


def _geometry(compound):
    """Bond lengths, bond angles and tetrahedral signed volumes.

    Returned in a fixed order so two conformations of the same molecule can
    be compared entry by entry. The signed volume of an atom with four
    neighbours is the determinant of its neighbour vectors; its sign is the
    handedness of that center.
    """
    particles = list(compound.particles())
    index = {p: i for i, p in enumerate(particles)}
    xyz = np.asarray(compound.xyz, dtype=float)
    neighbors = [[] for _ in particles]
    for a, b in compound.bonds():
        neighbors[index[a]].append(index[b])
        neighbors[index[b]].append(index[a])
    bonds = np.array(
        [
            np.linalg.norm(xyz[i] - xyz[j])
            for i in range(len(xyz))
            for j in sorted(neighbors[i])
            if j > i
        ]
    )
    angles, volumes = [], []
    for center, around in enumerate(neighbors):
        around = sorted(around)
        vectors = xyz[around] - xyz[center]
        for x in range(len(around)):
            for y in range(x + 1, len(around)):
                cosine = np.dot(vectors[x], vectors[y]) / (
                    np.linalg.norm(vectors[x]) * np.linalg.norm(vectors[y])
                )
                angles.append(cosine)
        if len(around) == 4:
            volumes.append(np.linalg.det(vectors[:3] - vectors[3]))
    return bonds, np.array(angles), np.array(volumes)


class TestAllAtomPlacement(BaseTest):
    def test_lattice_sites_cover_count(self):
        sites, dims = serpentine_lattice_sites(30, np.array([3.0, 3.0, 3.0]))
        assert sites.shape == (30, 3)
        assert np.prod(dims) >= 30
        assert np.all(sites > 0) and np.all(sites < 3.0)
        # consecutive sites are neighbours: one grid index changes by one
        cells = sites / (np.array([3.0, 3.0, 3.0]) / dims) - 0.5
        moves = np.abs(np.diff(np.rint(cells), axis=0)).sum(axis=1)
        assert np.all(moves == 1)

    def test_random_walk_box_and_geometry(self):
        chains = PolyEthylene(lengths=6, num_mols=4)
        before = [_geometry(c) for c in chains.molecules]
        system = AllAtomRandomWalk(molecules=chains, density=DENSITY, seed=7)
        box = system.system.box
        expected = get_target_box_mass_density(
            density=DENSITY, mass=system.mass
        ).to("nm")
        assert np.allclose(box.lengths, expected.value)
        assert system.system.n_particles == chains.n_particles
        for chain, (bonds, angles, volumes) in zip(
            system.system.children, before
        ):
            new_bonds, new_angles, new_volumes = _geometry(chain)
            # only torsions change: bonds, angles and handedness are kept
            assert np.allclose(new_bonds, bonds, atol=1e-6)
            assert np.allclose(new_angles, angles, atol=1e-6)
            assert np.all(np.sign(new_volumes) == np.sign(volumes))
            assert np.allclose(np.abs(new_volumes), np.abs(volumes))

    def test_random_walk_changes_conformation(self):
        chains = PolyEthylene(lengths=12, num_mols=1)
        built = np.asarray(chains.molecules[0].xyz, dtype=float)
        system = AllAtomRandomWalk(molecules=chains, density=DENSITY, seed=4)
        placed = np.asarray(system.system.children[0].xyz, dtype=float)

        def end_to_end(xyz):
            return np.linalg.norm(xyz[-1] - xyz[0])

        # the built chain is extended; random torsions coil it
        assert end_to_end(placed) < end_to_end(built)

    def test_random_walk_is_seeded(self):
        a = AllAtomRandomWalk(
            PolyEthylene(lengths=4, num_mols=2), DENSITY, seed=1
        )
        b = AllAtomRandomWalk(
            PolyEthylene(lengths=4, num_mols=2), DENSITY, seed=1
        )
        c = AllAtomRandomWalk(
            PolyEthylene(lengths=4, num_mols=2), DENSITY, seed=2
        )
        assert np.allclose(a.system.xyz, b.system.xyz)
        assert not np.allclose(a.system.xyz, c.system.xyz)

    def test_ring_junction_keeps_its_geometry(self):
        # two units joined by two bonds, as in a ladder polymer: the
        # junction is part of a ring and has no free torsion
        xyz = np.array(
            [[0.0, 0, 0], [0.15, 0, 0], [0.0, 0.15, 0], [0.15, 0.15, 0.05]]
        )
        units = [[0, 1], [2, 3]]
        bonds = np.array([[0, 1], [2, 3], [0, 2], [1, 3]])
        placed = random_walk_conformation(
            xyz, units, bonds, np.zeros(3), np.random.default_rng(0)
        )

        def distances(points):
            return np.linalg.norm(points[:, None] - points[None], axis=-1)

        assert np.allclose(distances(placed), distances(xyz))

    def test_lattice_whole_chains_keep_all_bonds(self):
        chains = PolyEthylene(lengths=5, num_mols=3)
        before = [_bond_lengths(c) for c in chains.molecules]
        system = AllAtomLattice(molecules=chains, density=DENSITY, seed=3)
        assert np.prod(system.grid_shape) >= 3
        for chain, b in zip(system.system.children, before):
            assert np.allclose(_bond_lengths(chain), b)

    def test_polydisperse_and_unitless_density(self):
        chains = PolyEthylene(lengths=[3, 5], num_mols=[2, 2])
        with pytest.warns(UserWarning, match="g/cm"):
            system = AllAtomLattice(molecules=chains, density=0.5)
        assert system.n_molecules == 4
        assert system.system.n_particles == chains.n_particles

    @pytest.mark.parametrize("system_cls", [AllAtomRandomWalk, AllAtomLattice])
    def test_end_to_end_dpd(self, system_cls):
        pytest.importorskip("rdkit")
        chains = PolyEthylene(lengths=4, num_mols=3)
        system = system_cls(molecules=chains, density=DENSITY, seed=11)
        ff = AllAtomDPD(system.system)
        sim = Simulation(
            initial_state=ff.frame, forcefield=ff.hoomd_forces, dt=0.001
        )
        result = sim.run_DPD(n_steps=40, chunk=20, stop=lambda s: False)
        assert result["steps"] == 40
        assert all(np.isfinite(f.energy) for f in sim.forces)
