import numpy as np
import pytest
import unyt as u

from flowermd import Simulation
from flowermd.internal.placement import (
    junction_atoms,
    repeat_units,
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
        intra_before = [
            _bond_lengths(c, within_children_only=True)
            for c in chains.molecules
        ]
        system = AllAtomRandomWalk(molecules=chains, density=DENSITY, seed=7)
        box = system.system.box
        expected = get_target_box_mass_density(
            density=DENSITY, mass=system.mass
        ).to("nm")
        assert np.allclose(box.lengths, expected.value)
        assert system.system.n_particles == chains.n_particles
        for chain, before in zip(system.system.children, intra_before):
            assert np.allclose(
                _bond_lengths(chain, within_children_only=True), before
            )
            repeats = repeat_units(chain)
            junctions = junction_atoms(chain, repeats)
            for inbound, outbound in junctions[1:-1]:
                assert inbound is not None and outbound is not None
            # junction bonds stretched or compressed, but not absurdly
            allb = _bond_lengths(chain)
            assert allb.max() < 0.6 and allb.min() > 0.05

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

    def test_random_walk_max_turn(self):
        with pytest.raises(ValueError):
            AllAtomRandomWalk(
                PolyEthylene(lengths=4, num_mols=1), DENSITY, max_turn=0
            )
        system = AllAtomRandomWalk(
            PolyEthylene(lengths=8, num_mols=1), DENSITY, max_turn=np.pi / 6
        )
        chain = system.system.children[0]
        centers = np.array([r.xyz.mean(axis=0) for r in repeat_units(chain)])
        steps = np.diff(centers, axis=0)
        cosines = [
            np.dot(steps[i], steps[i + 1])
            / np.linalg.norm(steps[i])
            / np.linalg.norm(steps[i + 1])
            for i in range(len(steps) - 1)
        ]
        assert min(cosines) >= np.cos(np.pi / 6) - 1e-6

    def test_lattice_repeat_units(self):
        chains = PolyEthylene(lengths=5, num_mols=3)
        intra_before = [
            _bond_lengths(c, within_children_only=True)
            for c in chains.molecules
        ]
        system = AllAtomLattice(molecules=chains, density=DENSITY, seed=3)
        assert np.prod(system.grid_shape) >= 15
        assert system.system.n_particles == chains.n_particles
        for chain, before in zip(system.system.children, intra_before):
            assert np.allclose(
                _bond_lengths(chain, within_children_only=True), before
            )

    def test_lattice_whole_chains_keep_all_bonds(self):
        chains = PolyEthylene(lengths=5, num_mols=3)
        before = [_bond_lengths(c) for c in chains.molecules]
        system = AllAtomLattice(
            molecules=chains, density=DENSITY, unit="chain", seed=3
        )
        assert np.prod(system.grid_shape) >= 3
        for chain, b in zip(system.system.children, before):
            assert np.allclose(_bond_lengths(chain), b)

    def test_lattice_bad_unit(self):
        with pytest.raises(ValueError):
            AllAtomLattice(
                PolyEthylene(lengths=2, num_mols=1), DENSITY, unit="atom"
            )

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
