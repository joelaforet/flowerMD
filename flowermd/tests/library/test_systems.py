import mbuild as mb
import numpy as np
import pytest

from flowermd.library import (
    OPLS_AA,
    EllipsoidChainRand,
    LJChain,
    PolyEthylene,
    SingleChainSystem,
    mbuildSystem,
)


class TestSystems:
    def test_mbuild_all_atom_forcefield(self):
        compound = mb.Compound()
        for position in ([0.5, 0.5, 0.5], [1.5, 1.5, 1.5]):
            molecule = mb.load("CC", smiles=True)
            molecule.translate_to(position)
            compound.add(molecule)
        compound.box = mb.Box(lengths=[3.0, 4.0, 5.0])
        particles = list(compound.particles())
        positions = compound.xyz.copy()
        box_lengths = np.array(compound.box.lengths)
        bonds = {
            tuple(sorted((particles.index(a), particles.index(b))))
            for a, b in compound.bonds()
        }
        atomic_numbers = [
            particle.element.atomic_number for particle in particles
        ]
        assert atomic_numbers.count(1) == 12

        system = mbuildSystem(molecules=compound)
        system.apply_forcefield(r_cut=1.0, force_field=OPLS_AA())

        assert system.n_mol_types == 1
        assert system.gmso_system.is_typed()
        assert list(system.system.particles()) == particles
        np.testing.assert_allclose(system.system.xyz, positions)
        np.testing.assert_allclose(system.box.lengths, box_lengths)
        sites = list(system.gmso_system.sites)
        assert all(site.group == "0" for site in sites)
        assert [site.element.atomic_number for site in sites] == atomic_numbers
        np.testing.assert_allclose(
            system.gmso_system.positions.to_value("nm"), positions
        )
        assert {
            tuple(sorted(sites.index(site) for site in bond.connection_members))
            for bond in system.gmso_system.bonds
        } == bonds
        snapshot = system.hoomd_snapshot
        snapshot.validate()
        assert snapshot.particles.N == len(particles)
        assert snapshot.bonds.N == len(bonds)
        assert {tuple(sorted(bond)) for bond in snapshot.bonds.group} == bonds
        assert set(snapshot.particles.types) == {"opls_135", "opls_140"}
        np.testing.assert_allclose(snapshot.configuration.box[:3], box_lengths)
        np.testing.assert_allclose(
            snapshot.particles.position, positions - box_lengths / 2, atol=1e-6
        )

    def test_single_chain_buffer(self):
        chain = PolyEthylene(lengths=10, num_mols=1)
        system = SingleChainSystem(molecules=chain, buffer=1.05)
        chain2 = PolyEthylene(lengths=10, num_mols=1)
        system2 = SingleChainSystem(molecules=chain2, buffer=2.10)

        assert np.allclose(system.box.Lx * 2, system2.box.Lx, atol=1e-3)
        assert np.allclose(system.box.Ly * 2, system2.box.Ly, atol=1e-3)
        assert np.allclose(system.box.Lz * 2, system2.box.Lz, atol=1e-3)

    def test_single_chain_lengths(self):
        lj_chain = LJChain(lengths=10, num_mols=1)
        system = SingleChainSystem(molecules=lj_chain, buffer=1.05)
        chain = lj_chain._molecules[0]
        chain_length = np.linalg.norm(
            chain.children[-1].pos - chain.children[0].pos
        )
        assert np.allclose(chain_length * 1.05, system.box.Lx, atol=1e-3)

    def test_multiple_molecules(self):
        lj_chain = LJChain(lengths=10, num_mols=2)
        with pytest.raises(ValueError):
            SingleChainSystem(molecules=lj_chain, buffer=1.05)

    def test_rand_walk(self):
        chains = EllipsoidChainRand(
            lengths=10, num_mols=10, lpar=0.5, bead_mass=1.0, density=0.85
        )
        system = mbuildSystem(molecules=chains)
        assert system.box
        assert system.n_particles == 400  # 10 x 10 x 4
