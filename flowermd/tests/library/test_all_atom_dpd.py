import hoomd
import numpy as np
import pytest
import unyt as u

from flowermd import Simulation
from flowermd.base import Pack
from flowermd.library import AllAtomDPD, PolyEthylene
from flowermd.tests import BaseTest

pytest.importorskip("rdkit")


def _pe_melt(lengths=3, num_mols=2, density=0.2):
    """Small all-atom polyethylene melt with a periodic box (nm)."""
    chains = PolyEthylene(lengths=lengths, num_mols=num_mols)
    system = Pack(molecules=chains, density=density * u.g / u.cm**3)
    compound = system.system
    if compound.box is None:
        compound.box = system.box
    return compound


class TestAllAtomDPD(BaseTest):
    def test_uff_types_match_frame(self):
        compound = _pe_melt()
        ff = AllAtomDPD(compound)
        frame = ff.frame
        assert frame.particles.N == compound.n_particles
        assert frame.bonds.N == compound.n_bonds
        assert frame.angles.N > 0 and frame.dihedrals.N > 0
        assert frame.impropers.N == 0  # UFF source has no impropers
        assert set(ff.forces_by_role) == {"bond", "angle", "dihedral", "pair"}
        assert ff.hoomd_forces[-1] is ff.forces_by_role["pair"]
        for role, block in (
            ("bond", frame.bonds),
            ("angle", frame.angles),
            ("dihedral", frame.dihedrals),
        ):
            assert set(ff.forces_by_role[role].params.keys()) == set(
                block.types
            )
        assert ff.parameters.source == "uff"
        assert ff.openff_topology is None
        # positions wrapped into the box, masses physical
        half = np.asarray(frame.configuration.box[:3]) / 2
        assert np.all(np.abs(frame.particles.position) <= half + 1e-9)
        assert 0.9 < frame.particles.mass.min() < 1.2  # hydrogen, amu
        assert 11.9 < frame.particles.mass.max() < 12.2  # carbon, amu

    def test_frame_images_recover_input_positions(self):
        compound = _pe_melt()
        # push one chain across the periodic boundary
        chain = list(compound.children)[0]
        chain.translate(np.asarray(compound.box.lengths) * 0.9)
        ff = AllAtomDPD(compound)
        frame = ff.frame
        box = np.asarray(frame.configuration.box[:3])
        assert np.all(np.abs(frame.particles.position) <= box / 2 + 1e-9)
        unwrapped = frame.particles.position + frame.particles.image * box
        assert np.allclose(unwrapped, np.asarray(compound.xyz) * 10.0)

    def test_bonded_scale_and_epsilon_weighting(self):
        compound = _pe_melt()
        ff = AllAtomDPD(compound, A=100.0, gamma=10.0, bonded_scale=7.0)
        raw_k = ff.parameters.bond_params[ff.frame.bonds.types[0]]["k"]
        assert ff.forces_by_role["bond"].params[ff.frame.bonds.types[0]][
            "k"
        ] == pytest.approx(7.0 * raw_k)
        eps = ff.parameters.particle_epsilons
        ref = max(eps.values())
        pair = ff.forces_by_role["pair"]
        types = list(eps)
        for a in types:
            for b in types:
                key = (a, b) if (a, b) in pair.params else (b, a)
                weight = np.sqrt(eps[a] * eps[b]) / ref
                assert pair.params[key]["A"] == pytest.approx(100.0 * weight)
                assert pair.params[key]["gamma"] == pytest.approx(10.0 * weight)
        # the strongest pair is unweighted
        strongest = max(eps, key=eps.get)
        assert pair.params[(strongest, strongest)]["A"] == pytest.approx(100.0)

    def test_uniform_pair_when_weighting_off(self):
        ff = AllAtomDPD(_pe_melt(), A=50.0, gamma=5.0, epsilon_weighting=False)
        for values in ff.forces_by_role["pair"].params.values():
            assert values["A"] == pytest.approx(50.0)
            assert values["gamma"] == pytest.approx(5.0)

    def test_ablation_flags(self):
        compound = _pe_melt()
        ff = AllAtomDPD(compound, include_angles=False, include_dihedrals=False)
        assert set(ff.forces_by_role) == {"bond", "pair"}
        assert len(ff.hoomd_forces) == 2
        # topology is still in the frame so exclusions are unchanged
        assert ff.frame.angles.N > 0 and ff.frame.dihedrals.N > 0

    def test_conservative_variant(self):
        ff = AllAtomDPD(_pe_melt(), conservative=True)
        pair = ff.forces_by_role["pair"]
        assert isinstance(pair, hoomd.md.pair.DPDConservative)
        assert "gamma" not in next(iter(pair.params.values()))

    def test_bad_arguments(self):
        compound = _pe_melt()
        with pytest.raises(ValueError):
            AllAtomDPD(compound, bonded="amber")
        with pytest.raises(ValueError):
            AllAtomDPD(compound, A=0.0)
        with pytest.raises(ValueError):
            AllAtomDPD(compound, gamma=-1.0)

    def test_identical_chains_share_types(self):
        ff = AllAtomDPD(_pe_melt(lengths=3, num_mols=3))
        # three identical chains: parameters merge into one table
        assert ff.frame.bonds.N == 3 * (ff.frame.bonds.N // 3)
        assert len(ff.frame.bonds.types) == len(ff.parameters.bond_params)

    def test_runs_dpd_and_fire(self):
        ff = AllAtomDPD(_pe_melt())
        sim = Simulation(
            initial_state=ff.frame, forcefield=ff.hoomd_forces, dt=0.001
        )
        result = sim.run_DPD(n_steps=40, chunk=20, stop=lambda s: False)
        assert result["steps"] == 40
        assert all(np.isfinite(f.energy) for f in sim.forces)
        fire = sim.run_FIRE(n_steps=20, dt=0.001, force_tol=1e-6)
        assert fire["steps"] == 20
        assert all(np.isfinite(f.energy) for f in sim.forces)


class TestAllAtomDPDOpenFF(BaseTest):
    @pytest.fixture(autouse=True)
    def _need_openff(self):
        pytest.importorskip("openff.toolkit")
        pytest.importorskip("openff.interchange")

    def test_openff_source(self):
        compound = _pe_melt()
        ff = AllAtomDPD(compound, bonded="openff")
        frame = ff.frame
        assert ff.parameters.source == "openff"
        assert frame.particles.N == compound.n_particles
        assert frame.bonds.N == compound.n_bonds
        assert frame.angles.N > 0 and frame.dihedrals.N > 0
        assert ff.openff_topology is not None
        assert ff.openff_topology.n_atoms == compound.n_particles
        assert set(ff.forces_by_role) >= {"bond", "angle", "dihedral", "pair"}
        for role, block in (
            ("bond", frame.bonds),
            ("angle", frame.angles),
            ("dihedral", frame.dihedrals),
        ):
            assert set(ff.forces_by_role[role].params.keys()) == set(
                block.types
            )
        # Sage epsilons weight the pair term, not UFF epsilons
        assert all(name.startswith("openff_") for name in frame.particles.types)
        # constrained hydrogens became flexible bonds: every bond has k > 0
        assert all(
            v["k"] > 0 for v in ff.forces_by_role["bond"].params.values()
        )

    def test_openff_runs(self):
        ff = AllAtomDPD(_pe_melt(), bonded="openff")
        sim = Simulation(
            initial_state=ff.frame, forcefield=ff.hoomd_forces, dt=0.001
        )
        sim.run_DPD(n_steps=20)
        assert all(np.isfinite(f.energy) for f in sim.forces)
