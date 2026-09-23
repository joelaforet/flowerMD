import numpy as np
import pytest
import unyt as u

from flowermd.internal.charges import assign_partial_charges
from flowermd.internal.electrostatics import COULOMB_CONSTANT
from flowermd.library import (
    AllAtomDPD,
    AllAtomLattice,
    AllAtomPhantomWalk,
    PEAAIonomer,
    PolyEthylene,
)
from flowermd.tests import BaseTest

pytest.importorskip("rdkit")

DENSITY = 0.8 * u.g / u.cm**3


def _ionomer_system(num_mols=3, lengths=2, seed=3):
    chains = PEAAIonomer(lengths=lengths, num_mols=num_mols, seed=seed)
    return chains, AllAtomLattice(
        molecules=[chains, chains.counterions()], density=DENSITY, seed=seed
    )


class TestPEAAIonomer(BaseTest):
    def test_chain_composition(self):
        chains = PEAAIonomer(lengths=1, num_mols=2, pattern="EEAEE")
        chain = chains.molecules[0]
        symbols = [p.element.symbol for p in chain.particles()]
        # 4 x C3H6 + C5H7O2 + 2 end hydrogens
        assert symbols.count("C") == 17
        assert symbols.count("O") == 2
        assert symbols.count("H") == 4 * 6 + 7 + 2
        assert chains.n_carboxylates == 2
        assert all(len(s) == 5 and s.count("E") == 4 for s in chains.sequences)

    def test_counterions(self):
        chains = PEAAIonomer(lengths=2, num_mols=3)
        sodium = chains.counterions()
        assert sodium.n_mols == chains.n_carboxylates == 6
        assert [p.element.symbol for p in sodium.molecules[0].particles()] == [
            "Na"
        ]
        assert chains.counterions("[Zn+2]").n_mols == 3
        for bad in ("[Cl-]", "CC"):
            with pytest.raises(ValueError):
                chains.counterions(bad)

    def test_tacticity(self):
        iso = PEAAIonomer(lengths=4, num_mols=1, tacticity="isotactic")
        assert "S" not in iso.sequences[0]
        a = PEAAIonomer(lengths=8, num_mols=1, seed=1)
        b = PEAAIonomer(lengths=8, num_mols=1, seed=2)
        assert a.sequences != b.sequences
        with pytest.raises(ValueError):
            PEAAIonomer(lengths=1, num_mols=1, pattern="EXA")

    def test_formal_and_gasteiger_charges(self):
        chains, system = _ionomer_system()
        formal = assign_partial_charges(system.system, "formal")
        assert formal.sum() == pytest.approx(0.0)
        assert (formal == -1).sum() == chains.n_carboxylates
        assert (formal == 1).sum() == chains.n_carboxylates
        gasteiger = assign_partial_charges(system.system, "gasteiger")
        assert np.all(np.isfinite(gasteiger))
        assert gasteiger.sum() == pytest.approx(0.0, abs=1e-9)
        assert np.count_nonzero(gasteiger) > np.count_nonzero(formal)


class TestSmearedElectrostatics(BaseTest):
    def test_forcefield_carries_charges(self):
        chains, system = _ionomer_system()
        ff = AllAtomDPD(
            system.system, charges="gasteiger", electrostatics="smeared"
        )
        assert "electrostatics" in ff.forces_by_role
        assert ff.hoomd_forces[-1] is ff.forces_by_role["pair"]
        assert ff.charge_method == "gasteiger"
        assert ff.net_charge == pytest.approx(0.0, abs=1e-9)
        assert np.allclose(
            ff.frame.particles.charge, ff.charges * np.sqrt(COULOMB_CONSTANT)
        )
        assert ff.stereo_centers == chains.n_carboxylates
        kappa = 1.0 / (2.0 * ff.charge_smearing)
        box = ff.frame.configuration.box[:3]
        assert ff.pppm_resolution == tuple(
            int(np.ceil(L * kappa / 0.5)) for L in box
        )
        # the conservative FIRE list keeps the electrostatics
        assert ff.forces_by_role["electrostatics"] in ff.conservative_forces()

    def test_default_has_no_charges(self):
        _, system = _ionomer_system()
        ff = AllAtomDPD(system.system)
        assert ff.charge_method is None
        assert not np.any(ff.frame.particles.charge)
        assert "electrostatics" not in ff.forces_by_role

    def test_argument_checks(self):
        chains = PEAAIonomer(lengths=2, num_mols=2)
        charged = AllAtomLattice(molecules=chains, density=DENSITY, seed=1)
        with pytest.raises(ValueError, match="net charge"):
            AllAtomDPD(
                charged.system, charges="formal", electrostatics="smeared"
            )
        with pytest.raises(ValueError, match="needs charges"):
            AllAtomDPD(charged.system, electrostatics="smeared")
        with pytest.raises(ValueError):
            AllAtomDPD(charged.system, charges="am1bcc")
        with pytest.raises(ValueError, match="one value per particle"):
            AllAtomDPD(charged.system, charges=[0.0, 0.0])

    def test_user_charges(self):
        pe = AllAtomLattice(
            molecules=PolyEthylene(lengths=3, num_mols=2),
            density=DENSITY,
            seed=2,
        )
        n = pe.system.n_particles
        charges = np.zeros(n)
        charges[0], charges[1] = 0.1, -0.1
        ff = AllAtomDPD(pe.system, charges=charges, electrostatics="smeared")
        assert ff.charge_method == "user"
        assert np.allclose(ff.charges, charges)

    def test_ionomer_initialization_end_to_end(self, tmp_path):
        chains, system = _ionomer_system()
        ff = AllAtomDPD(
            system.system, charges="gasteiger", electrostatics="smeared"
        )
        sim = AllAtomPhantomWalk.from_system(
            system,
            ff,
            gsd_file_name=str(tmp_path / "traj.gsd"),
            log_file_name=str(tmp_path / "log.txt"),
        )
        with pytest.warns(UserWarning):
            record = sim.run_initialization(
                dpd_min_steps=100,
                dpd_chunk=50,
                dpd_max_steps=200,
                fire_steps=20,
            )
        settings = record["forcefield"]
        assert settings["electrostatics"] == "smeared"
        assert settings["charges"] == "gasteiger"
        assert settings["charge_smearing_a"] == 2.0
        assert "electrostatics" in settings["included_terms"]
        assert np.all(np.isfinite(sim.final_positions()))
        assert record["stereochemistry"]["post_fire"]["passed"]
