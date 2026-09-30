import hoomd
import pytest
import unyt as u

from flowermd.base import Simulation
from flowermd.library import DPD, PPS, LJChain, PhantomWalk, RandomWalk
from flowermd.library.simulations.phantom_walk import (
    run_DPD,
    run_FIRE_until_converged,
)
from flowermd.tests import BaseTest
from flowermd.utils import EnergyStationarity


class TestPhantomWalkSimulation(BaseTest):
    def test_phantom_walk_run(self):
        pps = PPS(lengths=6, num_mols=32)
        pps.coarse_grain(beads={"_A": "c1cc(S)ccc1"})

        ref_length = 0.3438 * u.Unit("nm")
        ref_mass = 32.06 * u.Unit("amu")
        ref_energy = 1.065 * u.Unit("kJ/mol")
        ref_values_dict = {
            "length": ref_length,
            "mass": ref_mass,
            "energy": ref_energy,
        }

        system = RandomWalk(
            molecules=pps,
            density=1.32 * u.Unit("g/cm**3"),
            bond_length=1.4226,
            buffer=0.58,
            base_units=ref_values_dict,
        )

        dpd_ff = DPD(
            A=25000, gamma=800, kT=1.5, r_cut=1.5, bond_k=25000, bond_r0=1.4226
        )

        PhantomWalk(
            initial_state=system.hoomd_snapshot,
            forcefield=dpd_ff.hoomd_forces,
            gsd_write_freq=10,
            log_write_freq=50,
            n_steps_dpd=500,
            n_steps_fire=100,
        )

    def test_phantom_walk_comp(self):
        molecules = LJChain(
            num_mols=[10],
            lengths=[50],
            bead_sequence=["_A"],
            bead_mass={"_A": 1.0},
            bond_lengths={"_A-_A": 1.0},
        )

        ref_length = 1.0 * u.Unit("nm")
        ref_mass = 1.0 * u.Unit("g/mol")
        ref_energy = 1.0 * u.Unit("kcal / mol")
        ref_values_dict = {
            "length": ref_length,
            "mass": ref_mass,
            "energy": ref_energy,
        }

        system = RandomWalk(
            molecules=molecules,
            density=1.1 * u.Unit("nm**-3"),
            bond_length=1.0,
            buffer=0.5,
            base_units=ref_values_dict,
        )

        dpd_ff = DPD(
            A=25000, gamma=800, kT=1.0, r_cut=1.01, bond_k=25000, bond_r0=1.0
        )

        sim = Simulation(
            initial_state=system.hoomd_snapshot,
            forcefield=dpd_ff.hoomd_forces,
            reference_values=ref_values_dict,
        )

        sim.run_NVE(n_steps=10, write_at_start=False)
        assert isinstance(sim.forces[0], hoomd.md.pair.pair.DPD)
        assert isinstance(sim.integrator, hoomd.md.Integrator)
        assert isinstance(sim.method, hoomd.md.methods.ConstantVolume)

        sim.run_FIRE(n_steps=10)
        assert isinstance(sim.forces[0], hoomd.md.pair.pair.DPD)
        assert isinstance(sim.integrator, hoomd.md.minimize.FIRE)
        assert isinstance(sim.method, hoomd.md.methods.ConstantVolume)


class TestPhantomWalkRunLoops(BaseTest):
    def _dpd_simulation(self, benzene_cg_system):
        snapshot = benzene_cg_system.hoomd_snapshot
        dpd = hoomd.md.pair.DPD(
            nlist=hoomd.md.nlist.Cell(buffer=0.4), kT=1.0, default_r_cut=1.0
        )
        for i, type_i in enumerate(snapshot.particles.types):
            for type_j in snapshot.particles.types[i:]:
                dpd.params[(type_i, type_j)] = dict(A=25.0, gamma=4.5)
        return Simulation(initial_state=snapshot, forcefield=[dpd], dt=0.01)

    def test_run_DPD_fixed_steps(self, benzene_cg_system):
        sim = self._dpd_simulation(benzene_cg_system)
        result = run_DPD(sim, n_steps=300)
        assert result == {"steps": 300, "stopped_by_criterion": False}
        assert sim.timestep == 300
        assert isinstance(sim.method, hoomd.md.methods.ConstantVolume)

    def test_run_DPD_stop_callable(self, benzene_cg_system):
        sim = self._dpd_simulation(benzene_cg_system)
        calls = []

        def stop_after_two(s):
            calls.append(s.timestep)
            return len(calls) == 2

        result = run_DPD(
            sim, n_steps=5000, stop=stop_after_two, chunk=100, min_steps=50
        )
        assert result == {"steps": 250, "stopped_by_criterion": True}
        assert calls == [150, 250]

    def test_run_DPD_reaches_n_steps_without_stopping(self, benzene_cg_system):
        sim = self._dpd_simulation(benzene_cg_system)
        result = run_DPD(sim, n_steps=250, stop=lambda s: False, chunk=100)
        assert result == {"steps": 250, "stopped_by_criterion": False}

    def test_run_DPD_energy_stationarity(self, benzene_cg_system):
        sim = self._dpd_simulation(benzene_cg_system)
        criterion = EnergyStationarity(tol=1.0, consecutive=2)
        result = run_DPD(sim, n_steps=2000, stop=criterion, chunk=100)
        # tol=1.0 passes any finite change, so it stops after 3 chunks
        assert result == {"steps": 300, "stopped_by_criterion": True}
        assert len(criterion.history) == 3

    def test_run_DPD_warns_without_dpd_force(self, benzene_system):
        sim = Simulation.from_system(benzene_system)
        with pytest.warns(UserWarning, match="run_DPD"):
            run_DPD(sim, n_steps=10)

    def test_run_DPD_bad_arguments(self, benzene_cg_system):
        sim = self._dpd_simulation(benzene_cg_system)
        with pytest.raises(ValueError):
            run_DPD(sim, n_steps=10, chunk=0)
        with pytest.raises(ValueError):
            run_DPD(sim, n_steps=10, min_steps=20)

    def test_run_FIRE_until_converged(self, benzene_system):
        sim = Simulation.from_system(benzene_system)
        result = run_FIRE_until_converged(
            sim,
            n_steps=50,
            max_steps=500,
            dt=1e-4,
            force_tol=1e3,
            energy_tol=1e3,
        )
        assert result["converged"] is True
        assert 50 <= result["steps"] <= 500
        assert result["steps"] % 50 == 0
        sim.run_NVE(n_steps=10)
        assert type(sim.integrator) is hoomd.md.Integrator

    def test_run_FIRE_until_converged_bad_max_steps(self, benzene_system):
        sim = Simulation.from_system(benzene_system)
        with pytest.raises(ValueError):
            run_FIRE_until_converged(sim, n_steps=10, max_steps=5)
