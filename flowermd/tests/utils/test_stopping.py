import numpy as np
import pytest

from flowermd.utils import EnergyStationarity


class _FakeForce:
    def __init__(self, energy):
        self.energy = energy


class _FakeState:
    N_particles = 10


class _FakeSim:
    def __init__(self, *energies):
        self.forces = [_FakeForce(e) for e in energies]
        self.state = _FakeState()


class TestEnergyStationarity:
    def test_requires_consecutive_passes(self):
        criterion = EnergyStationarity(tol=0.02, consecutive=2)
        assert criterion(_FakeSim(100.0, -50.0)) is False  # first sample
        assert criterion(_FakeSim(100.5, -50.2)) is False  # pass 1
        assert criterion(_FakeSim(100.9, -50.1)) is True  # pass 2
        assert len(criterion.history) == 3
        np.testing.assert_allclose(criterion.history[0], [10.0, -5.0])

    def test_any_force_changing_resets_the_count(self):
        criterion = EnergyStationarity(tol=0.02, consecutive=2)
        criterion(_FakeSim(100.0, -50.0))
        assert criterion(_FakeSim(100.5, -50.2)) is False
        assert criterion(_FakeSim(100.5, -60.0)) is False  # second force moved
        assert criterion(_FakeSim(100.5, -60.1)) is False  # pass 1 again
        assert criterion(_FakeSim(100.5, -60.2)) is True

    def test_nonfinite_energy_resets(self):
        criterion = EnergyStationarity(tol=0.02, consecutive=1)
        criterion(_FakeSim(1.0))
        assert criterion(_FakeSim(np.nan)) is False
        assert criterion(_FakeSim(1.0)) is False
        assert criterion(_FakeSim(1.0)) is True

    def test_explicit_force_subset(self):
        sim = _FakeSim(1.0, 1000.0)
        criterion = EnergyStationarity(
            tol=0.02, consecutive=1, forces=sim.forces[:1]
        )
        criterion(sim)
        sim.forces[1].energy = -1000.0  # unmonitored force may change freely
        assert criterion(sim) is True

    def test_reset(self):
        criterion = EnergyStationarity(tol=0.02, consecutive=1)
        criterion(_FakeSim(1.0))
        criterion(_FakeSim(1.0))
        criterion.reset()
        assert criterion.history == []
        assert criterion(_FakeSim(1.0)) is False

    def test_bad_arguments(self):
        with pytest.raises(ValueError):
            EnergyStationarity(tol=0.0)
        with pytest.raises(ValueError):
            EnergyStationarity(consecutive=0)
