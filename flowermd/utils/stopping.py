"""Stopping criteria for chunked simulation runs (see `Simulation.run_DPD`)."""

import numpy as np


class EnergyStationarity:
    """Stop when the per-particle energy of every force stops changing.

    After each chunk of a chunked run, the per-particle energy of each
    monitored force is compared with its value after the previous chunk.
    A comparison passes when every force changed by at most `tol`
    (relative). The criterion returns True once `consecutive` comparisons
    in a row have passed. This is the energy-stationarity rule used by the
    all-atom PhantomWalk DPD initializer.

    Parameters
    ----------
    tol : float, default 0.02
        Maximum relative change, |new - old| / max(|new|, |old|), allowed
        for every monitored force between consecutive chunks.
    consecutive : int, default 2
        Number of consecutive passing comparisons required before stopping.
    forces : list of hoomd.md.force.Force, optional
        Forces to monitor. Defaults to every force in the simulation.

    Notes
    -----
    The criterion keeps its own state between calls. Call `reset()` before
    reusing one instance for a second run. `history` holds the per-particle
    energies recorded at each call, in the order the forces were monitored.

    """

    def __init__(self, tol=0.02, consecutive=2, forces=None):
        if tol <= 0:
            raise ValueError("tol must be positive.")
        if consecutive < 1:
            raise ValueError("consecutive must be at least 1.")
        self.tol = tol
        self.consecutive = consecutive
        self.forces = forces
        self.reset()

    def reset(self):
        """Forget previous energies and passing comparisons."""
        self._previous = None
        self._passes = 0
        self.history = []

    def __call__(self, sim):
        """Return True when the monitored energies have become stationary.

        Parameters
        ----------
        sim : flowermd.base.Simulation, required
            The running simulation. Its forces must be attached, which is
            the case whenever this is called from `Simulation.run_DPD`.

        """
        forces = self.forces if self.forces is not None else sim.forces
        n_particles = sim.state.N_particles
        current = np.array(
            [force.energy / n_particles for force in forces], dtype=float
        )
        self.history.append(current)
        if self._previous is None or not np.all(np.isfinite(current)):
            self._passes = 0
            self._previous = current
            return False
        scale = np.maximum(
            np.maximum(np.abs(current), np.abs(self._previous)),
            np.finfo(float).tiny,
        )
        relative_change = np.abs(current - self._previous) / scale
        if np.all(relative_change <= self.tol):
            self._passes += 1
        else:
            self._passes = 0
        self._previous = current
        return self._passes >= self.consecutive
