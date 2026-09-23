"""All-atom PhantomWalk initialization: DPD relaxation followed by FIRE."""

import json
import platform
import time
import warnings

import gsd
import hoomd
import numpy as np

import flowermd
from flowermd.base.simulation import Simulation
from flowermd.internal.stereochemistry import (
    StereoIntegrityError,
    audit_stereochemistry,
)
from flowermd.library.forcefields import AllAtomDPD
from flowermd.utils import EnergyStationarity, HOOMDThermostats


class AllAtomPhantomWalk(Simulation):
    """Turn an overlapping all-atom placement into minimizer-ready coordinates.

    The all-atom counterpart of the coarse-grained ``PhantomWalk`` proposed
    in cmelab/flowerMD#264. Chains placed at the target density by
    `flowermd.library.AllAtomRandomWalk` or `AllAtomLattice` are relaxed with
    dissipative particle dynamics under the `flowermd.library.AllAtomDPD`
    force field until the energy of every force is stationary, then cleaned
    up with a short FIRE minimization under conservative DPD repulsion. The
    result is a set of coordinates a standard force field can minimize; it
    is not an equilibrated melt.

    Build it from an `AllAtomDPD` so the stereochemistry audit, the
    conservative FIRE force list and the parameterization timings come
    along::

        ff = AllAtomDPD(system.system)
        sim = AllAtomPhantomWalk.from_system(system, forcefield=ff)
        record = sim.run_initialization()

    Parameters
    ----------
    initial_state : gsd.hoomd.Frame, required
        The frame from `AllAtomDPD.frame`.
    forcefield : flowermd.library.AllAtomDPD or list, required
        The all-atom DPD force field. A plain list of HOOMD forces is also
        accepted; then there is no stereochemistry audit and FIRE runs with
        the same forces as DPD.
    dt : float, default 0.001
        DPD time step in the force field's units (Angstrom, kcal/mol, amu).
    Other parameters are passed to `flowermd.base.Simulation`.

    """

    def __init__(
        self,
        initial_state,
        forcefield,
        reference_values=dict(),
        dt=0.001,
        device=hoomd.device.auto_select(),
        seed=42,
        gsd_write_freq=1e4,
        gsd_file_name="trajectory.gsd",
        log_write_freq=1e3,
        log_file_name="sim_data.txt",
        thermostat=HOOMDThermostats.MTTK,
    ):
        if isinstance(forcefield, AllAtomDPD):
            self.aa_forcefield = forcefield
            forces = forcefield.hoomd_forces
        else:
            self.aa_forcefield = None
            forces = forcefield
        self._setup_started = time.perf_counter()
        super(AllAtomPhantomWalk, self).__init__(
            initial_state=initial_state,
            forcefield=forces,
            reference_values=reference_values,
            dt=dt,
            device=device,
            seed=seed,
            gsd_write_freq=gsd_write_freq,
            gsd_file_name=gsd_file_name,
            log_write_freq=log_write_freq,
            log_file_name=log_file_name,
            thermostat=thermostat,
        )
        self._setup_s = time.perf_counter() - self._setup_started
        self.record = None

    @classmethod
    def from_system(cls, system, forcefield, **kwargs):
        """Create the simulation from a placement `System` and an `AllAtomDPD`.

        The initial state is ``forcefield.frame``, whose interaction types
        match the forces; `system` is kept as ``sim.system`` so
        `to_compound` can hand back updated coordinates.
        """
        if not isinstance(forcefield, AllAtomDPD):
            raise TypeError("forcefield must be an AllAtomDPD instance.")
        sim = cls(
            initial_state=forcefield.frame, forcefield=forcefield, **kwargs
        )
        sim.system = system
        return sim

    @property
    def stereo_reference(self):
        """The recorded stereocenters, or None."""
        if self.aa_forcefield is None:
            return None
        return self.aa_forcefield.stereo_reference

    def _positions(self, unwrap):
        snapshot = self.state.get_snapshot()
        positions = np.asarray(snapshot.particles.position, dtype=float)
        if unwrap:
            box = np.asarray(snapshot.configuration.box[:3], dtype=float)
            positions = positions + snapshot.particles.image * box
        return positions

    def audit_stereochemistry(self):
        """Audit the current coordinates against the recorded stereocenters."""
        if self.stereo_reference is None:
            return None
        snapshot = self.state.get_snapshot()
        return audit_stereochemistry(
            self.stereo_reference,
            np.asarray(snapshot.particles.position, dtype=float),
            box_lengths=np.asarray(snapshot.configuration.box[:3], dtype=float),
            planar_tolerance=self.aa_forcefield.stereo_planar_tolerance,
        )

    def final_positions(self, unwrap=True):
        """Current particle positions in nm (unwrapped by default)."""
        return self._positions(unwrap) / 10.0

    def to_compound(self, compound=None):
        """Write the current coordinates into an mBuild compound (nm).

        Defaults to ``self.system.system`` when built with `from_system`.
        Particle order is the order `AllAtomDPD` read from the compound.
        """
        if compound is None:
            if getattr(self, "system", None) is None:
                raise ValueError("Pass a compound or build with from_system.")
            compound = self.system.system
        positions = self.final_positions(unwrap=True)
        if compound.n_particles != len(positions):
            raise ValueError("Compound and simulation particle counts differ.")
        compound.xyz = positions
        return compound

    def run_initialization(
        self,
        dpd_min_steps=4000,
        dpd_chunk=500,
        dpd_max_steps=40000,
        energy_tol=0.02,
        consecutive=2,
        stop=None,
        fire_steps=100,
        fire_dt=0.001,
        fire_kwargs=None,
        require_convergence=False,
        require_stereochemistry=True,
        write_at_start=True,
    ):
        """Run DPD until stationary, then FIRE, and return a record.

        Parameters
        ----------
        dpd_min_steps : int, default 4000
            DPD steps before the stopping criterion is first evaluated.
        dpd_chunk : int, default 500
            Steps between evaluations.
        dpd_max_steps : int, default 40000
            Cap on DPD steps.
        energy_tol : float, default 0.02
            Relative energy change per force allowed between chunks.
        consecutive : int, default 2
            Consecutive stationary comparisons required.
        stop : callable, optional
            ``stop(sim) -> bool`` replacing the energy-stationarity rule.
        fire_steps : int, default 100
            FIRE steps after DPD, run with conservative DPD repulsion.
        fire_dt : float, default 0.001
        fire_kwargs : dict, optional
            Extra arguments for `Simulation.run_FIRE`.
        require_convergence : bool, default False
            When True, raise ``RuntimeError`` if DPD reaches `dpd_max_steps`
            without stationarity. When False (default) the coordinates are
            still returned, ``record["dpd_converged"]`` is False and a
            warning asks the user to proceed with caution.
        require_stereochemistry : bool, default True
            Raise `StereoIntegrityError` if any recorded stereocenter is
            inverted or planar after FIRE. When False, only warn.
        write_at_start : bool, default True

        Returns
        -------
        dict
            The run record: settings, phase wall times, DPD steps and energy
            history, FIRE result, stereochemistry audits (initial, post_dpd,
            post_fire), and software versions. Also stored as
            ``self.record``.

        """
        if dpd_min_steps > dpd_max_steps:
            raise ValueError("dpd_min_steps cannot exceed dpd_max_steps.")
        criterion = stop
        if criterion is None:
            criterion = EnergyStationarity(
                tol=energy_tol, consecutive=consecutive
            )
        record = {
            "settings": {
                "dpd_min_steps": dpd_min_steps,
                "dpd_chunk": dpd_chunk,
                "dpd_max_steps": dpd_max_steps,
                "energy_tol": energy_tol,
                "consecutive": consecutive,
                "custom_stop": stop is not None,
                "fire_steps": fire_steps,
                "fire_dt": fire_dt,
                "fire_kwargs": dict(fire_kwargs or {}),
                "dt": self.dt,
                "seed": self.seed,
                "require_convergence": require_convergence,
                "require_stereochemistry": require_stereochemistry,
            },
            "n_particles": int(self.state.N_particles),
            "box_lengths_a": [float(x) for x in self.state.box.L],
            "timings_s": {},
        }
        if self.aa_forcefield is not None:
            record["forcefield"] = {
                "bonded": self.aa_forcefield.bonded,
                "A": self.aa_forcefield.A,
                "gamma": self.aa_forcefield.gamma,
                "kT": self.aa_forcefield.kT,
                "r_cut": self.aa_forcefield.r_cut,
                "bonded_scale": self.aa_forcefield.bonded_scale,
                "epsilon_weighting": self.aa_forcefield.epsilon_weighting,
                "included_terms": sorted(
                    r for r in self.aa_forcefield.forces_by_role if r != "pair"
                ),
                "protect_stereochemistry": (
                    self.aa_forcefield.protect_stereochemistry
                ),
                "stereo_k": self.aa_forcefield.stereo_k,
                "stereo_centers": self.aa_forcefield.stereo_centers,
                "epsilon_ref_kcal_mol": self.aa_forcefield.parameters.epsilon_ref,
            }
            record["timings_s"].update(
                {
                    f"forcefield_{k}": v
                    for k, v in self.aa_forcefield.timings.items()
                }
            )
        record["timings_s"]["simulation_setup"] = self._setup_s
        stereo = {}
        if self.stereo_reference is not None and self.stereo_reference.centers:
            stereo["initial"] = self.audit_stereochemistry()

        started = time.perf_counter()
        dpd = self.run_DPD(
            n_steps=dpd_max_steps,
            stop=criterion,
            chunk=dpd_chunk,
            min_steps=dpd_min_steps,
            write_at_start=write_at_start,
        )
        record["timings_s"]["dpd"] = time.perf_counter() - started
        record["dpd_steps"] = dpd["steps"]
        record["dpd_converged"] = bool(dpd["stopped_by_criterion"])
        if isinstance(criterion, EnergyStationarity):
            record["dpd_energy_history"] = [
                [float(x) for x in row] for row in criterion.history
            ]
            record["dpd_monitored_forces"] = [
                type(f).__name__ for f in self.forces
            ]
        if not record["dpd_converged"]:
            message = (
                f"DPD reached the cap of {dpd_max_steps} steps without the "
                "energies becoming stationary. The coordinates are returned, "
                "but proceed with caution: they may not be minimizer-ready."
            )
            if require_convergence:
                raise RuntimeError(message)
            warnings.warn(message)
        if stereo:
            stereo["post_dpd"] = self.audit_stereochemistry()

        started = time.perf_counter()
        fire_forces = None
        if self.aa_forcefield is not None:
            fire_forces = self.aa_forcefield.conservative_forces()
        fire = self._run_fire_with(
            fire_forces, fire_steps, fire_dt, fire_kwargs or {}
        )
        record["timings_s"]["fire"] = time.perf_counter() - started
        record["fire_steps"] = fire["steps"]
        record["fire_converged"] = fire["converged"]

        started = time.perf_counter()
        if stereo:
            stereo["post_fire"] = self.audit_stereochemistry()
            if not stereo["post_fire"]["passed"]:
                if require_stereochemistry:
                    record["stereochemistry"] = stereo
                    self.record = record
                    raise StereoIntegrityError(stereo["post_fire"])
                warnings.warn(
                    "Stereochemistry audit failed after FIRE: "
                    f"{stereo['post_fire']['inverted_count']} inverted, "
                    f"{stereo['post_fire']['near_planar_count']} near-planar "
                    f"of {stereo['post_fire']['n_centers']} centers."
                )
        record["stereochemistry"] = stereo or None
        record["timings_s"]["audit"] = time.perf_counter() - started
        record["timings_s"]["total"] = sum(record["timings_s"].values())
        record["versions"] = {
            "flowermd": getattr(flowermd, "__version__", "unknown"),
            "hoomd": hoomd.version.version,
            "gsd": gsd.version.version,
            "python": platform.python_version(),
            "device": type(self.device).__name__,
        }
        self.record = record
        return record

    def _run_fire_with(self, forces, n_steps, dt, fire_kwargs):
        """Run FIRE, temporarily swapping in `forces` when given."""
        if forces is None:
            return self.run_FIRE(n_steps=n_steps, dt=dt, **fire_kwargs)
        dpd_forces = self._forcefield
        # The bonded force objects are shared; detach them from the MD
        # integrator before FIRE takes them, then restore afterwards.
        self.operations.integrator = None
        self.integrator = None
        self._forcefield = forces
        try:
            result = self.run_FIRE(n_steps=n_steps, dt=dt, **fire_kwargs)
        finally:
            self._forcefield = dpd_forces
        return result

    def write_record(self, path):
        """Write ``self.record`` as JSON."""
        if self.record is None:
            raise ValueError("Run run_initialization first.")
        with open(path, "w") as handle:
            json.dump(self.record, handle, indent=2, default=_json_default)


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )
