"""DPD energy relaxation simulation class and its run loops."""

import warnings

import hoomd

from flowermd.base.simulation import Simulation
from flowermd.utils import StdOutLogger, compute_closest_rdf


class PhantomWalk(Simulation):
    """Run an initial energy relaxation of overlapping particles with DPD and FIRE."""

    def __init__(
        self,
        initial_state,
        forcefield,
        n_steps_dpd,
        n_steps_fire,
        reference_values=dict(),
        dt=0.001,
        device=hoomd.device.auto_select(),
        seed=42,
        gsd_write_freq=1e4,
        gsd_file_name="trajectory.gsd",
        log_write_freq=1e3,
        log_file_name="log.txt",
    ):
        self.n_steps_dpd = n_steps_dpd
        self.n_steps_fire = n_steps_fire
        super(PhantomWalk, self).__init__(
            initial_state=initial_state,
            forcefield=forcefield,
            reference_values=reference_values,
            dt=dt,
            device=device,
            seed=seed,
            gsd_write_freq=gsd_write_freq,
            gsd_file_name=gsd_file_name,
            log_write_freq=log_write_freq,
            log_file_name=log_file_name,
        )
        self.run_NVE(n_steps=self.n_steps_dpd, write_at_start=False)
        self.run_FIRE(n_steps=self.n_steps_fire)
        for writer in self.operations.writers:
            if hasattr(writer, "flush"):
                writer.flush()
        print(
            "The closest particles after DPD + FIRE are: ",
            compute_closest_rdf(self, bins=100, r_max=1.0),
        )


def _check_chunks(n_steps, chunk, min_steps):
    if chunk < 1:
        raise ValueError("chunk must be at least 1.")
    if min_steps > n_steps:
        raise ValueError("min_steps cannot exceed n_steps.")


def run_until(
    sim,
    n_steps,
    stop,
    chunk=500,
    min_steps=0,
    write_at_start=True,
):
    """Advance `sim` with its current integrator until ``stop(sim)`` is True.

    Runs `min_steps` first, then chunks of `chunk` steps, calling
    ``stop(sim)`` after each chunk, and returns as soon as it is True or
    when `n_steps` total steps have been run. This is the convergence loop
    shared by `run_DPD` (stop on energy stationarity) and
    `run_FIRE_until_converged` (stop when FIRE has converged).

    Parameters
    ----------
    sim : flowermd.base.Simulation, required
        A simulation whose integrator is already set.
    n_steps : int, required
        Maximum total number of steps for this call.
    stop : callable, required
        ``stop(sim) -> bool``, evaluated after each chunk.
    chunk : int, default 500
        Number of steps between evaluations of `stop`.
    min_steps : int, default 0
        Steps to run before the first chunk.
    write_at_start : bool, default True
        When True, triggers writers that evaluate to True for the initial
        step before the first simulation step.

    Returns
    -------
    dict
        ``{"steps": int, "stopped_by_criterion": bool}``.

    """
    _check_chunks(n_steps, chunk, min_steps)
    steps_run = 0
    if min_steps > 0:
        sim.run(steps=min_steps, write_at_start=write_at_start)
        steps_run = min_steps
        write_at_start = False
    while steps_run < n_steps:
        this_chunk = min(chunk, n_steps - steps_run)
        sim.run(steps=this_chunk, write_at_start=write_at_start)
        write_at_start = False
        steps_run += this_chunk
        if stop(sim):
            return {"steps": steps_run, "stopped_by_criterion": True}
    return {"steps": steps_run, "stopped_by_criterion": False}


def run_DPD(
    sim,
    n_steps,
    stop=None,
    chunk=500,
    min_steps=0,
    write_at_start=True,
):
    """Run DPD, NVE dynamics thermostatted by the DPD pair force.

    With ``stop=None`` this runs exactly `n_steps`, like `run_NVE`.
    Otherwise it runs `min_steps`, then chunks of `chunk` steps until
    ``stop(sim)`` is True or `n_steps` have been run (see `run_until`).
    `flowermd.utils.EnergyStationarity` is the energy-based rule used by
    the all-atom PhantomWalk initializer.

    Parameters
    ----------
    sim : flowermd.base.Simulation, required
    n_steps : int, required
        Maximum total number of steps.
    stop, chunk, min_steps, write_at_start
        As in `run_until`.

    Returns
    -------
    dict
        ``{"steps": int, "stopped_by_criterion": bool}``.

    """
    _check_chunks(n_steps, chunk, min_steps)
    dpd_types = (hoomd.md.pair.DPD, hoomd.md.pair.DPDLJ)
    if not any(isinstance(f, dpd_types) for f in sim._forcefield):
        warnings.warn(
            "run_DPD: no hoomd.md.pair.DPD or DPDLJ force found; the NVE run "
            "will not be thermostatted."
        )
    sim.set_integrator_method(
        integrator_method=hoomd.md.methods.ConstantVolume,
        method_kwargs={"filter": sim.integrate_group},
    )
    std_out_logger = hoomd.update.CustomUpdater(
        trigger=hoomd.trigger.Periodic(sim._std_out_freq),
        action=StdOutLogger(n_steps=n_steps, sim=sim),
    )
    sim.operations.updaters.append(std_out_logger)
    try:
        if stop is None:
            sim.run(steps=n_steps, write_at_start=write_at_start)
            return {"steps": n_steps, "stopped_by_criterion": False}
        return run_until(
            sim,
            n_steps,
            stop,
            chunk=chunk,
            min_steps=min_steps,
            write_at_start=write_at_start,
        )
    finally:
        sim.operations.updaters.remove(std_out_logger)


def run_FIRE_until_converged(
    sim,
    n_steps,
    max_steps,
    force_tol=1e-1,
    angmom_tol=1000,
    energy_tol=1e-1,
    write_at_start=False,
    **fire_kwargs,
):
    """Run FIRE in chunks of `n_steps` until it converges or `max_steps`.

    One minimizer is kept for the whole call, so its adaptive time step
    and mixing carry over between chunks (see `run_until`).

    Parameters
    ----------
    sim : flowermd.base.Simulation, required
    n_steps : int, required
        Steps per chunk; convergence is checked after each chunk.
    max_steps : int, required
        Upper bound on the total number of FIRE steps.
    force_tol, angmom_tol, energy_tol, write_at_start, **fire_kwargs
        As in `flowermd.base.Simulation.run_FIRE`.

    Returns
    -------
    dict
        ``{"steps": int, "converged": bool}``.

    """
    if max_steps < n_steps:
        raise ValueError("max_steps must be at least n_steps.")
    sim.set_fire_minimizer(
        fire_kwargs={
            "force_tol": force_tol,
            "angmom_tol": angmom_tol,
            "energy_tol": energy_tol,
            **fire_kwargs,
        },
        integrator_method=hoomd.md.methods.ConstantVolume,
        method_kwargs={"filter": sim.integrate_group},
    )
    result = run_until(
        sim,
        max_steps,
        lambda s: s.integrator.converged,
        chunk=n_steps,
        write_at_start=write_at_start,
    )
    return {
        "steps": result["steps"],
        "converged": bool(sim.integrator.converged),
    }
