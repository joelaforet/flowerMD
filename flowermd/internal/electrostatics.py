"""Smeared (Gaussian-charge) electrostatics for the all-atom DPD force field.

Point charges cannot be used with a soft DPD core: as two opposite charges
pass through each other, ``q_i q_j / r`` diverges. Treating each charge as
a Gaussian cloud of standard deviation ``sigma`` gives the pair energy

    u(r) = C q_i q_j erf(kappa r) / r,    kappa = 1 / (2 sigma),

which is ordinary Coulomb beyond a few ``sigma`` and levels off at
``C q_i q_j 2 kappa / sqrt(pi)`` as ``r -> 0``, with ``C`` the Coulomb
constant. This is the smeared-charge model used in DPD electrostatics
(Groot, J. Chem. Phys. 118, 11265, 2003).

This is exactly the reciprocal-space part of an Ewald sum with splitting
parameter ``kappa``, so HOOMD's PPPM mesh computes it on the GPU with
per-particle charges and periodic images. `SmearedCoulomb` is
`hoomd.md.long_range.pppm.Coulomb` without the real-space `hoomd.md.pair.Ewald`
term and with ``kappa`` set from ``sigma`` instead of from an accuracy
target. PPPM subtracts the mesh interaction of neighbor-list exclusions, so
bonded 1-2, 1-3 and 1-4 pairs are excluded as they are for the DPD pair.
"""

import math

import hoomd
import numpy as np

COULOMB_CONSTANT = 332.0637133  # kcal/mol * Angstrom / e**2


def hoomd_charges(charges_e, scale=1.0):
    """Convert charges in e to HOOMD charge units for kcal/mol and Angstrom.

    HOOMD includes ``1 / (4 pi eps0)`` in the unit of charge, so a charge in
    e is multiplied by ``sqrt(COULOMB_CONSTANT)``.
    """
    return (
        np.asarray(charges_e, dtype=float) * scale * math.sqrt(COULOMB_CONSTANT)
    )


def mesh_resolution(box_lengths, kappa, spacing=0.5):
    """Return a PPPM grid with spacing at most ``spacing / kappa``.

    With fifth-order assignment, a spacing of ``0.5 / kappa`` reproduces
    ``erf(kappa r) / r`` forces to about 1 %.
    """
    return tuple(
        int(math.ceil(length * kappa / spacing)) for length in box_lengths
    )


class SmearedCoulomb(hoomd.md.long_range.pppm.Coulomb):
    """Gaussian-smeared Coulomb interactions on a PPPM mesh.

    Parameters
    ----------
    nlist : hoomd.md.nlist.NeighborList, required
        Neighbor list whose exclusions are removed from the interaction.
    resolution : tuple of int, required
        Grid points along x, y and z.
    order : int, required
        Charge assignment order.
    sigma : float, required
        Standard deviation of each Gaussian charge cloud (Angstrom).

    Notes
    -----
    Builds the same C++ object as `hoomd.md.long_range.pppm.Coulomb` but
    passes ``kappa = 1 / (2 sigma)`` directly. The real-space cutoff is set
    far beyond the Gaussian so HOOMD's error estimate for the (absent)
    real-space term stays negligible.

    """

    def __init__(self, nlist, resolution, order, sigma):
        if sigma <= 0:
            raise ValueError("sigma must be positive.")
        self.sigma = float(sigma)
        self.kappa = 1.0 / (2.0 * self.sigma)
        super(SmearedCoulomb, self).__init__(
            nlist=nlist,
            resolution=resolution,
            order=order,
            r_cut=6.0 / self.kappa,
            alpha=0.0,
            pair_force=None,
        )

    def _attach_hook(self):
        self.nlist._attach(self._simulation)
        if isinstance(self._simulation.device, hoomd.device.CPU):
            cls = hoomd.md._md.PPPMForceCompute
        else:
            cls = hoomd.md._md.PPPMForceComputeGPU
        group = self._simulation.state._get_group(hoomd.filter.All())
        self._cpp_obj = cls(
            self._simulation.state._cpp_sys_def, self.nlist._cpp_obj, group
        )
        nx, ny, nz = self.resolution
        self._cpp_obj.setParams(
            nx, ny, nz, self.order, self.kappa, self.r_cut, self.alpha
        )


def smeared_pair_energy(r, q_i, q_j, sigma):
    """Analytic smeared-Coulomb pair energy in kcal/mol (charges in e)."""
    kappa = 1.0 / (2.0 * sigma)
    r = np.asarray(r, dtype=float)
    limit = 2.0 * kappa / math.sqrt(math.pi)
    safe = np.where(r > 0, r, 1.0)
    shape = np.where(r > 0, np.vectorize(math.erf)(kappa * safe) / safe, limit)
    return COULOMB_CONSTANT * q_i * q_j * shape
