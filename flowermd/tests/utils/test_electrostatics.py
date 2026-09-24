import math

import gsd.hoomd
import hoomd
import numpy as np
import pytest

from flowermd.internal.electrostatics import (
    COULOMB_CONSTANT,
    SmearedCoulomb,
    hoomd_charges,
    mesh_resolution,
    smeared_pair_energy,
)
from flowermd.tests import BaseTest

BOX = 60.0
SIGMA = 2.0


def _pair(r, bonded=False):
    """Energy and x-force on particle 1 for a +1/-1 pair at separation r."""
    frame = gsd.hoomd.Frame()
    frame.configuration.box = [BOX, BOX, BOX, 0, 0, 0]
    frame.particles.N = 2
    frame.particles.types = ["A"]
    frame.particles.typeid = [0, 0]
    frame.particles.position = [[-r / 2, 0, 0], [r / 2, 0, 0]]
    frame.particles.charge = hoomd_charges([1.0, -1.0])
    frame.particles.mass = [1.0, 1.0]
    forces = []
    if bonded:
        frame.bonds.N = 1
        frame.bonds.types = ["b"]
        frame.bonds.typeid = [0]
        frame.bonds.group = [[0, 1]]
        bond = hoomd.md.bond.Harmonic()
        bond.params["b"] = dict(k=0.0, r0=1.0)
        forces.append(bond)
    sim = hoomd.Simulation(device=hoomd.device.CPU(), seed=1)
    sim.create_state_from_snapshot(frame)
    nlist = hoomd.md.nlist.Cell(buffer=0.4, exclusions=["bond"])
    kappa = 1.0 / (2.0 * SIGMA)
    coulomb = SmearedCoulomb(
        nlist=nlist,
        resolution=mesh_resolution([BOX] * 3, kappa),
        order=5,
        sigma=SIGMA,
    )
    forces.append(coulomb)
    sim.operations.integrator = hoomd.md.Integrator(dt=0.0, forces=forces)
    sim.run(0)
    return coulomb.energy, coulomb.forces[1][0]


class TestSmearedCoulomb(BaseTest):
    def test_matches_erf_form(self):
        rs = [0.2, 1.0, 3.0, 6.0]
        e_far, _ = _pair(10.0)
        analytic_far = smeared_pair_energy(10.0, 1.0, -1.0, SIGMA)
        kappa = 1.0 / (2.0 * SIGMA)
        for r in rs:
            energy, force = _pair(r)
            expected = smeared_pair_energy(r, 1.0, -1.0, SIGMA)
            # energy differences remove the r-independent self and image
            # terms; images of a neutral pair in a 60 A box are small
            assert energy - e_far == pytest.approx(
                expected - analytic_far, rel=0.01, abs=0.3
            )
            magnitude = COULOMB_CONSTANT * (
                math.erf(kappa * r) / r**2
                - 2
                * kappa
                / math.sqrt(math.pi)
                * math.exp(-((kappa * r) ** 2))
                / r
            )
            # the -1 charge at +r/2 is pulled toward -x
            assert force == pytest.approx(-magnitude, rel=0.02, abs=0.02)

    def test_bounded_at_contact(self):
        limit = smeared_pair_energy(0.0, 1.0, -1.0, SIGMA)
        assert np.isfinite(limit)
        assert limit == pytest.approx(
            -COULOMB_CONSTANT / (SIGMA * math.sqrt(math.pi))
        )

    def test_bond_exclusion_removes_pair(self):
        energy, force = _pair(2.0, bonded=True)
        assert abs(energy) < 0.1
        assert abs(force) < 0.05

    def test_mesh_resolution(self):
        assert mesh_resolution([10.0, 20.0, 30.0], 0.25) == (5, 10, 15)

    def test_bad_sigma(self):
        with pytest.raises(ValueError):
            SmearedCoulomb(
                nlist=hoomd.md.nlist.Cell(buffer=0.4),
                resolution=(8, 8, 8),
                order=5,
                sigma=0.0,
            )
