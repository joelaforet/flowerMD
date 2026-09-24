import json

import hoomd
import numpy as np
import pytest
import unyt as u

from flowermd.base import Molecule
from flowermd.internal.stereochemistry import StereoIntegrityError
from flowermd.library import (
    AllAtomDPD,
    AllAtomLattice,
    AllAtomPhantomWalk,
    PolyEthylene,
)
from flowermd.tests import BaseTest

pytest.importorskip("rdkit")

DENSITY = 0.5 * u.g / u.cm**3


def _pe_system(lengths=4, num_mols=3, seed=11):
    return AllAtomLattice(
        molecules=PolyEthylene(lengths=lengths, num_mols=num_mols),
        density=DENSITY,
        seed=seed,
    )


def _chiral_system(num_mols=6):
    # (S)-1-chloro-1-fluoroethane, one stereocenter per molecule
    mols = Molecule(num_mols=num_mols, smiles="C[C@H](F)Cl")
    return AllAtomLattice(molecules=mols, density=DENSITY, seed=5)


class TestAllAtomPhantomWalk(BaseTest):
    def test_guard_in_forcefield(self):
        system = _chiral_system()
        ff = AllAtomDPD(system.system)
        assert ff.stereo_centers == 6
        assert "stereochemistry" in ff.forces_by_role
        assert ff.hoomd_forces[-1] is ff.forces_by_role["pair"]
        stereo_types = set(ff.stereo_reference.type_names)
        assert stereo_types <= set(ff.frame.dihedrals.types)
        # both dihedral forces cover every dihedral type in the frame
        for role in ("dihedral", "stereochemistry"):
            assert set(ff.forces_by_role[role].params.keys()) == set(
                ff.frame.dihedrals.types
            )
        assert ff.timings["parameterization"] > 0
        unprotected = AllAtomDPD(system.system, protect_stereochemistry=False)
        assert unprotected.stereo_centers == 0
        assert "stereochemistry" not in unprotected.forces_by_role
        assert len(unprotected.frame.dihedrals.types) < len(
            ff.frame.dihedrals.types
        )

    def test_guard_without_centers_is_noop(self):
        ff = AllAtomDPD(_pe_system().system)
        assert ff.stereo_centers == 0
        assert "stereochemistry" not in ff.forces_by_role

    def test_conservative_forces(self):
        ff = AllAtomDPD(_pe_system().system)
        cons = ff.conservative_forces()
        assert isinstance(cons[-1], hoomd.md.pair.DPDConservative)
        assert cons[:-1] == ff.hoomd_forces[:-1]
        assert cons[-1] is not ff.forces_by_role["pair"]
        key = next(iter(cons[-1].params.keys()))
        assert cons[-1].params[key]["A"] == pytest.approx(
            ff.forces_by_role["pair"].params[key]["A"]
        )

    def test_run_initialization_record_and_handoff(self, tmp_path):
        system = _pe_system()
        ff = AllAtomDPD(system.system)
        sim = AllAtomPhantomWalk.from_system(system, forcefield=ff, dt=0.001)
        with pytest.warns(UserWarning, match="proceed with caution"):
            record = sim.run_initialization(
                dpd_min_steps=50,
                dpd_chunk=25,
                dpd_max_steps=100,
                energy_tol=1e-9,  # never stationary: exercise the cap path
                fire_steps=20,
            )
        assert record is sim.record
        assert record["dpd_steps"] == 100
        assert record["dpd_converged"] is False
        assert record["fire_steps"] == 20
        assert record["stereochemistry"] is None  # PE has no centers
        assert record["forcefield"]["bonded"] == "uff"
        assert set(record["timings_s"]) >= {
            "forcefield_parameterization",
            "forcefield_setup",
            "simulation_setup",
            "dpd",
            "fire",
            "audit",
            "total",
        }
        assert len(record["dpd_energy_history"]) == 2
        assert record["versions"]["hoomd"] == hoomd.version.version
        # MD still works after the conservative FIRE stage
        sim.run_NVE(n_steps=5)
        positions = sim.final_positions()
        assert positions.shape == (system.system.n_particles, 3)
        compound = sim.to_compound()
        assert np.allclose(compound.xyz, positions)
        path = tmp_path / "record.json"
        sim.write_record(path)
        assert json.loads(path.read_text())["dpd_steps"] == 100

    def test_require_convergence_raises(self):
        system = _pe_system()
        sim = AllAtomPhantomWalk.from_system(system, AllAtomDPD(system.system))
        with pytest.raises(RuntimeError, match="without the energies"):
            sim.run_initialization(
                dpd_min_steps=25,
                dpd_chunk=25,
                dpd_max_steps=50,
                energy_tol=1e-9,
                fire_steps=5,
                require_convergence=True,
            )

    def test_converges_with_loose_tolerance(self):
        system = _pe_system()
        sim = AllAtomPhantomWalk.from_system(system, AllAtomDPD(system.system))
        record = sim.run_initialization(
            dpd_min_steps=25,
            dpd_chunk=25,
            dpd_max_steps=500,
            energy_tol=10.0,  # any finite change passes
            consecutive=2,
            fire_steps=5,
        )
        assert record["dpd_converged"] is True
        assert (
            record["dpd_steps"] == 100
        )  # 25 + 3 chunks (first sample + 2 passes)

    def test_custom_stop(self):
        system = _pe_system()
        sim = AllAtomPhantomWalk.from_system(system, AllAtomDPD(system.system))
        calls = []
        record = sim.run_initialization(
            dpd_min_steps=0,
            dpd_chunk=10,
            dpd_max_steps=100,
            stop=lambda s: calls.append(s.timestep) or len(calls) == 3,
            fire_steps=5,
        )
        assert record["dpd_steps"] == 30 and record["settings"]["custom_stop"]
        assert "dpd_energy_history" not in record

    def test_stereochemistry_audited_and_preserved(self):
        system = _chiral_system()
        ff = AllAtomDPD(system.system)
        sim = AllAtomPhantomWalk.from_system(system, forcefield=ff)
        record = sim.run_initialization(
            dpd_min_steps=50,
            dpd_chunk=25,
            dpd_max_steps=200,
            energy_tol=10.0,
            fire_steps=20,
        )
        stereo = record["stereochemistry"]
        assert set(stereo) == {"initial", "post_dpd", "post_fire"}
        for stage in stereo.values():
            assert stage["n_centers"] == 6 and stage["passed"]

    def test_stereochemistry_failure_raises_or_warns(self):
        system = _chiral_system()
        ff = AllAtomDPD(system.system)
        sim = AllAtomPhantomWalk.from_system(system, forcefield=ff)
        sim.run_initialization(
            dpd_min_steps=10,
            dpd_chunk=10,
            dpd_max_steps=20,
            energy_tol=10.0,
            fire_steps=2,
        )
        # fake an inversion in the recorded reference to exercise the paths
        reference = ff.stereo_reference
        flipped = reference.__class__(
            n_atoms=reference.n_atoms,
            centers=tuple(
                c.__class__(**{**c.__dict__, "expected_sign": -c.expected_sign})
                for c in reference.centers
            ),
        )
        ff.stereo_reference = flipped
        with pytest.raises(StereoIntegrityError):
            sim.run_initialization(
                dpd_min_steps=10,
                dpd_chunk=10,
                dpd_max_steps=20,
                energy_tol=10.0,
                fire_steps=2,
            )
        with pytest.warns(UserWarning, match="Stereochemistry audit failed"):
            sim.run_initialization(
                dpd_min_steps=10,
                dpd_chunk=10,
                dpd_max_steps=20,
                energy_tol=10.0,
                fire_steps=2,
                require_stereochemistry=False,
            )

    def test_plain_force_list(self):
        system = _pe_system()
        ff = AllAtomDPD(system.system)
        sim = AllAtomPhantomWalk(
            initial_state=ff.frame, forcefield=ff.hoomd_forces
        )
        record = sim.run_initialization(
            dpd_min_steps=10,
            dpd_chunk=10,
            dpd_max_steps=20,
            energy_tol=10.0,
            fire_steps=2,
        )
        assert record["stereochemistry"] is None and "forcefield" not in record
        with pytest.raises(ValueError):
            sim.to_compound()
        with pytest.raises(TypeError):
            AllAtomPhantomWalk.from_system(system, ff.hoomd_forces)
