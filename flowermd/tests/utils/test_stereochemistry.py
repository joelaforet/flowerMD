import mbuild as mb
import numpy as np
import pytest

from flowermd.internal.stereochemistry import (
    STEREO_TYPE_PREFIX,
    StereoIntegrityError,
    append_stereochemistry_dihedrals,
    audit_stereochemistry,
    build_stereochemistry_force,
    capture_stereochemistry,
    normalized_volume,
    require_stereochemistry,
    signed_dihedral,
)

pytest.importorskip("rdkit")


def _chiral_compound():
    # (S)-1-chloro-1-fluoroethane: one tetrahedral stereocenter
    compound = mb.load("C[C@H](F)Cl", smiles=True)
    compound.box = mb.Box(lengths=[3.0, 3.0, 3.0])
    return compound


class TestStereochemistry:
    def test_normalized_volume_flips_under_reflection(self):
        points = np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
            ]
        )
        v = normalized_volume(points)
        assert abs(v) > 0.3
        mirrored = points * np.array([-1.0, 1.0, 1.0])
        assert normalized_volume(mirrored) == pytest.approx(-v)
        assert np.isnan(normalized_volume(points[:3]))

    def test_signed_dihedral_sign(self):
        pts = np.array(
            [[1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        phi = signed_dihedral(pts)
        mirrored = pts * np.array([-1.0, 1.0, 1.0])
        assert signed_dihedral(mirrored) == pytest.approx(-phi)

    def test_capture_and_audit(self):
        compound = _chiral_compound()
        reference = capture_stereochemistry(compound)
        assert reference.n_atoms == compound.n_particles
        assert len(reference.centers) == 1
        center = reference.centers[0]
        assert len(center.neighbors) == 4
        assert center.type_name.startswith(STEREO_TYPE_PREFIX)
        xyz = compound.xyz * 10.0
        report = audit_stereochemistry(reference, xyz)
        assert report["passed"] and report["inverted_count"] == 0
        mirrored = xyz * np.array([-1.0, 1.0, 1.0])
        report = audit_stereochemistry(reference, mirrored)
        assert not report["passed"] and report["inverted_count"] == 1
        with pytest.raises(StereoIntegrityError) as excinfo:
            require_stereochemistry(reference, mirrored)
        assert excinfo.value.report["inverted_count"] == 1
        # a translation and a proper rotation preserve it
        rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        moved = xyz @ rot.T + 5.0
        assert audit_stereochemistry(reference, moved)["passed"]

    def test_audit_with_wrapped_coordinates(self):
        compound = _chiral_compound()
        reference = capture_stereochemistry(compound)
        xyz = compound.xyz * 10.0
        box = np.array([30.0, 30.0, 30.0])
        wrapped = (xyz + box / 2) % box - box / 2
        wrapped[reference.centers[0].neighbors[0]] += box  # move one image
        assert audit_stereochemistry(reference, wrapped, box_lengths=box)[
            "passed"
        ]

    def test_no_centers(self):
        compound = mb.load("CCO", smiles=True)
        compound.box = mb.Box(lengths=[2.0, 2.0, 2.0])
        reference = capture_stereochemistry(compound)
        assert reference.centers == ()
        assert audit_stereochemistry(reference, compound.xyz * 10)["passed"]

    def test_frame_and_force(self):
        import gsd.hoomd

        compound = _chiral_compound()
        reference = capture_stereochemistry(compound)
        frame = gsd.hoomd.Frame()
        frame.particles.N = compound.n_particles
        frame.dihedrals.N = 1
        frame.dihedrals.types = ["existing"]
        frame.dihedrals.typeid = np.array([0], dtype=np.uint32)
        frame.dihedrals.group = np.array([[0, 1, 2, 3]], dtype=np.uint32)
        added = append_stereochemistry_dihedrals(frame, reference)
        assert added == 1
        assert frame.dihedrals.N == 2
        assert frame.dihedrals.types == ["existing", *reference.type_names]
        with pytest.raises(ValueError):
            append_stereochemistry_dihedrals(frame, reference)
        force = build_stereochemistry_force(
            reference, 30000.0, dihedral_types=["existing"]
        )
        assert force.params["existing"]["k"] == 0.0
        params = force.params[reference.centers[0].type_name]
        assert params["k"] == 30000.0 and params["n"] == 1 and params["d"] == -1
        assert params["phi0"] == pytest.approx(reference.centers[0].phi0)
        with pytest.raises(ValueError):
            build_stereochemistry_force(reference, 0.0)
