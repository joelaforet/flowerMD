import numpy as np
import pytest
import unyt as u

from flowermd.internal.monomers import monomer_from_marked_smiles
from flowermd.internal.stereochemistry import capture_stereochemistry
from flowermd.library import (
    PEI,
    PET,
    PMMA,
    AllAtomDPD,
    AllAtomLattice,
    MarkedSmilesPolymer,
    Polycarbonate,
    PolyStyrene,
)
from flowermd.tests import BaseTest

pytest.importorskip("rdkit")


class TestMarkedMonomer(BaseTest):
    def test_ports_become_hydrogens(self):
        comp, idx = monomer_from_marked_smiles("c1ccc([C@H](C[*:2])[*:1])cc1")
        parts = list(comp.particles())
        assert comp.n_particles == 18  # C8H8 + two attachment H
        assert [parts[i].name for i in idx] == ["H", "H"]
        for i in idx:
            heavy = next(iter(parts[i].direct_bonds()))
            d = np.linalg.norm(np.asarray(parts[i].pos) - np.asarray(heavy.pos))
            assert d == pytest.approx(0.109, abs=1e-6)

    def test_bad_marks(self):
        with pytest.raises(ValueError):
            monomer_from_marked_smiles("CC[*:1]")
        with pytest.raises(ValueError):
            monomer_from_marked_smiles("C([*:1])C[*:1]")

    def test_enantiomers_have_opposite_handedness(self):
        from flowermd.internal.stereochemistry import normalized_volume

        vols = []
        for smiles in (
            "c1ccc([C@H](C[*:2])[*:1])cc1",
            "c1ccc([C@@H](C[*:2])[*:1])cc1",
        ):
            comp, _ = monomer_from_marked_smiles(smiles)
            parts = list(comp.particles())
            center = 4  # the [C@H] atom in SMILES order
            neighbors = sorted(
                i
                for i, p in enumerate(parts)
                if p in parts[center].direct_bonds()
            )
            assert len(neighbors) == 4
            xyz = np.asarray(comp.xyz) * 10.0
            vols.append(normalized_volume(xyz[neighbors] - xyz[center]))
        assert abs(vols[0]) > 0.3
        assert np.sign(vols[0]) == -np.sign(vols[1])


class TestColinaPolymers(BaseTest):
    @pytest.mark.parametrize("cls", [PET, Polycarbonate, PEI])
    def test_linear_presets_build(self, cls):
        pol = cls(lengths=3, num_mols=2)
        chain = pol.molecules[0]
        assert len(pol.molecules) == 2
        # n repeats (monomer minus its two attachment H) plus two caps
        n_repeat_atoms = (
            monomer_from_marked_smiles(cls.smiles)[0].n_particles - 2
        )
        assert chain.n_particles == 3 * n_repeat_atoms + 2
        assert len(list(chain.children)) == 3
        assert cls.reference_density > 1.0
        assert isinstance(pol, MarkedSmilesPolymer)

    @pytest.mark.parametrize("cls", [PolyStyrene, PMMA])
    def test_tactic_presets_have_stereocenters(self, cls):
        pol = cls(lengths=4, num_mols=1, tacticity="atactic", seed=3)
        chain = pol.molecules[0]
        chain.box = None
        ref = capture_stereochemistry(chain)
        # the head-end repeat is capped with H, which makes its backbone
        # carbon achiral, so a chain of n repeats has n - 1 stereocenters
        assert len(ref.centers) == 3
        assert pol.tacticity == "atactic"

    def test_tacticity_sequences(self):
        iso = PolyStyrene(lengths=6, num_mols=1, tacticity="isotactic")
        syn = PolyStyrene(lengths=6, num_mols=1, tacticity="syndiotactic")
        assert iso.sequence == "A" and syn.sequence == "AB"
        assert iso.molecules[0].n_particles == syn.molecules[0].n_particles
        assert iso.molecules[0].name == "ps_6mer_AAAAAA"
        assert syn.molecules[0].name == "ps_6mer_ABABAB"
        assert len(list(syn.molecules[0].children)) == 6
        with pytest.raises(ValueError):
            PolyStyrene(lengths=2, num_mols=1, tacticity="random")

    def test_atactic_seed_changes_sequence(self):
        a = PolyStyrene(lengths=12, num_mols=1, tacticity="atactic", seed=1)
        b = PolyStyrene(lengths=12, num_mols=1, tacticity="atactic", seed=2)
        assert (
            a.molecules[0].name != b.molecules[0].name
        )  # name encodes the sequence

    def test_polystyrene_melt_end_to_end(self):
        chains = PolyStyrene(lengths=3, num_mols=3, tacticity="atactic", seed=5)
        system = AllAtomLattice(
            molecules=chains,
            density=PolyStyrene.reference_density * u.g / u.cm**3,
        )
        ff = AllAtomDPD(system.system)
        assert ff.stereo_centers == 3 * 2
        assert "stereochemistry" in ff.forces_by_role
        assert ff.frame.particles.N == system.system.n_particles
