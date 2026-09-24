import numpy as np
import pytest

from flowermd.utils import schulz_zimm_lengths


class TestSchulzZimm:
    def test_monodisperse(self):
        lengths, num_mols = schulz_zimm_lengths(50, pdi=1.0, n_chains=7)
        assert lengths == [50] and num_mols == [7]

    def test_moments(self):
        lengths, num_mols = schulz_zimm_lengths(
            mean_length=100, pdi=1.5, n_chains=20000, seed=3
        )
        n = np.array(lengths, dtype=float)
        w = np.array(num_mols, dtype=float)
        mn = (n * w).sum() / w.sum()
        mw = (n * n * w).sum() / (n * w).sum()
        assert mn == pytest.approx(100, rel=0.02)
        assert mw / mn == pytest.approx(1.5, rel=0.03)
        assert w.sum() == 20000

    def test_seeded_and_min_length(self):
        a = schulz_zimm_lengths(10, 2.0, 50, seed=1)
        b = schulz_zimm_lengths(10, 2.0, 50, seed=1)
        c = schulz_zimm_lengths(10, 2.0, 50, seed=2, min_length=3)
        assert a == b and a != c
        assert min(c[0]) >= 3

    def test_bad_arguments(self):
        with pytest.raises(ValueError):
            schulz_zimm_lengths(0, 1.2, 5)
        with pytest.raises(ValueError):
            schulz_zimm_lengths(10, 0.9, 5)
        with pytest.raises(ValueError):
            schulz_zimm_lengths(10, 1.2, 0)
