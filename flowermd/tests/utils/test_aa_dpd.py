import math
import sys

import pytest

from flowermd.internal.aa_dpd import (
    dpd_pair_parameters,
    epsilon_scaled_dpd_parameters,
)


class TestDPDPairParameters:
    @pytest.mark.parametrize("repulsion,gamma", [(1250, 200), (0, 200), (0, 0)])
    def test_uniform_without_epsilon_dependency(
        self, monkeypatch, repulsion, gamma
    ):
        def forbidden(*args, **kwargs):
            pytest.fail("uniform pairs must not use epsilon weighting")

        monkeypatch.setattr(
            "flowermd.internal.aa_dpd.epsilon_scaled_dpd_parameters", forbidden
        )
        names = ["z", "a"]
        pairs = dpd_pair_parameters(
            names, repulsion, gamma, epsilon_weighting=False
        )
        assert list(pairs) == [("z", "z"), ("z", "a"), ("a", "a")]
        assert all(
            values == {"A": repulsion, "gamma": gamma}
            for values in pairs.values()
        )
        assert names == ["z", "a"]
        pairs["z", "z"]["A"] = -1
        assert pairs["z", "a"]["A"] == repulsion

    @pytest.mark.parametrize("reference", [None, 0.5])
    def test_weighted_values_and_type_order(self, reference):
        epsilons = {"a": 1.0, "z": 0.25}
        original = epsilons.copy()
        pairs = dpd_pair_parameters(
            ("z", "a"),
            40,
            20,
            particle_epsilons=epsilons,
            epsilon_reference=reference,
        )
        assert list(pairs) == [("z", "z"), ("z", "a"), ("a", "a")]
        denominator = 1.0 if reference is None else reference
        for pair, numerator in zip(pairs, [0.25, 0.5, 1.0]):
            assert pairs[pair] == pytest.approx(
                {
                    "A": 40 * numerator / denominator,
                    "gamma": 20 * numerator / denominator,
                }
            )
        assert epsilons == original

    def test_single_type_uniform(self):
        assert dpd_pair_parameters(
            ["C"], 1250, 200, epsilon_weighting=False
        ) == {("C", "C"): {"A": 1250, "gamma": 200}}

    @pytest.mark.parametrize(
        "names", [None, [], (), "C", b"C", {"C"}, {"C": 1}]
    )
    def test_reject_unordered_or_empty_types(self, names):
        with pytest.raises(ValueError, match="ordered sequence"):
            dpd_pair_parameters(names, 40, 20, epsilon_weighting=False)

    @pytest.mark.parametrize("names", [[""], [1], [None], [["C"]], ["C", "C"]])
    def test_invalid_names(self, names):
        with pytest.raises(ValueError, match="type names"):
            dpd_pair_parameters(names, 40, 20, epsilon_weighting=False)

    @pytest.mark.parametrize("switch", [None, 0, 1, "false", [], {}])
    def test_invalid_switch(self, switch):
        with pytest.raises(ValueError, match="epsilon_weighting"):
            dpd_pair_parameters(["C"], 40, 20, epsilon_weighting=switch)

    @pytest.mark.parametrize("name", ["repulsion", "gamma"])
    @pytest.mark.parametrize("value", [-1, math.inf, math.nan, True, "1", None])
    def test_uniform_invalid_coefficients(self, name, value):
        kwargs = {"repulsion": 40, "gamma": 20, name: value}
        with pytest.raises(ValueError, match=name):
            dpd_pair_parameters(["C"], epsilon_weighting=False, **kwargs)

    @pytest.mark.parametrize(
        "epsilons", [None, [], {}, {"a": 1}, {"a": 1, "b": 2, "c": 3}]
    )
    def test_missing_or_mismatched_epsilons(self, epsilons):
        with pytest.raises(ValueError, match="mapping|exactly match"):
            dpd_pair_parameters(["a", "b"], 40, 20, particle_epsilons=epsilons)

    @pytest.mark.parametrize("value", [-1, 0, math.nan])
    def test_weighted_value_validation(self, value):
        with pytest.raises(ValueError, match="epsilon for a"):
            dpd_pair_parameters(["a"], 40, 20, particle_epsilons={"a": value})

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"particle_epsilons": {}},
            {"particle_epsilons": {"a": 1}},
            {"epsilon_reference": 1},
            {"epsilon_reference": 0},
        ],
    )
    def test_uniform_rejects_epsilon_arguments(self, kwargs):
        with pytest.raises(ValueError, match="omit epsilon arguments"):
            dpd_pair_parameters(
                ["a"], 40, 20, epsilon_weighting=False, **kwargs
            )


class TestEpsilonScaledDPDParameters:
    def test_pair_factors_order_and_input_preservation(self):
        epsilons = {"weak": 0.25, "strong": 1.0}
        original = epsilons.copy()
        pairs = epsilon_scaled_dpd_parameters(epsilons, 1250, 200)
        assert list(pairs) == [
            ("weak", "weak"),
            ("weak", "strong"),
            ("strong", "strong"),
        ]
        for values, factor in zip(pairs.values(), [0.25, 0.5, 1.0]):
            assert values == {"A": 1250 * factor, "gamma": 200 * factor}
        assert epsilons == original

    def test_equal_epsilons(self):
        pairs = epsilon_scaled_dpd_parameters({"a": 0.3, "b": 0.3}, 40, 20)
        for values in pairs.values():
            assert values == pytest.approx({"A": 40, "gamma": 20})

    @pytest.mark.parametrize("reference", [None, 0.5])
    @pytest.mark.parametrize("unit_factor", [1e-200, 4.184, 1e200])
    def test_energy_unit_invariance(self, reference, unit_factor):
        epsilons = {"a": 0.25, "b": 1.0}
        expected = epsilon_scaled_dpd_parameters(epsilons, 1250, 200, reference)
        scaled = epsilon_scaled_dpd_parameters(
            {key: value * unit_factor for key, value in epsilons.items()},
            1250,
            200,
            None if reference is None else reference * unit_factor,
        )
        for pair in expected:
            assert scaled[pair] == pytest.approx(expected[pair])

    def test_explicit_reference(self):
        pairs = epsilon_scaled_dpd_parameters({"a": 0.25, "b": 1}, 40, 20, 0.5)
        assert pairs["a", "b"] == {"A": 40, "gamma": 20}
        assert pairs["b", "b"] == {"A": 80, "gamma": 40}

    def test_frozen_protocol_algebra(self):
        # Frozen fastfire._forces: UFF epsilons, including an explicit
        # historical override reference rather than max(effective epsilons).
        epsilons = {"C_3": 0.105, "H_": 0.044, "O_3": 0.06}
        reference = 0.06
        pairs = epsilon_scaled_dpd_parameters(epsilons, 1250, 200, reference)
        assert len(pairs) == 6
        for (first, second), values in pairs.items():
            factor = math.sqrt(epsilons[first] * epsilons[second]) / reference
            assert values == pytest.approx(
                {"A": 1250 * factor, "gamma": 200 * factor}
            )

    @pytest.mark.parametrize("value", [1e-300, 1e300, sys.float_info.max])
    def test_extreme_finite_epsilons(self, value):
        pairs = epsilon_scaled_dpd_parameters({"a": value}, 1250, 200)
        assert pairs["a", "a"] == pytest.approx({"A": 1250, "gamma": 200})

    def test_representable_output_with_extreme_reference(self):
        pairs = epsilon_scaled_dpd_parameters({"a": 1e300}, 1e-300, 0, 1e-300)
        assert pairs["a", "a"] == pytest.approx({"A": 1e300, "gamma": 0})

    @pytest.mark.parametrize("repulsion,gamma", [(0, 20), (40, 0), (0, 0)])
    def test_zero_coefficients(self, repulsion, gamma):
        pairs = epsilon_scaled_dpd_parameters({"a": 1}, repulsion, gamma)
        assert pairs["a", "a"] == {"A": repulsion, "gamma": gamma}

    @pytest.mark.parametrize("mapping", [{}, [], None, {1: 1}, {"": 1}])
    def test_invalid_mapping(self, mapping):
        with pytest.raises(ValueError, match="mapping|type names"):
            epsilon_scaled_dpd_parameters(mapping, 40, 20)

    @pytest.mark.parametrize(
        "value", [0, -1, math.inf, -math.inf, math.nan, "1", None, True]
    )
    def test_invalid_epsilon(self, value):
        with pytest.raises(ValueError, match="epsilon for a"):
            epsilon_scaled_dpd_parameters({"a": value}, 40, 20)

    @pytest.mark.parametrize(
        "value", [0, -1, math.inf, -math.inf, math.nan, "1", True]
    )
    def test_invalid_reference(self, value):
        with pytest.raises(ValueError, match="epsilon_reference"):
            epsilon_scaled_dpd_parameters({"a": 1}, 40, 20, value)

    @pytest.mark.parametrize("name", ["repulsion", "gamma"])
    @pytest.mark.parametrize(
        "value", [-1, math.inf, -math.inf, math.nan, "1", None, True]
    )
    def test_invalid_coefficient(self, name, value):
        kwargs = {"repulsion": 40, "gamma": 20}
        kwargs[name] = value
        with pytest.raises(ValueError, match=name):
            epsilon_scaled_dpd_parameters({"a": 1}, **kwargs)

    @pytest.mark.parametrize(
        "epsilon,reference", [(1e300, 1e-300), (1e-300, 1e300)]
    )
    def test_unrepresentable_outputs(self, epsilon, reference):
        with pytest.raises(ValueError, match="float range"):
            epsilon_scaled_dpd_parameters({"a": epsilon}, 1, 1, reference)
