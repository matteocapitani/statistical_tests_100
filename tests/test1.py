import pytest
import numpy as np
from src.statistical_tests_100.statistical_tests_100 import z_test_population_mean
from src.statistical_tests_100.statistical_tests_100 import get_gaussian_p_value


def test_mu0_input():
    assert z_test_population_mean(1, 1, [1, 1]) == 0.0


def test_sigma_input():
    assert z_test_population_mean(1, 1, [1, 1]) == 0.0


def test_sample_input():
    assert z_test_population_mean(1, 1, [1, 1]) == 0.0


def test_mu0_invalid():
    mu0 = "a"

    with pytest.raises(Exception) as e:
        _ = z_test_population_mean(mu0, 0, [1, 2])

    print(e.value)
    assert f"Invalid input for the average" in str(e.value)
    assert e.type == TypeError


def test_mu0_nan():
    mu0 = np.nan

    with pytest.raises(Exception) as e:
        _ = z_test_population_mean(mu0, 0, [1, 2])

    assert f"Invalid input for the average" in str(e.value)
    assert e.type == TypeError


def test_sigma_invalid():
    sigma = "a"

    with pytest.raises(Exception) as e:
        _ = z_test_population_mean(0, sigma, [1, 2])

    assert f"Invalid input for the standard deviation" in str(e.value)
    assert e.type == TypeError


def test_sigma_nan():
    sigma = np.nan

    with pytest.raises(Exception) as e:
        _ = z_test_population_mean(0, sigma, [1, 2])

    assert f"Invalid input for the standard deviation" in str(e.value)
    assert e.type == TypeError


def test_sigma_positive():
    sigma = -1

    with pytest.raises(Exception) as e:
        _ = z_test_population_mean(0, sigma, [1, 2])

    assert f"Invalid input for the standard deviation" in str(e.value)
    assert e.type == TypeError


def test_sample_size():
    sample = []

    with pytest.raises(Exception) as e:
        _ = z_test_population_mean(0, 1, sample)

    assert f"Empty sample" in str(e.value)
    assert e.type == ValueError


def test_sigma_1():
    sample = [2]
    assert z_test_population_mean(1, 1, sample) == 1.0


def test_sigma_2():
    sample = [2]
    assert z_test_population_mean(1, 2, sample) == 0.5


def test_statistics_not_string():
    Z = "a"
    tail = "left"

    with pytest.raises(Exception) as e:
        _ = get_gaussian_p_value(Z, tail)

    assert f"Invalid statistics value" in str(e.value)
    assert e.type == TypeError


def test_tail_is_string():
    Z = 1.96
    tail = 1

    with pytest.raises(Exception) as e:
        _ = get_gaussian_p_value(Z, tail)

    assert f"Invalid value for tail" in str(e.value)
    assert e.type == TypeError


def test_tail_left():
    Z = 0
    tail = "left"
    p = get_gaussian_p_value(Z, tail)
    assert p == 0.5


def test_tail_right():
    Z = 0
    tail = "right"
    p = get_gaussian_p_value(Z, tail)
    assert p == 0.5


def test_tail_two():
    Z = 1.96
    tail = "two"
    p = get_gaussian_p_value(Z, tail)
    assert pytest.approx(p, 0.005) == 0.05


def test_tail_right_005():
    Z = 1.64
    tail = "right"
    p = get_gaussian_p_value(Z, tail)
    assert pytest.approx(p, 0.01) == 0.05


def test_tail_left_005():
    Z = -1.64
    tail = "left"
    p = get_gaussian_p_value(Z, tail)
    assert pytest.approx(p, 0.01) == 0.95


def test_proper_value_tail():
    Z = 1.96
    tail = "wrong"

    with pytest.raises(Exception) as e:
        _ = get_gaussian_p_value(Z, tail)

    assert f"Invalid value for tail" in str(e.value)
    assert e.type == ValueError


def test_sample_no_none():
    with pytest.raises(Exception) as e:
        z = z_test_population_mean(0, 1, [1, np.nan])

    assert f"Invalid value in the sample" in str(e.value)
    assert e.type == TypeError


def test_sample_no_str():
    with pytest.raises(Exception) as e:
        z = z_test_population_mean(0, 1, [1, "3"])

    assert f"Invalid value in the sample" in str(e.value)
    assert e.type == TypeError


def test_Z_formula():
    mu0 = 0
    sigma = 1

    assert (
        pytest.approx(z_test_population_mean(mu0, 1, [1.96, 1.96, 1.96, 1.96]), 0.01)
        == 1.96 * 2
    )
