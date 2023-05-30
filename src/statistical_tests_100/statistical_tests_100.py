"""
This module contains statistical tests, as found in

Gopal K Kanji, 100 Statistical Tests, Sage Publications, Third Edition (2006)
 
Functions:
    not_valid_float(float) -> float
    z_test_population_mean(float, float, list) -> float
    get_gaussian_p_value(float) -> float
"""
import math


def not_valid_float(sample_element: float) -> bool:
    """
    Tests whether a sample element is a valid float

    Arguments:
        sample_element (float): The input sample element

    Returns:
        bool: Is the input an invalid float?
    """

    if isinstance(sample_element, str):
        return True
    if math.isnan(sample_element):
        return True
    return False


def z_test_population_mean(mu0: float, sigma: float, sample: list) -> float:
    """
    Calculate the Z statistics

    Arguments:
        mu0 (float): Expected average
        sigma (float): Expected standard deviation
        sample (list): A list containing the sample elements

    Returns:
        float: The Z-statistics
    """

    if not_valid_float(mu0):
        raise TypeError(f"Invalid input for the average: {mu0}")

    if not_valid_float(sigma):
        raise TypeError(f"Invalid input for the standard deviation: {sigma}")

    if sigma <= 0.0:
        raise TypeError(f"Invalid input for the standard deviation: {sigma}")

    if not sample:
        raise ValueError("Empty sample")

    if any((not_valid_float(i) for i in sample)):
        issues = list((i for i in sample if not_valid_float(i)))
        raise TypeError(f"Invalid value in the sample: {issues}")

    average = sum(sample) / len(sample)
    z_statistics = (average - mu0) / (sigma / math.sqrt(len(sample)))

    return z_statistics


def get_gaussian_p_value(z_statistics: float, tail: str) -> float:
    """
    Tests whether a sample element is a valid float

    Arguments:
        sample_element (float): The input sample element

    Returns:
        bool: Is the input an invalid float?

    Raises:
        Invalid input for the average: the average is not a valid float
        Invalid input for the standard deviation: the standard deviation is not a valid float
        Invalid input for the standard deviation: the standard deviation is zero
        Empty sample: if there are no elements in the sample
        Invalid value in the sample: there are invalid elements in the sample
    """

    if not_valid_float(z_statistics):
        raise TypeError(f"Invalid statistics value: {z_statistics}")

    if not isinstance(tail, str):
        raise TypeError(f"Invalid value for tail: {tail}")

    if not tail in ["left", "right", "two"]:
        raise ValueError(f"Invalid value for tail: {tail}")

    if tail == "two":
        return 1 - math.erf(z_statistics / math.sqrt(2))

    return (1 - math.erf(z_statistics / math.sqrt(2))) * 0.5
