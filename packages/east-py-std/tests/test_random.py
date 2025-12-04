"""Tests for random platform functions."""

import pytest

from east_py_std.random import (
    _reset_to_crypto,
    random_bates_impl,
    random_bernoulli_impl,
    random_binomial_impl,
    random_exponential_impl,
    random_geometric_impl,
    random_irwin_hall_impl,
    random_log_normal_impl,
    random_normal_impl,
    random_pareto_impl,
    random_poisson_impl,
    random_range_impl,
    random_seed_impl,
    random_uniform_impl,
    random_weibull_impl,
)


@pytest.fixture(autouse=True)
def reset_rng():
    """Reset RNG to crypto mode before each test to avoid test pollution."""
    _reset_to_crypto()
    yield
    _reset_to_crypto()


def test_random_uniform():
    """Test uniform random returns value in [0, 1)."""
    for _ in range(100):
        result = random_uniform_impl()
        assert 0.0 <= result < 1.0


def test_random_uniform_distribution():
    """Test uniform random has reasonable distribution."""
    samples = [random_uniform_impl() for _ in range(1000)]
    mean = sum(samples) / len(samples)
    # Mean of uniform[0,1) should be around 0.5
    assert 0.45 < mean < 0.55


def test_random_normal():
    """Test normal random generates values."""
    for _ in range(100):
        result = random_normal_impl()
        assert isinstance(result, float)
        # Most values should be within ±4 standard deviations
        assert -10.0 < result < 10.0


def test_random_normal_distribution():
    """Test normal random has reasonable distribution."""
    samples = [random_normal_impl() for _ in range(1000)]
    mean = sum(samples) / len(samples)
    # Mean of N(0,1) should be around 0
    assert -0.2 < mean < 0.2


def test_random_range():
    """Test random_range returns value in [min, max]."""
    for _ in range(100):
        result = random_range_impl(10, 20)
        assert 10 <= result <= 20


def test_random_range_single_value():
    """Test random_range with min == max."""
    result = random_range_impl(5, 5)
    assert result == 5


def test_random_range_invalid():
    """Test random_range rejects min > max."""
    with pytest.raises(ValueError):
        random_range_impl(10, 5)


def test_random_exponential():
    """Test exponential distribution."""
    for _ in range(100):
        result = random_exponential_impl(1.0)
        assert result >= 0.0


def test_random_exponential_invalid():
    """Test exponential rejects non-positive lambda."""
    with pytest.raises(ValueError):
        random_exponential_impl(0.0)
    with pytest.raises(ValueError):
        random_exponential_impl(-1.0)


def test_random_weibull():
    """Test Weibull distribution."""
    for _ in range(100):
        result = random_weibull_impl(2.0)
        assert result >= 0.0


def test_random_weibull_invalid():
    """Test Weibull rejects non-positive shape."""
    with pytest.raises(ValueError):
        random_weibull_impl(0.0)
    with pytest.raises(ValueError):
        random_weibull_impl(-1.0)


def test_random_bernoulli():
    """Test Bernoulli trial returns 0 or 1."""
    for _ in range(100):
        result = random_bernoulli_impl(0.5)
        assert result in (0, 1)


def test_random_bernoulli_probability_zero():
    """Test Bernoulli with p=0 always returns 0."""
    for _ in range(10):
        result = random_bernoulli_impl(0.0)
        assert result == 0


def test_random_bernoulli_probability_one():
    """Test Bernoulli with p=1 always returns 1."""
    for _ in range(10):
        result = random_bernoulli_impl(1.0)
        assert result == 1


def test_random_bernoulli_invalid():
    """Test Bernoulli rejects invalid probability."""
    with pytest.raises(ValueError):
        random_bernoulli_impl(-0.1)
    with pytest.raises(ValueError):
        random_bernoulli_impl(1.1)


def test_random_binomial():
    """Test binomial distribution."""
    result = random_binomial_impl(10, 0.5)
    assert 0 <= result <= 10


def test_random_binomial_zero_trials():
    """Test binomial with n=0."""
    result = random_binomial_impl(0, 0.5)
    assert result == 0


def test_random_binomial_invalid():
    """Test binomial rejects invalid parameters."""
    with pytest.raises(ValueError):
        random_binomial_impl(-1, 0.5)
    with pytest.raises(ValueError):
        random_binomial_impl(10, -0.1)
    with pytest.raises(ValueError):
        random_binomial_impl(10, 1.1)


def test_random_geometric():
    """Test geometric distribution."""
    for _ in range(100):
        result = random_geometric_impl(0.5)
        assert result >= 1


def test_random_geometric_invalid():
    """Test geometric rejects invalid probability."""
    with pytest.raises(ValueError):
        random_geometric_impl(0.0)
    with pytest.raises(ValueError):
        random_geometric_impl(1.1)


def test_random_poisson():
    """Test Poisson distribution."""
    for _ in range(100):
        result = random_poisson_impl(5.0)
        assert result >= 0


def test_random_poisson_zero_lambda():
    """Test Poisson with lambda=0."""
    result = random_poisson_impl(0.0)
    assert result == 0


def test_random_poisson_invalid():
    """Test Poisson rejects negative lambda."""
    with pytest.raises(ValueError):
        random_poisson_impl(-1.0)


def test_random_pareto():
    """Test Pareto distribution."""
    for _ in range(100):
        result = random_pareto_impl(2.0)
        assert result >= 1.0


def test_random_pareto_invalid():
    """Test Pareto rejects non-positive alpha."""
    with pytest.raises(ValueError):
        random_pareto_impl(0.0)
    with pytest.raises(ValueError):
        random_pareto_impl(-1.0)


def test_random_log_normal():
    """Test log-normal distribution."""
    for _ in range(100):
        result = random_log_normal_impl(0.0, 1.0)
        assert result > 0.0  # Log-normal is always positive


def test_random_log_normal_invalid():
    """Test log-normal rejects non-positive sigma."""
    with pytest.raises(ValueError):
        random_log_normal_impl(0.0, 0.0)
    with pytest.raises(ValueError):
        random_log_normal_impl(0.0, -1.0)


def test_random_irwin_hall():
    """Test Irwin-Hall distribution."""
    result = random_irwin_hall_impl(5)
    assert 0.0 <= result <= 5.0


def test_random_irwin_hall_invalid():
    """Test Irwin-Hall rejects non-positive n."""
    with pytest.raises(ValueError):
        random_irwin_hall_impl(0)
    with pytest.raises(ValueError):
        random_irwin_hall_impl(-1)


def test_random_bates():
    """Test Bates distribution."""
    result = random_bates_impl(5)
    assert 0.0 <= result <= 1.0


def test_random_bates_invalid():
    """Test Bates rejects non-positive n."""
    with pytest.raises(ValueError):
        random_bates_impl(0)
    with pytest.raises(ValueError):
        random_bates_impl(-1)


def test_random_seed():
    """Test random_seed enables reproducible sequences."""
    # Seed and generate a sequence
    random_seed_impl(42)
    seq1 = [random_uniform_impl() for _ in range(10)]

    # Seed with same value and generate again
    random_seed_impl(42)
    seq2 = [random_uniform_impl() for _ in range(10)]

    # Sequences should be identical
    assert seq1 == seq2


def test_random_seed_different_seeds():
    """Test different seeds produce different sequences."""
    random_seed_impl(42)
    seq1 = [random_uniform_impl() for _ in range(10)]

    random_seed_impl(123)
    seq2 = [random_uniform_impl() for _ in range(10)]

    # Sequences should be different
    assert seq1 != seq2


def test_random_seed_reproducible_distributions():
    """Test seeding works across all distribution functions."""
    random_seed_impl(12345)

    # Generate one value from each distribution
    results1 = [
        random_uniform_impl(),
        random_normal_impl(),
        random_range_impl(1, 100),
        random_exponential_impl(1.0),
        random_bernoulli_impl(0.5),
        random_binomial_impl(10, 0.5),
        random_geometric_impl(0.5),
        random_poisson_impl(5.0),
    ]

    # Reset and generate again
    random_seed_impl(12345)

    results2 = [
        random_uniform_impl(),
        random_normal_impl(),
        random_range_impl(1, 100),
        random_exponential_impl(1.0),
        random_bernoulli_impl(0.5),
        random_binomial_impl(10, 0.5),
        random_geometric_impl(0.5),
        random_poisson_impl(5.0),
    ]

    # All results should be identical
    assert results1 == results2
