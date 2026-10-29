import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score
from sesgo import agreement, design


def test_cohen_matches_sklearn():
    rng = np.random.RandomState(0)
    a = rng.choice(list("ABC"), 200).tolist(); b = [x if rng.rand() < .7 else rng.choice(list("ABC")) for x in a]
    assert abs(agreement.cohen_kappa(a, b) - cohen_kappa_score(a, b)) < 1e-12


def test_fleiss_matches_statsmodels():
    sm_fleiss = pytest.importorskip("statsmodels.stats.inter_rater").fleiss_kappa
    rng = np.random.RandomState(1)
    t = np.array([np.bincount(rng.randint(0, 3, 3), minlength=3) for _ in range(60)])
    assert abs(agreement.fleiss_kappa(t) - sm_fleiss(t)) < 1e-12


def test_interpretation_scale():
    assert agreement.interpret_kappa(0.35) == "aceptable"
    assert agreement.interpret_kappa(0.65) == "sustancial"


def test_sample_size_known_value():
    # alpha=.05 bilateral, potencia .80, p1=.70, p2=.62 -> 547 (calculo manual: 7.849*0.4456/0.0064)
    assert design.n_per_group(0.70, 0.08) == 547
