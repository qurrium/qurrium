"""
================================================================
Test the qurry.qurrent module EntropyMeasure class.
================================================================

"""

import pytest
import numpy as np
from qurry.qurrent import EntropyMeasure
from qurry.tools.backend import GeneralSimulator
from qurry.capsule import mori, hoshi
from qurry.recipe import TrivialParamagnet, GHZ, TopologicalParamagnet

tag_list = mori.TagList()
statesheet = hoshi.Hoshi()

exp_method_01 = EntropyMeasure(method="hadamard")
exp_method_02 = EntropyMeasure(method="randomized")


wave_adds_01 = []
wave_adds_02 = []

for i in range(4, 7, 2):
    wave_adds_01.append(exp_method_01.add(TrivialParamagnet(i), f"{i}-trivial"))
    wave_adds_02.append(exp_method_02.add(TrivialParamagnet(i), f"{i}-trivial"))

    wave_adds_01.append(exp_method_01.add(GHZ(i), f"{i}-GHZ"))
    wave_adds_02.append(exp_method_02.add(GHZ(i), f"{i}-GHZ"))

    wave_adds_01.append(exp_method_01.add(TopologicalParamagnet(i), f"{i}-topological"))
    wave_adds_02.append(exp_method_02.add(TopologicalParamagnet(i), f"{i}-topological"))

backend = GeneralSimulator()
# backend = BasicAer.backends()[0]
print(backend.configuration())


@pytest.mark.parametrize("tgt", wave_adds_01)
def test_quantity_01(tgt):
    """Test the quantity of entropy and purity.

    Args:
        tgt (Hashable): The target wave key in Qurry.
    """

    exp_id = exp_method_01.measure(wave=tgt, backend=backend)
    exp_method_01.exps[exp_id].analyze()
    quantity = exp_method_01.exps[exp_id].reports[0].content._asdict()
    assert all(
        ["entropy" in quantity, "purity" in quantity]
    ), f"The necessary quantities 'entropy', 'purity' are not found: {quantity.keys()}."
    assert (
        np.abs(quantity["purity"] - 1.0) < 1e-0
    ), f"The hadamard test result is wrong: {np.abs(quantity['purity'] - 1.0)} !<= < 1e-0."


@pytest.mark.parametrize("tgt", wave_adds_02)
def test_quantity_02(tgt):
    """Test the quantity of entropy and purity.

    Args:
        tgt (Hashable): The target wave key in Qurry.
    """

    exp_id = exp_method_02.measure(wave=tgt, times=10, backend=backend)
    analysis_01 = exp_method_02.exps[exp_id].analyze(2)
    quantity_01 = analysis_01.content._asdict()
    analysis_02 = exp_method_02.exps[exp_id].analyze(2, counts_used=range(5))
    quantity_02 = analysis_02.content._asdict()
    assert all(
        ["entropy" in quantity_01, "purity" in quantity_01]
    ), f"The necessary quantities 'entropy', 'purity' are not found: {quantity_01.keys()}."
    assert quantity_02["entropyAllSys"] != quantity_01["entropyAllSys"], (
        "The all system entropy is not changed: "
        + f"counts_used: {quantity_01['counts_used']}: {quantity_02['entropyAllSys']}, "
        + f"counts_used: {quantity_02['counts_used']}: {quantity_02['entropyAllSys']},"
    )
