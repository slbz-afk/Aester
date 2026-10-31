"""Exercise the documented negative-range spelling through the actual CLI parser."""
import json
import sys

import pytest
from starstack import cli, starless_runtime


@pytest.mark.parametrize("range_value", ["-1..1", "0..1"])
def test_documented_range_reaches_backend(monkeypatch, capsys, range_value):
    calls = []

    def backend(*args, **kwargs):
        calls.append((args, kwargs))
        return {"state": "succeeded"}

    monkeypatch.setattr(starless_runtime, "starless", backend)
    monkeypatch.setattr(sys, "argv", ["starstack", "starless", "--source", "source.tif",
        "--sky-mask", "sky.tif", "--protected-mask", "protect.png", "--model", "model.onnx",
        "--value-range=" + range_value, "--threads", "4", "--channel-order", "bgr", "--stitch", "crop"])
    assert cli.main() == 0
    assert len(calls) == 1
    assert calls[0][1]["value_range"] == range_value
    assert calls[0][1]["intra_op_threads"] == 4
    assert calls[0][1]["channel_order"] == "bgr"
    assert json.loads(capsys.readouterr().out)["state"] == "succeeded"
