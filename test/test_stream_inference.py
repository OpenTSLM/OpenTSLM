# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

from queue import Queue
from types import SimpleNamespace

import pytest
import torch

import opentslm.model.llm.OpenTSLMFlamingo as flamingo_module
import opentslm.model.llm.OpenTSLMSP as sp_module
from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP


class FakeStreamer:
    instances = []

    def __init__(self, tokenizer, skip_prompt=False, timeout=None, **decode_kwargs):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.timeout = timeout
        self.decode_kwargs = decode_kwargs
        self._queue = Queue()
        FakeStreamer.instances.append(self)

    def put(self, text):
        self._queue.put(text)

    def end(self):
        self._queue.put(None)

    def __iter__(self):
        while True:
            item = self._queue.get()
            if item is None:
                break
            yield item


class FakePrompt:
    def __init__(self, payload):
        self.payload = payload

    def to_dict(self):
        return self.payload


def build_sp_model():
    model = OpenTSLMSP.__new__(OpenTSLMSP)
    torch.nn.Module.__init__(model)
    model.device = "cpu"
    model.tokenizer = object()
    model.pad_and_apply_batch = lambda batch: (
        torch.ones((1, 2, 3)),
        torch.ones((1, 2), dtype=torch.long),
    )
    return model


def build_flamingo_model():
    model = OpenTSLMFlamingo.__new__(OpenTSLMFlamingo)
    torch.nn.Module.__init__(model)
    model.device = "cpu"
    model.text_tokenizer = SimpleNamespace(eos_token_id=7, pad_token_id=0)
    model.pad_and_apply_batch = lambda batch, include_labels=True: (
        torch.tensor([[1, 2, 3]], dtype=torch.long),
        torch.ones((1, 1, 1, 4), dtype=torch.float32),
        torch.ones((1, 3), dtype=torch.long),
        None,
    )
    model._build_input_embeddings = lambda input_ids: torch.ones(
        (1, input_ids.shape[1], 4), dtype=torch.float32
    )
    model._condition_media_locations = lambda input_ids: None
    return model


def test_sp_stream_generate_yields_chunks_and_skips_prompt(monkeypatch):
    FakeStreamer.instances.clear()
    monkeypatch.setattr(sp_module, "TextIteratorStreamer", FakeStreamer)

    model = build_sp_model()

    class FakeLLM:
        def generate(self, **kwargs):
            streamer = kwargs["streamer"]
            streamer.put("Hel")
            streamer.put("")
            streamer.put("lo")
            streamer.end()

    model.llm = FakeLLM()

    chunks = list(model.stream_generate([{"sample": 1}], max_new_tokens=12))

    assert chunks == ["Hel", "lo"]
    assert "".join(chunks) == "Hello"
    assert FakeStreamer.instances[0].skip_prompt is True


def test_sp_stream_generate_surfaces_generation_errors(monkeypatch):
    monkeypatch.setattr(sp_module, "TextIteratorStreamer", FakeStreamer)

    model = build_sp_model()

    class FailingLLM:
        def generate(self, **kwargs):
            raise RuntimeError("generation failed")

    model.llm = FailingLLM()

    with pytest.raises(RuntimeError, match="generation failed"):
        list(model.stream_generate([{"sample": 1}]))


def test_stream_generate_rejects_batched_requests(monkeypatch):
    monkeypatch.setattr(sp_module, "TextIteratorStreamer", FakeStreamer)
    monkeypatch.setattr(flamingo_module, "TextIteratorStreamer", FakeStreamer)

    sp_model = build_sp_model()
    sp_model.llm = SimpleNamespace(generate=lambda **kwargs: None)

    flamingo_model = build_flamingo_model()
    flamingo_model.model = SimpleNamespace(
        _encode_vision_x=lambda vision_x: None,
        lang_encoder=SimpleNamespace(
            generate=lambda **kwargs: None,
            clear_conditioned_layers=lambda: None,
        ),
    )

    with pytest.raises(ValueError, match="exactly one sample"):
        list(sp_model.stream_generate([{"a": 1}, {"b": 2}]))

    with pytest.raises(ValueError, match="exactly one sample"):
        list(flamingo_model.stream_generate([{"a": 1}, {"b": 2}]))


def test_sp_stream_prompt_converts_prompt_and_forwards_kwargs(monkeypatch):
    model = build_sp_model()
    model.train()

    captured = {}

    def fake_extend(batch, normalize=False, patch_size=None):
        captured["extend"] = {
            "batch": batch,
            "normalize": normalize,
            "patch_size": patch_size,
        }
        return [{"prepared": True, **batch[0]}]

    def fake_stream_generate(batch, max_new_tokens=50, **generate_kwargs):
        captured["stream_generate"] = {
            "batch": batch,
            "max_new_tokens": max_new_tokens,
            "generate_kwargs": generate_kwargs,
        }
        yield "done"

    monkeypatch.setattr(
        sp_module,
        "extend_time_series_to_match_patch_size_and_aggregate",
        fake_extend,
    )
    model.stream_generate = fake_stream_generate

    chunks = list(
        model.stream_prompt(
            FakePrompt({"prompt": "value"}),
            max_new_tokens=9,
            normalize=True,
            temperature=0.2,
        )
    )

    assert chunks == ["done"]
    assert model.training is False
    assert captured["extend"]["batch"] == [{"prompt": "value"}]
    assert captured["extend"]["normalize"] is True
    assert captured["stream_generate"]["batch"] == [{"prepared": True, "prompt": "value"}]
    assert captured["stream_generate"]["max_new_tokens"] == 9
    assert captured["stream_generate"]["generate_kwargs"] == {"temperature": 0.2}


def test_flamingo_stream_generate_clears_conditioned_layers_on_success(monkeypatch):
    FakeStreamer.instances.clear()
    monkeypatch.setattr(flamingo_module, "TextIteratorStreamer", FakeStreamer)

    model = build_flamingo_model()
    clear_calls = []

    class FakeLangEncoder:
        def generate(self, **kwargs):
            streamer = kwargs["streamer"]
            streamer.put("A")
            streamer.put("B")
            streamer.end()

        def clear_conditioned_layers(self):
            clear_calls.append("cleared")

    model.model = SimpleNamespace(
        _encode_vision_x=lambda vision_x: None,
        lang_encoder=FakeLangEncoder(),
    )

    chunks = list(model.stream_generate([{"sample": 1}], max_new_tokens=5))

    assert chunks == ["A", "B"]
    assert clear_calls == ["cleared"]
    assert FakeStreamer.instances[0].skip_prompt is True


def test_flamingo_stream_generate_clears_conditioned_layers_on_error(monkeypatch):
    monkeypatch.setattr(flamingo_module, "TextIteratorStreamer", FakeStreamer)

    model = build_flamingo_model()
    clear_calls = []

    class FailingLangEncoder:
        def generate(self, **kwargs):
            raise RuntimeError("flamingo failure")

        def clear_conditioned_layers(self):
            clear_calls.append("cleared")

    model.model = SimpleNamespace(
        _encode_vision_x=lambda vision_x: None,
        lang_encoder=FailingLangEncoder(),
    )

    with pytest.raises(RuntimeError, match="flamingo failure"):
        list(model.stream_generate([{"sample": 1}]))

    assert clear_calls == ["cleared"]


def test_sp_init_disables_mean_resizing(monkeypatch):
    resize_calls = []

    class FakeTokenizer:
        pad_token = None
        eos_token = "</s>"

        def __len__(self):
            return 11

    class FakeLLM:
        config = SimpleNamespace(hidden_size=8)

        def resize_token_embeddings(self, size, **kwargs):
            resize_calls.append((size, kwargs))

        def parameters(self):
            return []

    monkeypatch.setattr(
        sp_module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    monkeypatch.setattr(
        sp_module.AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: FakeLLM(),
    )
    monkeypatch.setattr(
        sp_module,
        "TransformerCNNEncoder",
        lambda: torch.nn.Identity(),
    )
    monkeypatch.setattr(
        sp_module,
        "MLPProjector",
        lambda *args, **kwargs: torch.nn.Identity(),
    )

    OpenTSLMSP(llm_id="dummy", device="cpu")

    assert resize_calls == [(11, {"mean_resizing": False})]


def test_flamingo_init_disables_mean_resizing(monkeypatch):
    resize_calls = []

    class FakeTokenizer:
        pad_token = None

        def add_special_tokens(self, mapping):
            if "pad_token" in mapping:
                self.pad_token = mapping["pad_token"]

        def encode(self, value):
            return [1]

        def __len__(self):
            return 13

    class GemmaForCausalLM:
        config = SimpleNamespace(hidden_size=8)

        def resize_token_embeddings(self, size, **kwargs):
            resize_calls.append((size, kwargs))

        def set_decoder_layers_attr_name(self, value):
            self.decoder_layers_attr_name = value

    class FakeFlamingoModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.perceiver = torch.nn.Linear(1, 1)
            self.vision_encoder = torch.nn.Linear(1, 1)
            self.lang_encoder = SimpleNamespace(
                gated_cross_attn_layers=torch.nn.Linear(1, 1),
                get_input_embeddings=lambda: torch.nn.Embedding(2, 2),
            )

    monkeypatch.setattr(
        flamingo_module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    monkeypatch.setattr(
        flamingo_module.AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: GemmaForCausalLM(),
    )
    monkeypatch.setattr(
        flamingo_module,
        "CNNTokenizer",
        lambda: torch.nn.Identity(),
    )
    monkeypatch.setattr(flamingo_module, "extend_instance", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        flamingo_module,
        "TimeSeriesFlamingoWithTrainableEncoder",
        FakeFlamingoModel,
    )

    OpenTSLMFlamingo(device="cpu", llm_id="dummy")

    assert resize_calls == [(13, {"mean_resizing": False})]
