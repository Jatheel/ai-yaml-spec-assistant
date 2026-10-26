import importlib


def test_embeddings_are_lazy_initialized():
    module = importlib.import_module("tools.vector_store")
    assert module._embeddings is None


def test_get_embeddings_initializes_once(monkeypatch):
    module = importlib.import_module("tools.vector_store")
    module._embeddings = None

    calls = {"count": 0}

    class DummyEmbeddings:
        def __init__(self, model_name):
            calls["count"] += 1
            self.model_name = model_name

    monkeypatch.setattr(module, "HuggingFaceEmbeddings", DummyEmbeddings)

    first = module._get_embeddings()
    second = module._get_embeddings()

    assert calls["count"] == 1
    assert first is second
