import importlib


def _loading_module():
    spec = importlib.util.find_spec("failfast_arc.hf_loading")
    assert spec is not None, "offline Hugging Face loader module is missing"
    return importlib.import_module("failfast_arc.hf_loading")


def test_tokenizer_load_is_local_only(monkeypatch):
    loading = _loading_module()
    sentinel = object()

    def fake_from_pretrained(source, **kwargs):
        assert source == "model"
        assert kwargs == {"revision": "revision", "local_files_only": True}
        return sentinel

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", fake_from_pretrained)
    assert loading.load_tokenizer("model", revision="revision") is sentinel


def test_sequence_classifier_load_is_local_only(monkeypatch):
    loading = _loading_module()
    sentinel = object()

    def fake_from_pretrained(source, **kwargs):
        assert source == "model"
        assert kwargs["revision"] == "revision"
        assert kwargs["local_files_only"] is True
        return sentinel

    monkeypatch.setattr("transformers.AutoModelForSequenceClassification.from_pretrained", fake_from_pretrained)
    assert loading.load_sequence_classifier("model", revision="revision", num_labels=1) is sentinel


def test_dataset_download_is_local_only(monkeypatch):
    loading = _loading_module()
    sentinel = object()

    def fake_download(**kwargs):
        assert kwargs["repo_id"] == "dataset"
        assert kwargs["filename"] == "data.parquet"
        assert kwargs["local_files_only"] is True
        return sentinel

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    assert loading.load_cached_hub_file(repo_id="dataset", filename="data.parquet") is sentinel
