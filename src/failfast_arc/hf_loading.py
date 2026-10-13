from __future__ import annotations


def load_tokenizer(source, **kwargs):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(source, local_files_only=True, **kwargs)


def load_sequence_classifier(source, **kwargs):
    from transformers import AutoModelForSequenceClassification

    return AutoModelForSequenceClassification.from_pretrained(
        source,
        local_files_only=True,
        **kwargs,
    )


def load_cached_hub_file(**kwargs):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(local_files_only=True, **kwargs)
