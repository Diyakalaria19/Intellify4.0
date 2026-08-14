"""
specter2_encoder.py — wraps SPECTER2 so both abstract embeddings AND research
goal embeddings go through the same model, called from different places
(precompute scripts use it for abstracts; runtime uses it for research goals).

IMPORTANT DIFFERENCE FROM SPECTER v1:
SPECTER2 is NOT a simple sentence-transformers one-liner like v1 was. It uses
a base transformer model PLUS a swappable "adapter" on top, because AllenAI
split SPECTER2 into a base encoder + task-specific adapters (proximity,
classification, regression, adhoc query). For whole-document similarity
(what you're doing), you want the "proximity" adapter -- this is the one
trained on the same citation-relationship signal the original SPECTER used.

Requires: pip install adapters transformers torch
"""

import torch
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

MODEL_NAME = "allenai/specter2_base"
ADAPTER_NAME = "allenai/specter2"  # the "proximity" adapter -- general document similarity

_tokenizer = None
_model = None


def _load_model():
    """
    Loaded once and cached at module level (mirrors what @st.cache_resource
    does in Streamlit, but works the same whether called from a precompute
    script or from streamlit_app.py).
    """
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoAdapterModel.from_pretrained(MODEL_NAME)
        _model.load_adapter(ADAPTER_NAME, source="hf", load_as="proximity", set_active=True)
        _model.eval()  # inference mode -- disables dropout etc., we're not training
    return _tokenizer, _model


def encode(texts, batch_size: int = 16) -> "np.ndarray":
    """
    texts: a single string OR a list of strings.
    Returns a NumPy array: shape (768,) for a single string, (N, 768) for a list.
    Output is L2-normalized (unit length) so downstream cosine similarity is a
    plain dot product, matching how you've been treating SPECTER v1 embeddings.
    """
    import numpy as np

    tokenizer, model = _load_model()

    single_input = isinstance(texts, str)
    if single_input:
        texts = [texts]

    all_embeddings = []
    with torch.no_grad():  # no gradient tracking needed -- we're not training, saves memory/time
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            outputs = model(**inputs)
            # SPECTER2 (like v1) uses the [CLS] token's final hidden state as
            # the document embedding -- index 0 along the sequence dimension
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_embeddings.numpy())

    embeddings = np.vstack(all_embeddings)

    # normalize to unit length -- same reasoning as before: turns cosine
    # similarity into a plain dot product at comparison time
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    return embeddings[0] if single_input else embeddings


if __name__ == "__main__":
    # quick manual test
    vec = encode("A lightweight neural architecture for real-time protein folding prediction.")
    print(vec.shape)  # expect (768,)
