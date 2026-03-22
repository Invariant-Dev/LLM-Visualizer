# Native PyTorch / HuggingFace Backend

**Goal:** Provide real, detailed internal model data (attention weights, per-token embeddings) to replace the remaining mock data, running a small model locally via Python.

### What to build

**1. Python FastAPI Server**
- Set up a small Python backend alongside the Node proxy.
- **Dependencies:** `fastapi`, `uvicorn`, `torch`, `transformers`
- Endpoints: 
  - `/pytorch/attention`: Returns attention matrices.
  - `/pytorch/embeddings`: Returns per-token embeddings.
- Load a lightweight model (e.g., `gpt2`, `Qwen/Qwen2.5-0.5B`, or `SmolLM`) using `transformers`.
- Configure the model to output internals: `output_attentions=True` and `output_hidden_states=True`.

**2. Real Extractors**
- **Attention Exporter:** Extract the raw attention matrices from the transformer layers, format them as N×N matrices per head, and return them as JSON.
- **Embedding Exporter:** Extract the `last_hidden_state` for each token, providing the real 4096-dim (or whatever dimension size) vector for the PCA visualization.

**3. Frontend Integration**
- Add a new "Backend Mode" toggle in the header: `[ Ollama | PyTorch Native | Mock ]`.
- When "PyTorch Native" is selected:
  - Stage 3 (Embeddings) bypasses the mock data and fetches accurate per-token vectors from `/pytorch/embeddings`.
  - Stage 4 (Attention) bypasses the mock data and fetches actual N×N attention weights per head from `/pytorch/attention`.
- Ensure the UI gracefully handles the processing time for extracting these massive tensors (e.g., adding loading states).

**Done when:** You can see real, mathematically accurate attention heads clustering around specific tokens, and the 2D PCA accurately reflects the geometry of the actual latent space of the loaded PyTorch model.