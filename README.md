<div align="center">
  <h1>🧠 LLM Internals Explorer</h1>
  <p><em>A visual, interactive, stage-by-stage explainer of exactly what happens when you send a prompt to a large language model. Built for learners who want to see under the hood.</em></p>
  
  [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
  [![React](https://img.shields.io/badge/React-18-blue.svg)](https://reactjs.org/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.103-green.svg)](https://fastapi.tiangolo.com/)
  [![Ollama](https://img.shields.io/badge/Ollama-Compatible-black.svg)](https://ollama.com/)
</div>

---

## ✨ Key Features

- **🚀 Real Ollama Integration** — Connects directly to your local `ollama serve` to visualize actual tokenization and generation.
- **🔥 Native PyTorch Backend** — An optional Python FastAPI backend that runs small HuggingFace models (like GPT-2 or SmolLM) to extract **real** mathematical data like attention weights and per-token embeddings.
- **🎨 Beautiful Dark Theme** — A sleek, modern UI designed for clarity, built with CSS variables.
- **🔄 Smart Mock Mode** — Automatically falls back to high-quality mock data if no backends are running, perfect for quick demos or learning on the go.
- **📱 Responsive & Accessible** — Navigate through stages using your keyboard arrow keys.
- **📚 Built-in Learning** — "Why does this matter?" explainers on every single stage.

---

## 📖 The 8 Interactive Stages

1. **Raw Input** — See how your text is broken down into UTF-8 bytes with hex codes.
2. **Tokenizer** — Watch the text split into tokens with IDs, featuring color-coded BPE classifications.
3. **Embeddings** — View high-dimensional vectors projected to 2D using PCA. Inspect per-token dimensions and cosine similarities!
4. **Self-Attention** — The core of the Transformer. Visualize attention weights in a heatmap or arc view. Analyze head entropy and layer-by-layer attention stats.
5. **Feed-Forward Network** — Watch token data flow through the MLP layers with animated node activations.
6. **Layer Stack** — Trace a token's journey through all 32 layers of the deep neural network.
7. **Softmax & Temperature** — Interactively tweak Temperature, Top-K, and Top-P to see how the probability distribution bends and warps in real-time.
8. **Autoregressive Generation** — Watch the model generate text token-by-token with live streaming stats (tokens/sec, latency sparklines).

---

## 🛠️ Setup Process & Installation

You can run this project in **Standard Mode** (Node.js + Ollama) or **Advanced Mode** (adding the Python PyTorch backend for deep internal data extraction).

### 1. Prerequisites
- **Node.js** v18+ — [Download](https://nodejs.org/)
- **Ollama** — [Download](https://ollama.com) (Pull at least one model: `ollama pull llama3`)
- *(Optional)* **Python 3.9+** — For the PyTorch Native backend.

### 2. Standard Installation

```bash
# 1. Clone or extract the project
git clone https://github.com/yourusername/llm-internals-explorer.git
cd llm-internals-explorer

# 2. Install dependencies (installs root, server, and client packages)
npm install
npm --prefix client install
npm --prefix server install

# 3. Start Ollama in a separate terminal
ollama serve

# 4. Start the development server
npm run dev
```
*The app will be available at `http://localhost:5173`. The Node proxy server runs on `http://localhost:3001`.*

### 3. Advanced PyTorch Backend Setup (Highly Recommended)
To visualize **real** attention matrices and continuous vector embeddings, you can spin up the PyTorch server directly from the main app interface.

```bash
# 1. First time only: Install Python dependencies
cd pytorch_server
pip install -r requirements.txt
```

Once dependencies are installed, just run `npm run dev` and click the **+ Start PyTorch** button in the app header to launch the companion backend automatically in a new window!

---

## 🏗️ Architecture & Project Structure

The project uses a dual-backend architecture to provide both conversational generation and deep internal tensor inspection.

```text
llm-internals-explorer/
├── client/                      # React + Vite frontend
│   ├── src/stages/              # The 8 interactive stage components
│   ├── src/components/          # Shared UI (Nav, Toggles, Tour)
│   ├── src/workers/             # Web Workers (e.g., PCA calculations)
│   └── src/context/             # Global AppContext state
├── server/                      # Node.js / Express Proxy
│   ├── index.js                 # Routes to Ollama & Python backend
│   └── mock/                    # Fallback data when offline
└── pytorch_server/              # Native PyTorch API
    ├── main.py                  # FastAPI server extracting Tensors
    └── requirements.txt         # Python dependencies
```

---

## 📡 API Proxy Routes

The Node server at `localhost:3001` intelligently proxies your frontend requests:

- `/api/generate`, `/api/model-info`, `/api/tags` ➡️ Local Ollama Server
- `/api/pytorch/attention`, `/api/pytorch/embeddings` ➡️ Local PyTorch Python Server

---

## ⚙️ Known Limitations

1. **Ollama's API Limits:** Ollama does not expose internal layer attentions or per-token continuous embeddings natively. Our solution is the `pytorch_server` backend. If you only use Ollama, Stages 3 and 4 will gracefully fall back to realistic mock data.
2. **Logprobs Requirement:** Visualizing alternate token paths in Stage 7 and 8 requires Ollama v0.1.33+.
3. **Heavy Tensors:** The PyTorch backend extracts massive tensors (`output_attentions=True`). On older machines, extracting data for long prompts might take a few seconds.

---

## 🤝 Contributing

We welcome contributions! Feel free to improve visualizations, add new stages, or optimize performance. Some open ideas:

- [ ] Add KV-cache visualization stage.
- [ ] Show gradient flow during autoregressive generation.
- [ ] Support for multiple PyTorch models side-by-side.
- [ ] Export raw mathematically accurate attention matrices as CSV.

---

## 📝 License

This project is licensed under the **GNUv3** License.

---

<div align="center">
  <b>Happy exploring!</b> 🧠✨<br>
  <i>Don't forget to take the guided tour in the app to get familiar with the interface!</i>
</div>
