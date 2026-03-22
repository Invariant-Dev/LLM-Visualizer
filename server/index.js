import express from 'express';
import cors from 'cors';
import axios from 'axios';
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3001;
const OLLAMA_BASE_URL = 'http://localhost:11434';
const PYTORCH_BASE_URL = process.env.PYTORCH_URL || 'http://localhost:8000';
app.use(cors());
app.use(express.json());
app.get('/api/health', async (req, res) => {
  try {
    await axios.get(`${OLLAMA_BASE_URL}/api/tags`, { timeout: 3000 });
    res.json({ status: 'ok', ollama: 'connected' });
  } catch {
    res.status(503).json({ status: 'ok', ollama: 'unreachable' });
  }
});
app.get('/api/pytorch/health', async (req, res) => {
  try {
    const response = await axios.get(`${PYTORCH_BASE_URL}/pytorch/health`, { timeout: 5000 });
    res.json(response.data);
  } catch {
    res.status(503).json({ status: 'unavailable' });
  }
});

app.post('/api/pytorch/start', (req, res) => {
  try {
    const pytorchDir = path.resolve(__dirname, '../pytorch_server');
    // On Windows, 'start cmd /k' opens a new visible terminal window
    const child = spawn('start', ['cmd', '/k', 'python main.py'], {
      cwd: pytorchDir,
      shell: true,
      detached: true,
    });
    child.unref(); // Detach the child process so the node server can exist independently
    res.json({ status: 'starting' });
  } catch (err) {
    res.status(500).json({ error: 'failed to start pytorch backend', details: err.message });
  }
});
app.post('/api/pytorch/tokenize', async (req, res) => {
  try {
    const response = await axios.post(`${PYTORCH_BASE_URL}/pytorch/tokenize`, req.body, { timeout: 30000 });
    res.json(response.data);
  } catch (err) {
    res.status(503).json({ error: 'pytorch backend unreachable', details: err.message });
  }
});
app.post('/api/pytorch/embeddings', async (req, res) => {
  try {
    const response = await axios.post(`${PYTORCH_BASE_URL}/pytorch/embeddings`, req.body, { timeout: 60000 });
    res.json(response.data);
  } catch (err) {
    res.status(503).json({ error: 'pytorch backend unreachable', details: err.message });
  }
});
app.post('/api/pytorch/attention', async (req, res) => {
  try {
    const response = await axios.post(`${PYTORCH_BASE_URL}/pytorch/attention`, req.body, { timeout: 60000 });
    res.json(response.data);
  } catch (err) {
    res.status(503).json({ error: 'pytorch backend unreachable', details: err.message });
  }
});
app.post('/api/pytorch/full_forward', async (req, res) => {
  try {
    const response = await axios.post(`${PYTORCH_BASE_URL}/pytorch/full_forward`, req.body, { timeout: 60000 });
    res.json(response.data);
  } catch (err) {
    res.status(503).json({ error: 'pytorch backend unreachable', details: err.message });
  }
});

app.post('/api/pytorch/generate', async (req, res) => {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');
  res.flushHeaders();
  let pytorchStream = null;
  try {
    const response = await axios.post(`${PYTORCH_BASE_URL}/pytorch/generate`, req.body, { responseType: 'stream', timeout: 0 });
    pytorchStream = response.data;
    pytorchStream.pipe(res);
  } catch (err) {
    res.write(`data: ${JSON.stringify({ error: err.message })}\n\n`);
    res.end();
  }
  req.on('close', () => {
    if (pytorchStream) pytorchStream.destroy();
  });
});

app.get('/api/tags', async (req, res) => {
  try {
    const response = await axios.get(`${OLLAMA_BASE_URL}/api/tags`, { timeout: 5000 });
    res.json(response.data);
  } catch (err) {
    res.status(503).json({ error: 'Ollama unreachable', details: err.message });
  }
});
app.post('/api/model-info', async (req, res) => {
  try {
    const { model } = req.body;
    if (!model) return res.status(400).json({ error: 'model field is required' });
    const response = await axios.post(`${OLLAMA_BASE_URL}/api/show`, { name: model }, { timeout: 8000 });
    res.json(response.data);
  } catch (err) {
    res.status(503).json({ error: 'Failed to fetch model info', details: err.message });
  }
});
app.post('/api/tokenize', async (req, res) => {
  try {
    const { model, prompt } = req.body;
    if (!model || !prompt) return res.status(400).json({ error: 'model and prompt fields are required' });
    const response = await axios.post(`${OLLAMA_BASE_URL}/api/tokenize`, { model, prompt }, { timeout: 10000 });
    res.json(response.data);
  } catch (err) {
    res.status(503).json({ error: 'Failed to tokenize', details: err.message });
  }
});
app.post('/api/embeddings', async (req, res) => {
  try {
    const { model, prompt } = req.body;
    if (!model || !prompt) return res.status(400).json({ error: 'model and prompt fields are required' });
    const response = await axios.post(`${OLLAMA_BASE_URL}/api/embeddings`, { model, prompt }, { timeout: 30000 });
    res.json(response.data);
  } catch (err) {
    res.status(503).json({ error: 'Failed to generate embeddings', details: err.message });
  }
});
app.post('/api/generate', async (req, res) => {
  const { model, prompt, temperature, top_k, top_p, num_predict } = req.body;
  if (!model || !prompt) {
    return res.status(400).json({ error: 'model and prompt fields are required' });
  }
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');
  res.flushHeaders();
  let ollamaStream = null;
  try {
    const response = await axios.post(
      `${OLLAMA_BASE_URL}/api/generate`,
      {
        model,
        prompt,
        stream: true,
        options: {
          temperature: temperature ?? 0.8,
          top_k: top_k ?? 40,
          top_p: top_p ?? 0.9,
          ...(num_predict != null ? { num_predict } : {})
        }
      },
      { responseType: 'stream', timeout: 0 }
    );
    ollamaStream = response.data;
    let lineBuffer = '';
    ollamaStream.on('data', (chunk) => {
      lineBuffer += chunk.toString('utf8');
      const lines = lineBuffer.split('\n');
      lineBuffer = lines.pop();
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        try {
          JSON.parse(trimmed);
          res.write(`data: ${trimmed}\n\n`);
        } catch {
        }
      }
    });
    ollamaStream.on('end', () => {
      if (lineBuffer.trim()) {
        try {
          JSON.parse(lineBuffer.trim());
          res.write(`data: ${lineBuffer.trim()}\n\n`);
        } catch {  }
      }
      res.write('data: [DONE]\n\n');
      res.end();
    });
    ollamaStream.on('error', (err) => {
      console.error('Ollama stream error:', err.message);
      res.write(`data: ${JSON.stringify({ error: err.message })}\n\n`);
      res.end();
    });
  } catch (err) {
    console.error('Generate error:', err.message);
    res.write(`data: ${JSON.stringify({ error: err.message })}\n\n`);
    res.end();
    return;
  }
  req.on('close', () => {
    if (ollamaStream) ollamaStream.destroy();
  });
});
app.listen(PORT, () => {
  console.log(`LLM Internals Explorer server -> http://localhost:${PORT}`);
  console.log(`Proxying Ollama at ${OLLAMA_BASE_URL}`);
  console.log(`Proxying PyTorch at ${PYTORCH_BASE_URL}`);
});