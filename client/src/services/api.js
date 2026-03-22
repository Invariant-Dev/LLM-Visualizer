import axios from 'axios';

const API_TIMEOUT = 30000;
const MAX_RETRIES = 3;
const BASE_DELAY = 500;

async function withRetry(fn, retries = MAX_RETRIES) {
  for (let attempt = 0; attempt < retries; attempt++) {
    try {
      return await fn();
    } catch (err) {
      const isLast = attempt === retries - 1;
      const isRetryable = !err.response || err.response.status >= 500;
      if (isLast || !isRetryable) throw err;
      const delay = BASE_DELAY * Math.pow(2, attempt) + Math.random() * 200;
      await new Promise(r => setTimeout(r, delay));
    }
  }
}

export async function fetchTags() {
  return withRetry(async () => {
    const res = await axios.get('/api/tags', { timeout: 5000 });
    return res.data?.models ?? [];
  });
}

export async function fetchModelInfo(model) {
  return withRetry(async () => {
    const res = await axios.post('/api/model-info', { model }, { timeout: 8000 });
    return res.data;
  });
}

export async function fetchTokenize(payload) {
  return withRetry(async () => {
    const res = await axios.post('/api/tokenize', payload, { timeout: API_TIMEOUT });
    return res.data;
  });
}

export async function fetchEmbeddings(payload) {
  return withRetry(async () => {
    const res = await axios.post('/api/embeddings', payload, { timeout: API_TIMEOUT });
    return res.data;
  });
}
