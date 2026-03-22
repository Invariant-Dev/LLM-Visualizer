import { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import * as mockData from '../mock/tokens';

export function useOllama(endpoint, payload, isOllamaMocked = false) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchData = async () => {
    if (isOllamaMocked) {
      setLoading(true);
      await new Promise(r => setTimeout(r, 200));
      if (endpoint === 'tokenize') setData(mockData.getMockTokens());
      else if (endpoint === 'model-info') setData(mockData.getMockModelInfo());
      else if (endpoint === 'embeddings') setData(mockData.getMockEmbeddings());
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const response = await axios.post(`/api/${endpoint}`, payload, { timeout: 30000 });
      setData(response.data);
    } catch (err) {
      setError(err.response?.data?.error || err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (payload) fetchData();
  }, [endpoint, isOllamaMocked, JSON.stringify(payload)]);

  return { data, loading, error, refetch: fetchData };
}

export function useStream(options = {}) {
  const { throttleMs = 150 } = options;
  const [tokens, setTokens] = useState([]);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState(null);
  const [finalStats, setFinalStats] = useState(null);

  const abortRef = useRef(null);
  const bufferRef = useRef([]);
  const flushIntervalRef = useRef(null);

  const flushBuffer = () => {
    if (bufferRef.current.length > 0) {
      const newTokens = [...bufferRef.current];
      setTokens(prev => [...prev, ...newTokens]);
      bufferRef.current = [];
    }
  };

  const startStream = async (payload, onToken) => {
    if (abortRef.current) abortRef.current.abort();
    if (flushIntervalRef.current) clearInterval(flushIntervalRef.current);

    const controller = new AbortController();
    abortRef.current = controller;
    bufferRef.current = [];
    setTokens([]);
    setError(null);
    setFinalStats(null);
    setStreaming(true);

    if (throttleMs > 0) {
      flushIntervalRef.current = setInterval(flushBuffer, throttleMs);
    }

    try {
      const endpoint = payload.backendMode === 'pytorch' ? '/api/pytorch/generate' : '/api/generate';
      const requestPayload = { ...payload };
      delete requestPayload.backendMode; // Don't send this internally

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestPayload),
        signal: controller.signal,
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(`Server ${response.status}: ${text}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let sseBuffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        sseBuffer += decoder.decode(value, { stream: true });
        const lines = sseBuffer.split('\n');
        sseBuffer = lines.pop(); 

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const raw = line.slice(6).trim();
          if (raw === '[DONE]') continue;

          let json;
          try { json = JSON.parse(raw); } catch { continue; }

          if (json.error) {
            setError(json.error);
            continue;
          }

          if (json.done) {
            setFinalStats({
              totalDuration: json.total_duration || json.eval_duration,
              loadDuration: json.load_duration || 0,
              promptEvalCount: json.prompt_eval_count || 0,
              evalCount: json.eval_count,
              evalDuration: json.eval_duration,
            });
            continue;
          }

          const tokenObj = {
            text: json.response ?? '',
            timestamp: Date.now(),
            logprobs: json.logprobs ?? null,
          };

          if (throttleMs > 0) {
            bufferRef.current.push(tokenObj);
          } else {
            setTokens(prev => [...prev, tokenObj]);
          }

          if (onToken) onToken(tokenObj);
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        setError(err.message);
      }
    } finally {
      if (flushIntervalRef.current) {
        clearInterval(flushIntervalRef.current);
        flushBuffer();
      }
      setStreaming(false);
      abortRef.current = null;
    }
  };

  const stopStream = () => {
    if (abortRef.current) abortRef.current.abort();
    if (flushIntervalRef.current) {
      clearInterval(flushIntervalRef.current);
      flushBuffer();
    }
    setStreaming(false);
  };

  const startMockStream = async (prompt, onToken) => {
    if (abortRef.current) abortRef.current.abort();
    if (flushIntervalRef.current) clearInterval(flushIntervalRef.current);

    const controller = new AbortController();
    abortRef.current = controller;
    bufferRef.current = [];
    setTokens([]);
    setError(null);
    setFinalStats(null);
    setStreaming(true);

    if (throttleMs > 0) {
      flushIntervalRef.current = setInterval(flushBuffer, throttleMs);
    }

    const mockResponse =
      'The sky appears blue because of a phenomenon called Rayleigh scattering. ' +
      'When sunlight enters Earth\'s atmosphere, it collides with gas molecules. ' +
      'Blue light has a shorter wavelength and scatters much more than red light, ' +
      'so we see blue when we look up at the sky.';
    const words = mockResponse.split(' ');

    for (const word of words) {
      if (controller.signal.aborted) break;
      await new Promise(r => setTimeout(r, 60 + Math.random() * 40));
      const tok = { text: word + ' ', timestamp: Date.now(), logprobs: null };
      if (throttleMs > 0) {
        bufferRef.current.push(tok);
      } else {
        setTokens(prev => [...prev, tok]);
      }
      if (onToken) onToken(tok);
    }

    if (!controller.signal.aborted) {
      setFinalStats({ evalCount: words.length, evalDuration: words.length * 90e6 });
    }

    if (flushIntervalRef.current) {
      clearInterval(flushIntervalRef.current);
      flushBuffer();
    }
    setStreaming(false);
    abortRef.current = null;
  };

  return { tokens, streaming, error, finalStats, startStream, startMockStream, stopStream };
}