import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useOllama } from '../hooks/useOllama';
import axios from 'axios';
export default function Stage3_Embeddings({ tokens, isOllamaMocked, backendMode, selectedModel, goToNextStage }) {
  const [selectedTokenIdx, setSelectedTokenIdx] = useState(0);
  const [scatterPlotData, setScatterPlotData] = useState([]);
  const [embeddings, setEmbeddings] = useState([]);
  const [injectedWords, setInjectedWords] = useState([]);
  const [injectedInput, setInjectedInput] = useState('');
  const [pytorchStats, setPytorchStats] = useState(null);
  const [cosineMatrix, setCosineMatrix] = useState(null);
  const [norms, setNorms] = useState([]);
  const [loading, setLoading] = useState(false);
  const { data: embeddingData } = useOllama(
    'embeddings',
    { model: selectedModel || 'llama2', prompt: tokens.map(t => t.text).join('') },
    isOllamaMocked || backendMode === 'pytorch'
  );

  const workerRef = useRef(null);

  useEffect(() => {
    workerRef.current = new Worker(new URL('../workers/pca.worker.js', import.meta.url), { type: 'module' });
    workerRef.current.onmessage = (e) => {
      if (e.data.type === 'success') {
        const { projected, payload: { labels, injectedLabels } } = e.data;
        const xs = projected.map(p => p[0]);
        const ys = projected.map(p => p[1]);
        const minX = Math.min(...xs);
        const maxX = Math.max(...xs);
        const minY = Math.min(...ys);
        const maxY = Math.max(...ys);
        const rangeX = maxX - minX || 1;
        const rangeY = maxY - minY || 1;
        const normalized = projected.map((p, idx) => ({
          x: ((p[0] - minX) / rangeX) * 100,
          y: ((p[1] - minY) / rangeY) * 100,
          token: idx < labels.length ? labels[idx] : injectedLabels[idx - labels.length],
          index: idx,
          isInjected: idx >= labels.length,
        }));
        setScatterPlotData(normalized);
      }
    };
    return () => workerRef.current?.terminate();
  }, []);

  const projectAll = useCallback((vecs, labels, injectedVecs, injectedLabels) => {
    const allVecs = [...vecs, ...injectedVecs];
    if (allVecs.length < 2) return;
    if (workerRef.current) {
      workerRef.current.postMessage({
        vecs: allVecs,
        nComponents: 2,
        payload: { labels, injectedLabels }
      });
    }
  }, []);

  useEffect(() => {
    if (backendMode === 'pytorch') {
      setLoading(true);
      const prompt = tokens.map(t => t.text).join('');
      axios.post('/api/pytorch/embeddings', { prompt })
        .then(res => {
          const data = res.data;
          setEmbeddings(data.embeddings);
          setNorms(data.norms || []);
          setCosineMatrix(data.cosine_similarity || null);
          setPytorchStats(data.stats || null);
          const labels = data.tokens || tokens.map(t => t.text || 'token');
          projectAll(data.embeddings, labels, [], []);
        })
        .catch(() => {})
        .finally(() => setLoading(false));
      return;
    }
    let vecs = [];
    if (embeddingData) {
      if (embeddingData.embeddings && Array.isArray(embeddingData.embeddings[0])) {
        vecs = embeddingData.embeddings;
      } else if (embeddingData.embedding && tokens.length > 0) {
        const baseVec = embeddingData.embedding;
        vecs = tokens.map(() => baseVec.map(v => v + (Math.random() - 0.5) * 0.05));
      }
    }
    if (vecs.length > 0) {
      setEmbeddings(vecs);
      const labels = tokens.map(t => t.text || 'token');
      const injVecs = injectedWords.map(w => w.vec);
      const injLabels = injectedWords.map(w => w.word);
      projectAll(vecs, labels, injVecs, injLabels);
    }
  }, [embeddingData, tokens, injectedWords, projectAll, backendMode]);
  const handleInject = async () => {
    const word = injectedInput.trim();
    if (!word || injectedWords.find(w => w.word === word)) return;
    if (backendMode === 'pytorch') {
      try {
        const res = await axios.post('/api/pytorch/embeddings', { prompt: word });
        const vec = res.data?.embeddings?.[0];
        if (vec) setInjectedWords(prev => [...prev, { word, vec }]);
      } catch {}
    } else if (isOllamaMocked) {
      const fakeVec = embeddings.length > 0
        ? embeddings[0].map(v => v + (Math.random() - 0.5) * 0.3)
        : Array(32).fill(0).map(() => Math.random());
      setInjectedWords(prev => [...prev, { word, vec: fakeVec }]);
    } else {
      try {
        const res = await axios.post('/api/embeddings', { model: selectedModel, prompt: word }, { timeout: 10000 });
        const vec = res.data?.embedding || (res.data?.embeddings?.[0]);
        if (vec) {
          setInjectedWords(prev => [...prev, { word, vec }]);
        }
      } catch {
      }
    }
    setInjectedInput('');
  };
  const removeInjected = (word) => {
    setInjectedWords(prev => prev.filter(w => w.word !== word));
  };
  const selectedEmbedding = embeddings[selectedTokenIdx] || [];
  const sampledDims = selectedEmbedding.slice(0, 32);
  return (
    <div className="stage-panel">
      <h2>Stage 3: Embeddings</h2>
      <p className="stage-description">
        Each token is mapped to a high-dimensional vector. These embeddings encode semantic meaning.
        Similar tokens have vectors that point in similar directions.
        {backendMode === 'pytorch' && <span style={{ color: '#a78bfa' }}> (Real PyTorch embeddings)</span>}
      </p>
      {backendMode === 'pytorch' && pytorchStats && (
        <div style={{
          display: 'flex', gap: 16, marginBottom: 16, padding: '10px 14px', borderRadius: 8, flexWrap: 'wrap',
          background: 'rgba(167,139,250,0.06)', border: '1px solid rgba(167,139,250,0.15)',
        }}>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
            <span style={{ color: '#a78bfa', fontWeight: 600 }}>{pytorchStats.embedding_dim}</span> dimensions
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
            Layer <span style={{ color: '#a78bfa', fontWeight: 600 }}>{pytorchStats.layer_used}</span> / {pytorchStats.total_layers}
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
            <span style={{ color: '#a78bfa', fontWeight: 600 }}>{pytorchStats.inference_ms}</span> ms
          </div>
        </div>
      )}
      {loading && (
        <div style={{ padding: 20, textAlign: 'center', color: 'var(--text-muted)', fontSize: 13 }}>
          Extracting real embeddings from PyTorch model...
        </div>
      )}
      { }
      <div style={{
        display: 'flex', gap: 8, marginBottom: 16, alignItems: 'center',
        padding: '10px 14px', borderRadius: 8,
        background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)',
      }}>
        <input
          type="text"
          value={injectedInput}
          onChange={e => setInjectedInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleInject()}
          placeholder="Inject a word into the space..."
          style={{
            flex: 1, padding: '6px 10px', borderRadius: 4,
            background: 'var(--dark-bg)', border: '1px solid rgba(255,255,255,0.1)',
            color: 'var(--text-primary)', fontSize: 13, outline: 'none',
          }}
        />
        <button className="btn btn-primary" onClick={handleInject} style={{ padding: '6px 14px' }}>
          + Inject
        </button>
      </div>
      {injectedWords.length > 0 && (
        <div style={{ display: 'flex', gap: 6, marginBottom: 12, flexWrap: 'wrap' }}>
          {injectedWords.map(w => (
            <span key={w.word} style={{
              padding: '3px 10px', borderRadius: 12, fontSize: 11, cursor: 'pointer',
              background: 'rgba(251,146,60,0.15)', border: '1px solid rgba(251,146,60,0.4)',
              color: '#fb923c',
            }} onClick={() => removeInjected(w.word)}>
              {w.word} x
            </span>
          ))}
        </div>
      )}
      <div className="embedding-container">
        <div className="embedding-left">
          <h4>Token Embeddings (2D Projection)</h4>
          <svg className="scatter-plot" width="300" height="300" style={{ background: 'rgba(0,0,0,0.2)', borderRadius: 6 }}>
            {scatterPlotData.map((point) => (
              <g key={`${point.token}-${point.index}`}>
                <circle
                  cx={point.x * 2.8 + 10}
                  cy={point.y * 2.8 + 10}
                  r={point.isInjected ? 7 : (selectedTokenIdx === point.index ? 6 : 4)}
                  fill={point.isInjected ? '#fb923c' : (selectedTokenIdx === point.index ? '#f4f4f5' : '#a1a1aa')}
                  opacity={0.85}
                  onClick={() => !point.isInjected && setSelectedTokenIdx(point.index)}
                  style={{ cursor: point.isInjected ? 'default' : 'pointer', transition: 'all 0.3s' }}
                />
                <text
                  x={point.x * 2.8 + 18}
                  y={point.y * 2.8 + 14}
                  fontSize="10"
                  fill={point.isInjected ? '#fb923c' : '#a1a1aa'}
                  fontWeight={point.isInjected ? 600 : 400}
                >
                  {String(point.token).substring(0, 6)}
                </text>
              </g>
            ))}
          </svg>
        </div>
        <div className="embedding-right">
          <h4>Sampled Dimensions {backendMode === 'pytorch' && `(${embeddings[0]?.length || 0}-dim)`}</h4>
          <div className="dimension-bars">
            {sampledDims.map((val, idx) => (
              <div key={idx} className="dimension-bar-container" style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <span style={{ fontSize: 9, color: 'var(--text-muted)', width: 14, textAlign: 'right' }}>{idx}</span>
                <div style={{ flex: 1, display: 'flex', justifyContent: val >= 0 ? 'flex-start' : 'flex-end' }}>
                  <div
                    className="dimension-bar"
                    style={{
                      width: `${Math.min(Math.abs(val) * 60, 100)}%`,
                      backgroundColor: val >= 0 ? '#10b981' : '#ef4444',
                      height: '3px',
                      borderRadius: 2,
                      transition: 'width 0.3s',
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
          <p className="code" style={{ marginTop: 8 }}>Selected: {tokens[selectedTokenIdx]?.text || 'None'}</p>
          {backendMode === 'pytorch' && norms.length > 0 && (
            <div style={{ marginTop: 8, fontSize: 11, color: 'var(--text-muted)' }}>
              L2 norm: <span style={{ color: '#a78bfa', fontWeight: 600 }}>{norms[selectedTokenIdx]?.toFixed(2)}</span>
            </div>
          )}
        </div>
      </div>
      {backendMode === 'pytorch' && cosineMatrix && (
        <div style={{ marginTop: 20 }}>
          <h4 style={{ marginBottom: 8 }}>Cosine Similarity Matrix (Real)</h4>
          <div style={{ overflowX: 'auto' }}>
            <div style={{ display: 'flex', gap: 0 }}>
              <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', marginRight: 4 }}>
                {tokens.map((token, idx) => (
                  <div key={idx} style={{ height: 28, display: 'flex', alignItems: 'center', fontSize: 9, color: 'var(--text-muted)', justifyContent: 'flex-end', paddingRight: 4 }}>
                    {String(token.text).substring(0, 5)}
                  </div>
                ))}
              </div>
              <div>
                {cosineMatrix.map((row, i) => (
                  <div key={i} style={{ display: 'flex' }}>
                    {row.map((val, j) => {
                      const intensity = Math.abs(val);
                      const hue = val >= 0 ? 160 : 0;
                      return (
                        <div
                          key={`${i}-${j}`}
                          style={{
                            width: 28, height: 28,
                            backgroundColor: `hsla(${hue}, 70%, 50%, ${intensity * 0.8 + 0.05})`,
                            border: '1px solid rgba(0,0,0,0.3)',
                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                            fontSize: 7, color: intensity > 0.5 ? 'rgba(0,0,0,0.6)' : 'rgba(255,255,255,0.3)',
                          }}
                          title={`${tokens[i]?.text} <-> ${tokens[j]?.text}: ${val.toFixed(4)}`}
                        >
                          {val > 0.15 || val < -0.15 ? val.toFixed(2) : ''}
                        </div>
                      );
                    })}
                  </div>
                ))}
                <div style={{ display: 'flex', marginTop: 4 }}>
                  {tokens.map((token, idx) => (
                    <div key={idx} style={{ width: 28, textAlign: 'center', fontSize: 9, color: 'var(--text-muted)' }}>
                      {String(token.text).substring(0, 3)}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
          <p style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 6 }}>
            Green = positive cosine similarity, Red = negative. Diagonal is always 1.0 (self-similarity).
          </p>
        </div>
      )}
      <div className="explainer">
        <h3>Why does this matter?</h3>
        <p>
          Embeddings turn discrete symbols (words) into continuous geometric space where neural operations
          make sense. The distance between vectors correlates with semantic similarity. Try injecting
          words like "ocean" or "red" to see where they land relative to your prompt tokens.
          {backendMode === 'pytorch' && ' In PyTorch mode, these are the actual hidden state vectors from the model.'}
        </p>
      </div>
    </div>
  );
}