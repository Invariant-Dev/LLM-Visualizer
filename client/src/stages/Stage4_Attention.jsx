import React, { useState, useEffect } from 'react';
import * as mockData from '../mock/tokens';
import axios from 'axios';
const HEAD_PROFILES = [
  { name: 'Previous Token', desc: 'Attends strongly to the immediately preceding token. Common in early layers.' },
  { name: 'Positional', desc: 'Focuses on fixed relative positions. Encodes word order and distance.' },
  { name: 'Syntactic Subject', desc: 'Links verbs to their subjects across the sentence.' },
  { name: 'Delimiter', desc: 'Attends to punctuation and sentence boundaries.' },
  { name: 'Copy / Induction', desc: 'Looks for repeated patterns and copies tokens that followed similar prefixes before.' },
  { name: 'Rare Token', desc: 'Pays attention to uncommon or surprising tokens in context.' },
  { name: 'Coreference', desc: 'Links pronouns and references back to the nouns they refer to.' },
  { name: 'Broad Context', desc: 'Distributes attention roughly evenly. Gathers global sentence meaning.' },
  { name: 'Adjective-Noun', desc: 'Connects modifiers to the nouns they describe.' },
  { name: 'Negation Tracker', desc: 'Watches for negation words like "not", "never", "no" and propagates their signal.' },
  { name: 'Semantic Similarity', desc: 'Attends to tokens with similar meaning regardless of position.' },
  { name: 'Question Word', desc: 'Focuses on interrogative words (who, what, why, how) in questions.' },
  { name: 'Preposition', desc: 'Links prepositions to their objects and the words they modify.' },
  { name: 'Long-Range', desc: 'Connects tokens that are far apart. Important for discourse coherence.' },
  { name: 'Local Window', desc: 'Attends only to a tight local window of 2-3 tokens.' },
  { name: 'BOS / EOS', desc: 'Focuses on beginning or end-of-sequence tokens. Anchors global representations.' },
];
function getHeadProfile(headIdx) {
  return HEAD_PROFILES[headIdx % HEAD_PROFILES.length];
}
export default function Stage4_Attention({ tokens, isOllamaMocked, backendMode, goToNextStage }) {
  const [selectedHead, setSelectedHead] = useState(0);
  const [selectedLayer, setSelectedLayer] = useState(-1);
  const [isArcView, setIsArcView] = useState(false);
  const [attentionHeads, setAttentionHeads] = useState([]);
  const [headEntropies, setHeadEntropies] = useState([]);
  const [layerSummary, setLayerSummary] = useState([]);
  const [pytorchStats, setPytorchStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [realTokens, setRealTokens] = useState([]);
  useEffect(() => {
    if (backendMode === 'pytorch') {
      setLoading(true);
      const prompt = tokens.map(t => t.text).join('');
      axios.post('/api/pytorch/attention', { prompt, layer: selectedLayer })
        .then(res => {
          const data = res.data;
          setAttentionHeads(data.heads);
          setHeadEntropies(data.head_entropies || []);
          setLayerSummary(data.layer_summary || []);
          setPytorchStats(data.stats || null);
          setRealTokens(data.tokens || []);
          setSelectedHead(0);
        })
        .catch(() => {
          const { heads } = mockData.getMockAttentionHeads(tokens.length);
          setAttentionHeads(heads);
        })
        .finally(() => setLoading(false));
    } else {
      const { heads } = mockData.getMockAttentionHeads(tokens.length);
      setAttentionHeads(heads);
      setHeadEntropies([]);
      setLayerSummary([]);
      setPytorchStats(null);
      setRealTokens([]);
    }
  }, [tokens.length, backendMode, selectedLayer]);
  if (attentionHeads.length === 0 && !loading) {
    return <div className="stage-panel"><div className="loading">Loading attention...</div></div>;
  }
  const displayTokens = backendMode === 'pytorch' && realTokens.length > 0
    ? realTokens.map(t => ({ text: t }))
    : tokens;
  const currentMatrix = attentionHeads[selectedHead] || [];
  const tokenCount = displayTokens.length;
  const profile = getHeadProfile(selectedHead);
  const getColor = (value) => {
    const r = Math.round(9 + value * 242);
    const g = Math.round(9 + value * 137);
    const b = Math.round(11 + value * 49);
    return `rgb(${r}, ${g}, ${b})`;
  };
  const topAttentions = [];
  if (currentMatrix.length > 0) {
    currentMatrix.forEach((row, i) => {
      row.forEach((val, j) => {
        topAttentions.push({ from: i, to: j, weight: val });
      });
    });
    topAttentions.sort((a, b) => b.weight - a.weight);
  }
  const topArcs = topAttentions.slice(0, Math.min(6, tokenCount * 2));
  return (
    <div className="stage-panel">
      <h2>Stage 4: Self-Attention</h2>
      <p className="stage-description">
        Tokens "attend" to other tokens. This heatmap shows attention weights from each token (rows) to every
        other token (columns). The model uses {attentionHeads.length} attention heads in parallel, each learning different patterns.
        {backendMode === 'pytorch' && <span style={{ color: '#a78bfa' }}> (Real PyTorch attention weights)</span>}
      </p>
      {backendMode === 'pytorch' && pytorchStats && (
        <div style={{
          display: 'flex', gap: 16, marginBottom: 16, padding: '10px 14px', borderRadius: 8, flexWrap: 'wrap',
          background: 'rgba(167,139,250,0.06)', border: '1px solid rgba(167,139,250,0.15)',
        }}>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
            <span style={{ color: '#a78bfa', fontWeight: 600 }}>{pytorchStats.num_layers}</span> layers
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
            <span style={{ color: '#a78bfa', fontWeight: 600 }}>{pytorchStats.num_heads}</span> heads
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
            <span style={{ color: '#a78bfa', fontWeight: 600 }}>{pytorchStats.num_tokens}</span> tokens
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
            Layer <span style={{ color: '#a78bfa', fontWeight: 600 }}>{pytorchStats.layer_used}</span>
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
            <span style={{ color: '#a78bfa', fontWeight: 600 }}>{pytorchStats.inference_ms}</span> ms
          </div>
        </div>
      )}
      {loading && (
        <div style={{ padding: 20, textAlign: 'center', color: 'var(--text-muted)', fontSize: 13 }}>
          Extracting real attention weights from PyTorch model...
        </div>
      )}
      <div className="attention-controls">
        {backendMode === 'pytorch' && layerSummary.length > 0 && (
          <div style={{ marginRight: 12 }}>
            <label>Layer: </label>
            <select value={selectedLayer} onChange={(e) => setSelectedLayer(parseInt(e.target.value))}>
              <option value={-1}>Last layer ({layerSummary.length - 1})</option>
              {layerSummary.map((ls) => (
                <option key={ls.layer} value={ls.layer}>
                  Layer {ls.layer} (entropy: {ls.avg_entropy.toFixed(2)}, max attn: {ls.avg_max_attention.toFixed(2)})
                </option>
              ))}
            </select>
          </div>
        )}
        <div className="head-selector">
          <label>Head: </label>
          <select value={selectedHead} onChange={(e) => setSelectedHead(parseInt(e.target.value))}>
            {attentionHeads.map((_, idx) => (
              <option key={idx} value={idx}>
                Head {idx + 1} - {getHeadProfile(idx).name}
                {headEntropies[idx] != null ? ` (H=${headEntropies[idx].toFixed(2)})` : ''}
              </option>
            ))}
          </select>
        </div>
        <button
          className={`btn ${isArcView ? 'btn-secondary' : 'btn-primary'}`}
          onClick={() => setIsArcView(!isArcView)}
        >
          {isArcView ? 'Heatmap View' : 'Arc View'}
        </button>
      </div>
      { }
      <div style={{
        padding: '12px 16px', marginBottom: 16, borderRadius: 8,
        background: 'rgba(251,146,60,0.06)', border: '1px solid rgba(251,146,60,0.2)',
        borderLeft: '4px solid #fb923c',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
          <span style={{ fontSize: 13, fontWeight: 700, color: '#fb923c' }}>
            {profile.name}
          </span>
          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Head {selectedHead + 1}</span>
          {headEntropies[selectedHead] != null && (
            <span style={{ fontSize: 10, color: '#a78bfa', marginLeft: 8 }}>
              entropy: {headEntropies[selectedHead].toFixed(3)} bits
            </span>
          )}
        </div>
        <p style={{ fontSize: 12, color: 'var(--text-secondary)', margin: 0 }}>{profile.desc}</p>
      </div>
      {!isArcView ? (
        <div className="attention-heatmap">
          <div style={{ display: 'flex', gap: 0 }}>
            <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', marginRight: 4 }}>
              {displayTokens.map((token, idx) => (
                <div key={idx} style={{ height: 36, display: 'flex', alignItems: 'center', fontSize: 10, color: 'var(--text-muted)', justifyContent: 'flex-end', paddingRight: 4 }}>
                  {String(token.text).substring(0, 5)}
                </div>
              ))}
            </div>
            <div>
              {currentMatrix.map((row, i) => (
                <div key={i} style={{ display: 'flex' }}>
                  {row.map((val, j) => (
                    <div
                      key={`${i}-${j}`}
                      style={{
                        width: 36, height: 36,
                        backgroundColor: getColor(val),
                        border: '1px solid rgba(0,0,0,0.3)',
                        cursor: 'pointer',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        fontSize: 8, color: val > 0.5 ? 'rgba(0,0,0,0.6)' : 'rgba(255,255,255,0.3)',
                        transition: 'all 0.15s',
                      }}
                      title={`${displayTokens[i]?.text} -> ${displayTokens[j]?.text}: ${val.toFixed(4)}`}
                    >
                      {val > 0.15 ? val.toFixed(1) : ''}
                    </div>
                  ))}
                </div>
              ))}
              <div style={{ display: 'flex', marginTop: 4 }}>
                {displayTokens.map((token, idx) => (
                  <div key={idx} style={{ width: 36, textAlign: 'center', fontSize: 10, color: 'var(--text-muted)' }}>
                    {String(token.text).substring(0, 4)}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="arc-view">
          <svg width="100%" height="200">
            <defs>
              <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
                <polygon points="0 0, 10 3, 0 6" fill="#fb923c" />
              </marker>
            </defs>
            {displayTokens.map((token, idx) => {
              const x = (idx / Math.max(tokenCount - 1, 1)) * 95 + 2.5;
              return (
                <text key={`label-${idx}`} x={`${x}%`} y="190" textAnchor="middle" fill="#a1a1aa" fontSize="11">
                  {String(token.text).substring(0, 4)}
                </text>
              );
            })}
            {topArcs.map((arc, idx) => {
              const x1 = (arc.from / Math.max(tokenCount - 1, 1)) * 95 + 2.5;
              const x2 = (arc.to / Math.max(tokenCount - 1, 1)) * 95 + 2.5;
              const midX = (x1 + x2) / 2;
              const height = Math.abs(x2 - x1) * 2;
              const path = `M ${x1}% 180, Q ${midX}% ${Math.max(50, height / 2)}, ${x2}% 180`;
              return (
                <path
                  key={idx}
                  d={path}
                  stroke="#fb923c"
                  strokeWidth={arc.weight * 4}
                  fill="none"
                  opacity={0.7}
                />
              );
            })}
          </svg>
        </div>
      )}
      {backendMode === 'pytorch' && layerSummary.length > 0 && (
        <div style={{ marginTop: 20 }}>
          <h4 style={{ marginBottom: 8 }}>Layer-by-Layer Attention Overview</h4>
          <div style={{ display: 'flex', gap: 2, alignItems: 'flex-end', height: 80 }}>
            {layerSummary.map((ls) => (
              <div
                key={ls.layer}
                title={`Layer ${ls.layer}: avg entropy ${ls.avg_entropy.toFixed(3)}, avg max ${ls.avg_max_attention.toFixed(3)}`}
                onClick={() => setSelectedLayer(ls.layer)}
                style={{
                  flex: 1,
                  height: `${ls.avg_max_attention * 100}%`,
                  minHeight: 4,
                  background: selectedLayer === ls.layer || (selectedLayer === -1 && ls.layer === layerSummary.length - 1)
                    ? 'rgba(167,139,250,0.6)'
                    : 'rgba(167,139,250,0.2)',
                  borderRadius: 2,
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                }}
              />
            ))}
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, color: 'var(--text-muted)', marginTop: 4 }}>
            <span>Layer 0</span>
            <span>Layer {layerSummary.length - 1}</span>
          </div>
          <p style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>
            Bar height = average max attention weight per layer. Click a bar to inspect that layer.
          </p>
        </div>
      )}
      <div className="explainer">
        <h3>Why does this matter?</h3>
        <p>
          Self-attention is the core innovation of transformers. It lets the model weigh the importance of
          each token when processing every other token. Different heads specialise in different linguistic
          patterns: some track syntax, others track semantics or long-range dependencies.
          {backendMode === 'pytorch' && ' In PyTorch mode, these are the actual attention matrices extracted from the model forward pass. You can browse every layer and every head.'}
        </p>
      </div>
    </div>
  );
}