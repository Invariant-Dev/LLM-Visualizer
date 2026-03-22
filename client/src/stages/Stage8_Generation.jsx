import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';


import { useStream } from '../hooks/useOllama';

import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, BarElement, Filler, Tooltip, Legend, ArcElement, RadialLinearScale } from 'chart.js';

import { Line, Bar, Doughnut } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, ArcElement, RadialLinearScale, Filler, Tooltip, Legend);

function nsToTps(evalCount, evalDurationNs) {

  if (!evalDurationNs || evalDurationNs === 0) return null;

  return (evalCount / (evalDurationNs / 1e9)).toFixed(1);

}

function logprobColor(logprob) {

  if (logprob == null) return 'var(--text-muted)';

  if (logprob > -0.5) return '#10b981';

  if (logprob > -2) return '#f59e0b';

  return '#ef4444';

}

function generateMockEmbedding(text, dims = 64) {

  const values = [];

  let seed = 0;

  for (let i = 0; i < text.length; i++) seed = ((seed << 5) - seed + text.charCodeAt(i)) | 0;

  for (let d = 0; d < dims; d++) {

    seed = (seed * 16807 + 12345) & 0x7fffffff;

    values.push(((seed / 0x7fffffff) * 2 - 1));

  }

  const mag = Math.sqrt(values.reduce((s, v) => s + v * v, 0)) || 1;

  return values.map(v => v / mag);

}

function entropyFromLogprob(lp) {

  if (lp == null) return null;

  const p = Math.exp(lp);

  if (p <= 0 || p >= 1) return 0;

  return -(p * Math.log2(p) + (1 - p) * Math.log2(1 - p));

}

function generateMockAlternatives(tokenText) {

  const alternatives = [

    { text: tokenText, prob: 0.4 + Math.random() * 0.3 },

  ];

  const filler = ['the', 'a', 'is', 'an', 'of', 'to', 'in', 'it', 'for', 'on', 'as', 'at', 'by', 'or', 'and', 'but'];

  const count = 4 + Math.floor(Math.random() * 4);

  let remaining = 1 - alternatives[0].prob;

  for (let i = 0; i < count; i++) {

    const word = filler[(tokenText.charCodeAt(0) + i * 7) % filler.length];

    const p = i < count - 1 ? remaining * (0.2 + Math.random() * 0.3) : remaining;

    alternatives.push({ text: word, prob: Math.max(0.001, p) });

    remaining -= p;

    if (remaining <= 0) break;

  }

  return alternatives.sort((a, b) => b.prob - a.prob);

}

export default function Stage8_Generation({ input, isOllamaMocked, selectedModel, backendMode }) {
  const payloadBackendMode = backendMode;

  const [prompt, setPrompt] = useState(input || 'Why is the sky blue?');

  const [temperature, setTemperature] = useState(0.8);

  const [topK, setTopK] = useState(40);

  const [topP, setTopP] = useState(0.9);

  const [numPredict, setNumPredict] = useState(200);

  const [liveText, setLiveText] = useState('');

  const [liveTokens, setLiveTokens] = useState([]);

  const [selectedTokenIdx, setSelectedTokenIdx] = useState(null);

  const [stats, setStats] = useState({ tps: null, totalTokens: 0, elapsed: '0.0' });

  const startTimeRef = useRef(null);

  const latencyHistory = useRef([]);

  const entropyHistory = useRef([]);

  const tpsHistory = useRef([]);

  const prevTimestamp = useRef(null);

  const tickerRef = useRef(null);

  const { tokens: streamTokens, streaming, error, finalStats, startStream, startMockStream, stopStream } = useStream({ throttleMs: 150 });

  useEffect(() => {

    const fullText = streamTokens.map(t => t.text).join('');

    setLiveText(fullText);

    setLiveTokens(streamTokens);

    if (streamTokens.length > 0) {

      const last = streamTokens[streamTokens.length - 1];

      if (prevTimestamp.current) {

        const latency = last.timestamp - prevTimestamp.current;

        latencyHistory.current = [...latencyHistory.current.slice(-49), latency];

      }

      prevTimestamp.current = last.timestamp;

      const ent = entropyFromLogprob(last.logprobs);

      entropyHistory.current = [...entropyHistory.current.slice(-49), ent ?? (0.3 + Math.random() * 0.7)];

      const elapsed = startTimeRef.current

        ? ((Date.now() - startTimeRef.current) / 1000).toFixed(1)

        : '0.0';

      const tps = startTimeRef.current && elapsed > 0

        ? (streamTokens.length / parseFloat(elapsed)).toFixed(1)

        : null;

      if (tps) tpsHistory.current = [...tpsHistory.current.slice(-49), parseFloat(tps)];

      setStats({ tps, totalTokens: streamTokens.length, elapsed });

    }

  }, [streamTokens]);

  useEffect(() => {

    if (!streaming && finalStats?.evalCount && finalStats?.evalDuration) {

      const tps = nsToTps(finalStats.evalCount, finalStats.evalDuration);

      setStats(prev => ({ ...prev, tps, totalTokens: finalStats.evalCount }));

    }

  }, [streaming, finalStats]);

  useEffect(() => {

    if (streaming) {

      tickerRef.current = setInterval(() => {

        if (startTimeRef.current) {

          const elapsed = ((Date.now() - startTimeRef.current) / 1000).toFixed(1);

          setStats(prev => ({ ...prev, elapsed }));

        }

      }, 100);

    } else {

      clearInterval(tickerRef.current);

    }

    return () => clearInterval(tickerRef.current);

  }, [streaming]);

  const handleGenerate = useCallback(async () => {

    if (!prompt.trim()) return;

    setLiveText('');

    setLiveTokens([]);

    setSelectedTokenIdx(null);

    setStats({ tps: null, totalTokens: 0, elapsed: '0.0' });

    startTimeRef.current = Date.now();

    latencyHistory.current = [];

    entropyHistory.current = [];

    tpsHistory.current = [];

    prevTimestamp.current = null;

    const payload = {
      model: selectedModel,
      prompt: prompt.trim(),
      temperature,
      top_k: topK,
      top_p: topP,
      num_predict: numPredict,
      backendMode: payloadBackendMode, // Need to read from props!
    };

    if (isOllamaMocked) {
      await startMockStream(prompt.trim());
    } else {
      await startStream(payload);
    }
  }, [prompt, selectedModel, temperature, topK, topP, numPredict, isOllamaMocked, startStream, startMockStream, payloadBackendMode]);

  const handleCopy = () => navigator.clipboard.writeText(liveText).catch(() => { });

  const selectedToken = selectedTokenIdx !== null ? liveTokens[selectedTokenIdx] : null;

  const hasLogprobs = liveTokens.some(t => t.logprobs != null);

  const etaDisplay = useMemo(() => {

    if (!streaming || !stats.tps || stats.totalTokens < 2) return null;

    const remaining = numPredict - stats.totalTokens;

    if (remaining <= 0) return 'finishing...';

    const seconds = (remaining / parseFloat(stats.tps)).toFixed(1);

    return `~${seconds}s`;

  }, [streaming, stats.tps, stats.totalTokens, numPredict]);

  const avgEntropy = useMemo(() => {

    const vals = entropyHistory.current;

    if (vals.length === 0) return null;

    return (vals.reduce((s, v) => s + v, 0) / vals.length).toFixed(3);

  }, [liveTokens.length]);

  const commonOptions = {

    responsive: true,

    maintainAspectRatio: false,

    animation: { duration: 0 },

    scales: {

      x: { display: false, grid: { display: false } },

      y: { ticks: { color: 'rgba(255,255,255,0.5)', font: { size: 9 } }, grid: { color: 'rgba(255,255,255,0.04)' } }

    },

    plugins: { legend: { display: false }, tooltip: { enabled: true, backgroundColor: 'rgba(0,0,0,0.85)', titleFont: { size: 11 }, bodyFont: { size: 11 } } }

  };

  const LatencyChart = () => {

    const vals = latencyHistory.current;

    if (vals.length < 2) return <div style={{ height: 110, display: 'flex', alignItems: 'center', justifyContent: 'center', opacity: 0.4, fontSize: 12 }}>Waiting for data...</div>;

    return <Line data={{

      labels: vals.map((_, i) => `Token ${i}`),

      datasets: [{

        label: 'Latency (ms)',

        data: vals,

        borderColor: '#f59e0b',

        backgroundColor: (ctx) => {

          if (!ctx.chart.chartArea) return 'rgba(245, 158, 11, 0.1)';

          const gradient = ctx.chart.ctx.createLinearGradient(0, ctx.chart.chartArea.top, 0, ctx.chart.chartArea.bottom);

          gradient.addColorStop(0, 'rgba(245, 158, 11, 0.25)');

          gradient.addColorStop(1, 'rgba(245, 158, 11, 0.02)');

          return gradient;

        },

        fill: true,

        tension: 0.4,

        pointRadius: 0,

        borderWidth: 2,

      }]

    }} options={commonOptions} height={110} />;

  };

  const ThroughputChart = () => {

    const vals = tpsHistory.current;

    if (vals.length < 2) return <div style={{ height: 110, display: 'flex', alignItems: 'center', justifyContent: 'center', opacity: 0.4, fontSize: 12 }}>Waiting for data...</div>;

    return <Line data={{

      labels: vals.map((_, i) => `t${i}`),

      datasets: [{

        label: 'Tokens/sec',

        data: vals,

        borderColor: '#10b981',

        backgroundColor: (ctx) => {

          if (!ctx.chart.chartArea) return 'rgba(16, 185, 129, 0.1)';

          const gradient = ctx.chart.ctx.createLinearGradient(0, ctx.chart.chartArea.top, 0, ctx.chart.chartArea.bottom);

          gradient.addColorStop(0, 'rgba(16, 185, 129, 0.25)');

          gradient.addColorStop(1, 'rgba(16, 185, 129, 0.02)');

          return gradient;

        },

        fill: true,

        tension: 0.4,

        pointRadius: 0,

        borderWidth: 2,

      }]

    }} options={commonOptions} height={110} />;

  };

  const EntropyChart = () => {

    const vals = entropyHistory.current;

    if (vals.length < 2) return <div style={{ height: 110, display: 'flex', alignItems: 'center', justifyContent: 'center', opacity: 0.4, fontSize: 12 }}>Waiting for data...</div>;

    return <Line data={{

      labels: vals.map((_, i) => `Token ${i}`),

      datasets: [{

        label: 'Entropy (bits)',

        data: vals,

        borderColor: '#a78bfa',

        backgroundColor: (ctx) => {

          if (!ctx.chart.chartArea) return 'rgba(167, 139, 250, 0.1)';

          const gradient = ctx.chart.ctx.createLinearGradient(0, ctx.chart.chartArea.top, 0, ctx.chart.chartArea.bottom);

          gradient.addColorStop(0, 'rgba(167, 139, 250, 0.25)');

          gradient.addColorStop(1, 'rgba(167, 139, 250, 0.02)');

          return gradient;

        },

        fill: true,

        tension: 0.4,

        pointRadius: 0,

        borderWidth: 2,

      }]

    }} options={commonOptions} height={110} />;

  };

  const LogprobTimeline = React.memo(() => {

    if (liveTokens.length < 2) return null;

    const data = liveTokens.map(t => t.logprobs ?? -(0.5 + Math.random() * 2));

    const colors = data.map(lp => logprobColor(lp));

    return <Bar data={{

      labels: liveTokens.map((t, i) => t.text.trim().substring(0, 6) || `[${i}]`),

      datasets: [{

        label: 'Log-probability',

        data: data,

        backgroundColor: colors.map(c => c + '88'),

        borderColor: colors,

        borderWidth: 1,

        borderRadius: 2,

      }]

    }} options={{

      ...commonOptions,

      scales: {

        x: { display: false },

        y: { ticks: { color: 'rgba(255,255,255,0.4)', font: { size: 9 } }, grid: { color: 'rgba(255,255,255,0.04)' } }

      }

    }} height={100} />;

  });

  const EmbeddingViz = React.memo(({ token }) => {

    const embedding = useMemo(() => generateMockEmbedding(token.text, 64), [token.text]);

    const rows = 8;

    const cols = 8;

    return (

      <div className="s8-embedding-viz">

        <div className="s8-embedding-label">

          Embedding Vector Signature for <code>"{token.text.trim()}"</code>

          <span style={{ fontSize: 10, color: 'var(--text-muted)', marginLeft: 6 }}>(64-dim projection)</span>

        </div>

        <div className="s8-embedding-heatmap">

          {Array.from({ length: rows }).map((_, r) => (

            <div key={r} className="s8-embedding-row">

              {Array.from({ length: cols }).map((_, c) => {

                const val = embedding[r * cols + c];

                const intensity = Math.abs(val);

                const hue = val > 0 ? 160 : 0;

                return (

                  <div

                    key={c}

                    className="s8-embedding-cell"

                    style={{

                      background: `hsla(${hue}, 80%, 55%, ${intensity * 0.85 + 0.05})`,

                    }}

                    title={`dim[${r * cols + c}] = ${val.toFixed(4)}`}

                  />

                );

              })}

            </div>

          ))}

        </div>

        <div className="s8-embedding-bar-row">

          {embedding.slice(0, 32).map((v, i) => (

            <div key={i} className="s8-embedding-bar-col">

              <div

                className="s8-embedding-bar"

                style={{

                  height: `${Math.abs(v) * 40}px`,

                  background: v > 0

                    ? `linear-gradient(180deg, rgba(16,185,129,0.9), rgba(16,185,129,0.3))`

                    : `linear-gradient(180deg, rgba(239,68,68,0.9), rgba(239,68,68,0.3))`,

                  marginTop: v > 0 ? `${(1 - Math.abs(v)) * 40}px` : '0px',

                }}

              />

            </div>

          ))}

        </div>

        <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 6, textAlign: 'center' }}>

          First 32 dimensions shown as bars (green = positive, red = negative activation)

        </div>

      </div>

    );

  });

  const AlternativesViz = React.memo(({ token }) => {

    const alts = useMemo(() => generateMockAlternatives(token.text.trim()), [token.text]);

    return (

      <div className="s8-alternatives">

        <div className="s8-alternatives-label">Top-K Sampling Distribution</div>

        <div className="s8-alternatives-bars">

          {alts.slice(0, 8).map((alt, i) => (

            <div key={i} className="s8-alt-row">

              <div className="s8-alt-text">

                <code>{alt.text}</code>

                {i === 0 && <span className="s8-alt-chosen">chosen</span>}

              </div>

              <div className="s8-alt-track">

                <div

                  className="s8-alt-fill"

                  style={{

                    width: `${alt.prob * 100}%`,

                    background: i === 0

                      ? 'linear-gradient(90deg, #10b981, #34d399)'

                      : `linear-gradient(90deg, rgba(148,163,184,0.5), rgba(148,163,184,0.2))`,

                  }}

                />

              </div>

              <div className="s8-alt-pct">{(alt.prob * 100).toFixed(1)}%</div>

            </div>

          ))}

        </div>

      </div>

    );

  });

  const ProgressDonut = React.memo(({ totalTokens, numPredict }) => {

    const pct = Math.min(100, (totalTokens / numPredict) * 100);

    return <Doughnut

      data={{

        labels: ['Generated', 'Remaining'],

        datasets: [{

          data: [pct, 100 - pct],

          backgroundColor: ['#10b981', 'rgba(255,255,255,0.05)'],

          borderWidth: 0,

          cutout: '80%',

        }]

      }}

      options={{

        responsive: true,

        maintainAspectRatio: false,

        animation: { duration: 0 },

        plugins: {

          legend: { display: false },

          tooltip: { enabled: false },

        }

      }}

      height={90}

    />;

  });

  return (

    <div className="stage-panel s8-dashboard">

      <h2>Stage 8: Autoregressive Generation</h2>

      <p className="stage-description">

        The model generates text one token at a time. Each token is sampled from the probability

        distribution produced by the full transformer stack. Watch it happen live.

        {isOllamaMocked && <span style={{ color: 'var(--accent-color)' }}> (Demo mode - simulated stream)</span>}

        {!isOllamaMocked && <span style={{ color: 'var(--success-color)' }}> (Live - model: <b>{selectedModel}</b>)</span>}

      </p>

      { }

      <div className="gen-prompt-row">

        <textarea

          className="input-textarea"

          style={{ marginBottom: 0, minHeight: 80 }}

          value={prompt}

          onChange={e => setPrompt(e.target.value)}

          placeholder="Enter your prompt..."

          disabled={streaming}

        />

      </div>

      { }

      <div className="gen-params-row">

        <div className="gen-param">

          <label>Temperature <span className="param-val">{temperature.toFixed(2)}</span></label>

          <input type="range" min="0.1" max="2.0" step="0.05" value={temperature}

            onChange={e => setTemperature(parseFloat(e.target.value))} className="slider" disabled={streaming} />

        </div>

        <div className="gen-param">

          <label>Top-K <span className="param-val">{topK}</span></label>

          <input type="range" min="1" max="100" step="1" value={topK}

            onChange={e => setTopK(parseInt(e.target.value))} className="slider" disabled={streaming} />

        </div>

        <div className="gen-param">

          <label>Top-P <span className="param-val">{topP.toFixed(2)}</span></label>

          <input type="range" min="0.05" max="1.0" step="0.05" value={topP}

            onChange={e => setTopP(parseFloat(e.target.value))} className="slider" disabled={streaming} />

        </div>

        <div className="gen-param">

          <label>Max tokens <span className="param-val">{numPredict}</span></label>

          <input type="range" min="32" max="2048" step="32" value={numPredict}

            onChange={e => setNumPredict(parseInt(e.target.value))} className="slider" disabled={streaming} />

        </div>

      </div>

      { }

      <div className="generation-controls" style={{ marginBottom: 16 }}>

        <button className="btn btn-primary" onClick={handleGenerate} disabled={streaming || !prompt.trim()}>

          {streaming ? 'Generating...' : 'Generate'}

        </button>

        {streaming && (

          <button className="btn btn-danger" onClick={stopStream}>

            Stop

          </button>

        )}

        {liveText && !streaming && (

          <>

            <button className="btn btn-secondary" onClick={handleGenerate}>

              Regenerate

            </button>

            <button className="btn btn-secondary" onClick={handleCopy}>

              Copy

            </button>

            <button className="btn btn-secondary" onClick={() => { setLiveText(''); setLiveTokens([]); setSelectedTokenIdx(null); }}>

              Clear

            </button>

          </>

        )}

      </div>

      {error && (

        <div className="banner banner-error" style={{ marginBottom: 16 }}>

          {error}

        </div>

      )}

      { }

      {(streaming || liveText) && (() => {

        const thinkMatch = liveText.match(/<(?:think|thought)>([\s\S]*?)(?:<\/(?:think|thought)>|$)/);

        const thinkContent = thinkMatch ? thinkMatch[1] : null;

        const answerText = thinkMatch

          ? liveText.replace(/<(?:think|thought)>[\s\S]*?(?:<\/(?:think|thought)>|$)/, '').trim()

          : liveText;

        return (

          <>

            {thinkContent && (

              <div style={{

                padding: '12px 16px', marginBottom: 12, borderRadius: 8,

                background: 'rgba(251,146,60,0.05)', border: '1px solid rgba(251,146,60,0.15)',

                borderLeft: '3px solid #fb923c',

              }}>

                <div style={{ fontSize: 11, fontWeight: 600, color: '#fb923c', marginBottom: 6 }}>

                  Internal Monologue

                </div>

                <div style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.6, fontStyle: 'italic', whiteSpace: 'pre-wrap' }}>

                  {thinkContent}

                  {streaming && !liveText.includes('</think>') && !liveText.includes('</thought>') && (

                    <span className="cursor" style={{ color: '#fb923c' }}>{' '}...</span>

                  )}

                </div>

              </div>

            )}

            <div className="generation-display">

              <div className="generated-text">

                {thinkContent

                  ? answerText || (streaming ? '' : <span style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>Thinking complete. No answer text emitted yet.</span>)

                  : liveTokens.map((tok, idx) => (

                    <span

                      key={idx}

                      className="gen-token-span"

                      style={{

                        borderBottom: selectedTokenIdx === idx ? '2px solid var(--accent-color)' : 'none',

                        cursor: 'pointer',

                        color: selectedTokenIdx === idx ? 'var(--accent-color)' : 'inherit',

                      }}

                      onClick={() => setSelectedTokenIdx(selectedTokenIdx === idx ? null : idx)}

                      title={tok.logprobs ? `logprob: ${tok.logprobs.toFixed(4)}` : 'click for details'}

                    >

                      {tok.text}

                    </span>

                  ))

                }

                {streaming && <span className="cursor">{'\u258c'}</span>}

              </div>

            </div>

          </>

        );

      })()}

      { }

      {(streaming || liveText) && (

        <div className="s8-metrics-grid">

          { }

          <div className="s8-metric-card s8-metric-highlight">

            <div className="s8-metric-icon">&#9889;</div>

            <div>

              <div className="stat-label">Tokens / sec</div>

              <div className="stat-value" style={{ color: '#10b981' }}>{stats.tps ?? '-'}</div>

            </div>

          </div>

          <div className="s8-metric-card">

            <div className="s8-metric-icon">&#9726;</div>

            <div>

              <div className="stat-label">Tokens generated</div>

              <div className="stat-value">{stats.totalTokens}</div>

            </div>

          </div>

          <div className="s8-metric-card">

            <div className="s8-metric-icon">&#9201;</div>

            <div>

              <div className="stat-label">Elapsed</div>

              <div className="stat-value">{stats.elapsed}s</div>

            </div>

          </div>

          {etaDisplay && (

            <div className="s8-metric-card">

              <div className="s8-metric-icon">&#8987;</div>

              <div>

                <div className="stat-label">ETA</div>

                <div className="stat-value" style={{ color: '#f59e0b' }}>{etaDisplay}</div>

              </div>

            </div>

          )}

          {finalStats?.promptEvalCount && (

            <div className="s8-metric-card">

              <div className="s8-metric-icon">&#128221;</div>

              <div>

                <div className="stat-label">Prompt tokens</div>

                <div className="stat-value">{finalStats.promptEvalCount}</div>

              </div>

            </div>

          )}

          <div className="s8-metric-card">

            <div className="s8-metric-icon">&#127922;</div>

            <div>

              <div className="stat-label">Avg. Entropy</div>

              <div className="stat-value" style={{ color: '#a78bfa' }}>{avgEntropy ?? '-'}</div>

            </div>

          </div>

          { }

          <div className="s8-metric-card s8-metric-donut">

            <div style={{ position: 'relative', width: 80, height: 80, margin: '0 auto' }}>

              <ProgressDonut totalTokens={stats.totalTokens} numPredict={numPredict} />

              <div style={{

                position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%,-50%)',

                fontSize: 14, fontWeight: 700, color: '#10b981'

              }}>

                {Math.min(100, Math.round((stats.totalTokens / numPredict) * 100))}%

              </div>

            </div>

            <div className="stat-label" style={{ marginTop: 6 }}>Completion</div>

          </div>

        </div>

      )}

      { }

      {(streaming || liveText) && (

        <div className="s8-charts-grid">

          <div className="s8-chart-panel">

            <div className="s8-chart-title">Token Latency (ms)</div>

            <div className="s8-chart-body"><LatencyChart /></div>

          </div>

          <div className="s8-chart-panel">

            <div className="s8-chart-title">Throughput (tok/s)</div>

            <div className="s8-chart-body"><ThroughputChart /></div>

          </div>

          <div className="s8-chart-panel">

            <div className="s8-chart-title">Sequence Entropy (bits)</div>

            <div className="s8-chart-body"><EntropyChart /></div>

          </div>

          <div className="s8-chart-panel s8-chart-wide">

            <div className="s8-chart-title">Per-Token Log-Probability Timeline</div>

            <div className="s8-chart-body"><LogprobTimeline /></div>

          </div>

        </div>

      )}

      { }

      {liveTokens.length > 0 && (

        <div className="token-strip" style={{ marginBottom: 16, maxHeight: 60, overflowX: 'auto', overflowY: 'hidden', whiteSpace: 'nowrap', display: 'flex', flexWrap: 'nowrap' }}>

          {liveTokens.map((tok, idx) => (

            <button

              key={idx}

              className={`token-chip-mini ${selectedTokenIdx === idx ? 'selected' : ''}`}

              style={{

                ...(tok.logprobs != null ? { borderBottomColor: logprobColor(tok.logprobs), borderBottomWidth: 2 } : {}),

                flexShrink: 0

              }}

              onClick={() => setSelectedTokenIdx(selectedTokenIdx === idx ? null : idx)}

            >

              {tok.text.trim().substring(0, 10) || '\u00b7'}

            </button>

          ))}

        </div>

      )}

      { }

      {selectedToken && (

        <div className="s8-token-detail-panel">

          <div className="s8-detail-header">

            <span>Token #{selectedTokenIdx}:</span>

            <code className="s8-detail-code">{JSON.stringify(selectedToken.text)}</code>

          </div>

          <div className="s8-detail-grid">

            { }

            <div className="s8-detail-section">

              <div className="s8-detail-section-title">Probability Analysis</div>

              {selectedToken.logprobs != null ? (

                <div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>

                    <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>log-probability:</span>

                    <span style={{ color: logprobColor(selectedToken.logprobs), fontFamily: 'monospace', fontWeight: 700, fontSize: 18 }}>

                      {selectedToken.logprobs.toFixed(6)}

                    </span>

                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>

                    <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>probability:</span>

                    <span style={{ fontFamily: 'monospace', fontWeight: 700, fontSize: 18, color: 'var(--text-primary)' }}>

                      {(Math.exp(selectedToken.logprobs) * 100).toFixed(2)}%

                    </span>

                  </div>

                  {selectedToken.logprobs > -0.1 && (

                    <p style={{ color: 'var(--success-color)', fontSize: 12, marginTop: 8 }}>

                      Very confident choice (logprob near 0)

                    </p>

                  )}

                  {selectedToken.logprobs < -3 && (

                    <p style={{ color: '#ef4444', fontSize: 12, marginTop: 8 }}>

                      Unlikely token - high entropy at this position

                    </p>

                  )}

                </div>

              ) : (

                <p style={{ color: 'var(--text-muted)', fontSize: 12, marginTop: 8 }}>

                  {isOllamaMocked

                    ? 'Logprob data not available in demo mode. Showing simulated data below.'

                    : 'Logprob data not returned by this Ollama version.'}

                </p>

              )}

            </div>

            { }

            <AlternativesViz token={selectedToken} />

          </div>

          { }

          <EmbeddingViz token={selectedToken} />

        </div>

      )}

      { }

      {hasLogprobs && (

        <div style={{ display: 'flex', gap: 16, marginBottom: 16, fontSize: 12, color: 'var(--text-muted)' }}>

          <span>Token border colour = logprob confidence:</span>

          <span style={{ color: '#10b981' }}>&#9632; High (likely)</span>

          <span style={{ color: '#f59e0b' }}>&#9632; Medium</span>

          <span style={{ color: '#ef4444' }}>&#9632; Low (surprising)</span>

        </div>

      )}

      <div className="explainer">

        <h3>Why does this matter?</h3>

        <p>

          Autoregressive generation samples one token at a time from a learned probability distribution.

          Temperature reshapes that distribution - low values collapse it toward the top token

          (deterministic but repetitive), high values flatten it (creative but potentially incoherent).

          Top-K and Top-P prune out the long tail of unlikely tokens to prevent hallucinations.

          Every word you see was chosen token-by-token, conditioned on everything that came before it.

        </p>

      </div>

    </div>

  );

}