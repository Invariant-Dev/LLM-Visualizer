import React, { useState, useMemo } from 'react';
function applyActivation(x, fn) {
  if (fn === 'relu') return Math.max(0, x);
  if (fn === 'silu') return x / (1 + Math.exp(-x));
  return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))));
}
function vecMagnitude(arr) {
  return Math.sqrt(arr.reduce((s, v) => s + v * v, 0));
}
export default function Stage5_FFN({ tokens, goToNextStage }) {
  const [isActivating, setIsActivating] = useState(false);
  const [activationFunction, setActivationFunction] = useState('gelu');
  const [selectedNodeLayer, setSelectedNodeLayer] = useState(null);
  const [activations, setActivations] = useState({
    input: Array(16).fill(0).map(() => (Math.random() - 0.5) * 2),
    hidden: Array(32).fill(0).map(() => (Math.random() - 0.5) * 2),
    output: Array(16).fill(0).map(() => (Math.random() - 0.5) * 2),
  });
  const magnitudes = useMemo(() => ({
    input: vecMagnitude(activations.input),
    hidden: vecMagnitude(activations.hidden),
    output: vecMagnitude(activations.output),
  }), [activations]);
  const runActivation = () => {
    setIsActivating(true);
    const newInput = Array(16).fill(0).map(() => (Math.random() - 0.5) * 2);
    const weights1 = Array(32).fill(0).map(() => Array(16).fill(0).map(() => (Math.random() - 0.5)));
    const newHidden = weights1.map(w => {
      const z = w.reduce((sum, wi, i) => sum + wi * newInput[i], 0);
      return applyActivation(z, activationFunction);
    });
    const weights2 = Array(16).fill(0).map(() => Array(32).fill(0).map(() => (Math.random() - 0.5)));
    const newOutput = weights2.map(w =>
      w.reduce((sum, wi, i) => sum + wi * newHidden[i], 0)
    );
    setActivations({ input: newInput, hidden: newHidden, output: newOutput });
    setTimeout(() => setIsActivating(false), 1200);
  };
  const getActivationCurve = () => {
    const points = [];
    for (let x = -3; x <= 3; x += 0.1) {
      const y = applyActivation(x, activationFunction);
      points.push({ x: ((x + 3) / 6) * 200, y: (100 - y * 30) });
    }
    return points;
  };
  const nodeBg = (val, base) => {
    const abs = Math.min(Math.abs(val), 2) / 2;
    const sign = val >= 0;
    if (base === 'input') return sign ? `rgba(16,185,129,${abs})` : `rgba(239,68,68,${abs})`;
    if (base === 'hidden') return sign ? `rgba(251,146,60,${abs})` : `rgba(239,68,68,${abs})`;
    return sign ? `rgba(59,130,246,${abs})` : `rgba(239,68,68,${abs})`;
  };
  const activationCurve = getActivationCurve();
  const pathStr = activationCurve.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ');
  const MagBar = ({ label, value, color }) => (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11, marginBottom: 4 }}>
      <span style={{ color: 'var(--text-muted)', width: 50 }}>{label}</span>
      <div style={{ flex: 1, height: 6, borderRadius: 3, background: 'rgba(255,255,255,0.06)' }}>
        <div style={{
          width: `${Math.min((value / 6) * 100, 100)}%`, height: '100%',
          borderRadius: 3, background: color, transition: 'width 0.4s',
        }} />
      </div>
      <span style={{ color: 'var(--text-secondary)', fontFamily: 'monospace', fontSize: 10, width: 36 }}>
        {value.toFixed(2)}
      </span>
    </div>
  );
  return (
    <div className="stage-panel">
      <h2>Stage 5: Feed-Forward Network</h2>
      <p className="stage-description">
        After attention, each token is processed by an MLP. Input (16 dims) expands to Hidden (64 dims) then
        contracts to Output (16 dims). An activation function introduces non-linearity.
      </p>
      {}
      <div style={{
        padding: '12px 16px', marginBottom: 16, borderRadius: 8,
        background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)',
      }}>
        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8, color: 'var(--text-primary)' }}>
          Vector Magnitude Trace
        </div>
        <MagBar label="Input" value={magnitudes.input} color="#10b981" />
        <MagBar label="Hidden" value={magnitudes.hidden} color="#fb923c" />
        <MagBar label="Output" value={magnitudes.output} color="#3b82f6" />
      </div>
      <div className="ffn-container">
        <div className="ffn-layers">
          <div className="network-layer">
            <div className="layer-label">Input</div>
            {activations.input.map((val, idx) => (
              <div
                key={idx}
                className="node"
                style={{
                  backgroundColor: nodeBg(val, 'input'),
                  border: '1px solid rgba(255,255,255,0.1)',
                  animation: isActivating ? `pulse 0.3s ${idx * 20}ms` : 'none',
                }}
                title={`dim[${idx}] = ${val.toFixed(4)}`}
              />
            ))}
          </div>
          <div className="layer-arrow">{'\u2192'}</div>
          <div className="network-layer">
            <div className="layer-label">Hidden (64)</div>
            <div className="network-subgrid">
              {activations.hidden.map((val, idx) => (
                <div
                  key={idx}
                  className="node node-small"
                  style={{
                    backgroundColor: nodeBg(val, 'hidden'),
                    animation: isActivating ? `pulse 0.3s ${300 + idx * 10}ms` : 'none',
                  }}
                />
              ))}
            </div>
          </div>
          <div className="layer-arrow">{'\u2192'}</div>
          <div className="network-layer">
            <div className="layer-label">Output</div>
            {activations.output.map((val, idx) => (
              <div
                key={idx}
                className="node"
                style={{
                  backgroundColor: nodeBg(val, 'output'),
                  border: '1px solid rgba(255,255,255,0.1)',
                  animation: isActivating ? `pulse 0.3s ${600 + idx * 20}ms` : 'none',
                }}
                title={`dim[${idx}] = ${val.toFixed(4)}`}
              />
            ))}
          </div>
        </div>
        <div className="activation-function">
          <h4>Activation Function</h4>
          <svg width="200" height="120">
            <rect x="0" y="0" width="200" height="120" fill="transparent" stroke="rgba(255,255,255,0.1)" strokeWidth="1" />
            <path d={pathStr} stroke="#fb923c" strokeWidth="2" fill="none" />
            <line x1="100" y1="0" x2="100" y2="120" stroke="rgba(255,255,255,0.1)" strokeWidth="0.5" strokeDasharray="2,2" />
            <line x1="0" y1="50" x2="200" y2="50" stroke="rgba(255,255,255,0.1)" strokeWidth="0.5" strokeDasharray="2,2" />
          </svg>
          <select value={activationFunction} onChange={(e) => setActivationFunction(e.target.value)}>
            <option value="gelu">GeLU</option>
            <option value="relu">ReLU</option>
            <option value="silu">SiLU</option>
          </select>
        </div>
      </div>
      <button className="btn btn-primary" onClick={runActivation} disabled={isActivating} style={{ marginBottom: 16 }}>
        {isActivating ? 'Activating...' : 'Activate'}
      </button>
      <div className="explainer">
        <h3>Why does this matter?</h3>
        <p>
          Each transformer layer alternates attention (learns context) with FFN (learns non-linear transformations).
          The FFN expansion and contraction allows the model to learn complex feature interactions.
          The magnitude trace shows how the vector's "energy" transforms through each stage.
        </p>
      </div>
    </div>
  );
}