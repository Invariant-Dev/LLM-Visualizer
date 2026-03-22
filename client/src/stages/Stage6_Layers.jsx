import React, { useState, useMemo } from 'react';
function seededRandom(seed) {
  let x = Math.sin(seed + 1) * 10000;
  return x - Math.floor(x);
}
export default function Stage6_Layers({ modelInfo, goToNextStage }) {
  const [selectedLayer, setSelectedLayer] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [beadPosition, setBeadPosition] = useState(0);
  const layerCount = modelInfo?.llama?.block_count || 32;
  const embeddingDim = modelInfo?.llama?.embedding_length || 4096;
  const headCount = modelInfo?.llama?.attention?.head_count || 32;
  const ffnDim = modelInfo?.llama?.feed_forward_length || 11008;
  const layerSaturation = useMemo(() => {
    return Array.from({ length: layerCount }, (_, i) => {
      const base = 0.3 + seededRandom(i * 7) * 0.7;
      const curve = Math.sin((i / layerCount) * Math.PI);
      return Math.min(base * (0.4 + curve * 0.6), 1.0);
    });
  }, [layerCount]);
  const layerEntropy = useMemo(() => {
    return Array.from({ length: layerCount }, (_, i) => {
      return 1.0 - (i / layerCount) * 0.6 + seededRandom(i * 13) * 0.2;
    });
  }, [layerCount]);
  const runForwardPass = async () => {
    setIsRunning(true);
    for (let i = 0; i <= layerCount; i++) {
      setBeadPosition(i);
      await new Promise(resolve => setTimeout(resolve, 80));
    }
    setIsRunning(false);
  };
  const satColor = (val) => {
    if (val > 0.7) return '#10b981';
    if (val > 0.4) return '#f59e0b';
    return '#ef4444';
  };
  return (
    <div className="stage-panel">
      <h2>Stage 6: Layer Stack</h2>
      <p className="stage-description">
        The transformer has {layerCount} identical layers stacked vertically. Each layer performs attention, then FFN,
        with residual connections and normalization around each.
      </p>
      <div className="model-info-bar">
        <div className="info-item">
          <div className="info-label">Total Layers</div>
          <div className="info-value">{layerCount}</div>
        </div>
        <div className="info-item">
          <div className="info-label">Model Dim</div>
          <div className="info-value">{embeddingDim}</div>
        </div>
        <div className="info-item">
          <div className="info-label">Attention Heads</div>
          <div className="info-value">{headCount}</div>
        </div>
        <div className="info-item">
          <div className="info-label">FFN Hidden</div>
          <div className="info-value">{ffnDim}</div>
        </div>
      </div>
      <div style={{ display: 'flex', gap: 16 }}>
        {}
        <div className="layer-stack" style={{ flex: 1 }}>
          {Array.from({ length: layerCount }).map((_, idx) => (
            <div
              key={idx}
              className={`layer-block ${beadPosition === idx + 1 ? 'active' : ''} ${selectedLayer === idx ? 'selected' : ''}`}
              onClick={() => setSelectedLayer(selectedLayer === idx ? null : idx)}
              style={{ display: 'flex', alignItems: 'center', gap: 8 }}
            >
              <div style={{ flex: 1 }}>
                <div className="layer-number">Layer {idx + 1}</div>
                <div className="layer-pipeline">
                  <div className="pipeline-stage">LN</div>
                  <div className="pipeline-stage">MHA</div>
                  <div className="pipeline-stage">+</div>
                  <div className="pipeline-stage">LN</div>
                  <div className="pipeline-stage">FFN</div>
                  <div className="pipeline-stage">+</div>
                </div>
              </div>
              {}
              <div style={{ width: 60, display: 'flex', flexDirection: 'column', gap: 2 }}>
                <div style={{ height: 4, borderRadius: 2, background: 'rgba(255,255,255,0.06)', overflow: 'hidden' }}>
                  <div style={{
                    width: `${layerSaturation[idx] * 100}%`, height: '100%',
                    background: satColor(layerSaturation[idx]),
                    transition: 'width 0.3s',
                  }} />
                </div>
                <span style={{ fontSize: 8, color: 'var(--text-muted)', textAlign: 'right' }}>
                  {(layerSaturation[idx] * 100).toFixed(0)}%
                </span>
              </div>
              {beadPosition === idx + 1 && <div className="bead"></div>}
            </div>
          ))}
        </div>
        {}
        <div style={{
          width: 200, padding: 14, borderRadius: 8,
          background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)',
          alignSelf: 'flex-start', position: 'sticky', top: 16,
        }}>
          <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 10, color: 'var(--text-primary)' }}>
            Layer Saturation
          </div>
          <p style={{ fontSize: 11, color: 'var(--text-secondary)', marginBottom: 12 }}>
            How "active" each layer's hidden state is. High saturation = strong feature extraction.
          </p>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', display: 'flex', flexDirection: 'column', gap: 4 }}>
            <span><span style={{ color: '#10b981' }}>&#9632;</span> High (&gt;70%)</span>
            <span><span style={{ color: '#f59e0b' }}>&#9632;</span> Medium (40-70%)</span>
            <span><span style={{ color: '#ef4444' }}>&#9632;</span> Low (&lt;40%)</span>
          </div>
          {selectedLayer !== null && (
            <div style={{ marginTop: 16, paddingTop: 12, borderTop: '1px solid rgba(255,255,255,0.08)' }}>
              <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6, color: '#fb923c' }}>
                Layer {selectedLayer + 1}
              </div>
              <div style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
                <p>Heads: {headCount} x {embeddingDim / headCount} dims</p>
                <p>FFN: {embeddingDim} -&gt; {ffnDim} -&gt; {embeddingDim}</p>
                <p>Saturation: {(layerSaturation[selectedLayer] * 100).toFixed(1)}%</p>
                <p>Entropy: {layerEntropy[selectedLayer].toFixed(3)}</p>
              </div>
            </div>
          )}
        </div>
      </div>
      <button className="btn btn-primary" onClick={runForwardPass} disabled={isRunning} style={{ marginTop: 16, marginBottom: 16 }}>
        {isRunning ? 'Running...' : 'Run Forward Pass'}
      </button>
      <div className="explainer">
        <h3>Why does this matter?</h3>
        <p>
          Stacking layers is where depth comes from. Each layer refines the representation further.
          Deeper models can learn more abstract concepts, but require more computation and data.
          The saturation bars show each layer's contribution intensity to the final prediction.
        </p>
      </div>
    </div>
  );
}