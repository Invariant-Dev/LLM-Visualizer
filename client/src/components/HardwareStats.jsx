import React, { useMemo } from 'react';
function estimateVram(modelInfo) {
  if (!modelInfo) return null;
  const params = parseFloat(modelInfo.parameters) || 7;
  const quant = modelInfo.quantization_level || 'Q4_K_M';
  const bitsMap = {
    'Q2_K': 2.5, 'Q3_K_S': 3, 'Q3_K_M': 3.5, 'Q3_K_L': 3.5,
    'Q4_0': 4, 'Q4_1': 4.5, 'Q4_K_S': 4.5, 'Q4_K_M': 4.5,
    'Q5_0': 5, 'Q5_1': 5.5, 'Q5_K_S': 5, 'Q5_K_M': 5.5,
    'Q6_K': 6.5, 'Q8_0': 8, 'F16': 16, 'F32': 32,
  };
  const bits = bitsMap[quant] || 4.5;
  const modelGB = (params * 1e9 * bits) / 8 / 1e9;
  const ctxLen = modelInfo.llama?.context_length || 4096;
  const layers = modelInfo.llama?.block_count || 32;
  const heads = modelInfo.llama?.attention?.head_count || 32;
  const dim = modelInfo.llama?.embedding_length || 4096;
  const headDim = dim / heads;
  const kvGB = (2 * layers * ctxLen * heads * headDim * 2) / 1e9;
  return {
    modelGB: modelGB.toFixed(2),
    kvCacheGB: kvGB.toFixed(2),
    totalGB: (modelGB + kvGB).toFixed(2),
    bits,
    params,
    quant,
  };
}
export default function HardwareStats({ modelInfo, selectedModel }) {
  const stats = useMemo(() => estimateVram(modelInfo), [modelInfo]);
  if (!stats) return null;
  const pct = Math.min((parseFloat(stats.totalGB) / 24) * 100, 100);
  return (
    <div className="hw-stats" style={{
      display: 'flex', alignItems: 'center', gap: 12,
      padding: '6px 14px', borderRadius: 8,
      background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)',
      fontSize: 11, color: 'var(--text-secondary)',
    }}>
      <span title={selectedModel} style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
        {stats.params}B
      </span>
      <span>{stats.quant}</span>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 2, minWidth: 100 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span>VRAM</span>
          <span style={{ fontWeight: 600, color: pct > 80 ? '#ef4444' : '#10b981' }}>
            {stats.totalGB} GB
          </span>
        </div>
        <div style={{
          width: '100%', height: 4, borderRadius: 2,
          background: 'rgba(255,255,255,0.1)', overflow: 'hidden'
        }}>
          <div style={{
            width: `${pct}%`, height: '100%', borderRadius: 2,
            background: pct > 80 ? '#ef4444' : pct > 50 ? '#f59e0b' : '#10b981',
            transition: 'width 0.3s'
          }} />
        </div>
      </div>
      <span style={{ opacity: 0.6 }}>
        Model {stats.modelGB} + KV {stats.kvCacheGB}
      </span>
    </div>
  );
}