import React, { useState } from 'react';
import { useAppContext } from '../context/AppContext';
import axios from 'axios';

export default function BackendToggle() {
  const { backendMode, setBackendMode, pytorchAvailable, pytorchModel } = useAppContext();
  const [starting, setStarting] = useState(false);

  const handleStartPytorch = async () => {
    if (pytorchAvailable || starting) return;
    setStarting(true);
    try {
      await axios.post('/api/pytorch/start');
      // The context polling will pick it up once it's actually alive
    } catch {
      setStarting(false);
    }
  };

  return (
    <div className="backend-toggle" style={{
      display: 'flex', alignItems: 'center', gap: 4,
      background: 'rgba(255,255,255,0.04)', borderRadius: 6,
      padding: '3px 4px', border: '1px solid rgba(255,255,255,0.08)',
    }}>
      <button
        onClick={() => setBackendMode('ollama')}
        title="Use local Ollama server"
        style={{
          padding: '4px 10px', fontSize: 11, borderRadius: 4, border: 'none', cursor: 'pointer',
          background: backendMode === 'ollama' ? 'rgba(16,185,129,0.2)' : 'transparent',
          color: backendMode === 'ollama' ? '#10b981' : 'var(--text-muted)',
          fontWeight: backendMode === 'ollama' ? 600 : 400,
          transition: 'all 0.2s',
        }}
      >
        Ollama
      </button>
      
      {pytorchAvailable ? (
        <button
          onClick={() => setBackendMode('pytorch')}
          title={`PyTorch native (${pytorchModel})`}
          style={{
            padding: '4px 10px', fontSize: 11, borderRadius: 4, border: 'none', cursor: 'pointer',
            background: backendMode === 'pytorch' ? 'rgba(167,139,250,0.2)' : 'transparent',
            color: backendMode === 'pytorch' ? '#a78bfa' : 'var(--text-muted)',
            fontWeight: backendMode === 'pytorch' ? 600 : 400,
            transition: 'all 0.2s',
          }}
        >
          PyTorch
        </button>
      ) : (
        <button
          onClick={handleStartPytorch}
          title="Start the PyTorch Python Server in a new window"
          style={{
            padding: '4px 10px', fontSize: 11, borderRadius: 4, border: '1px dashed rgba(167,139,250,0.4)',
            cursor: starting ? 'wait' : 'pointer',
            background: 'transparent',
            color: starting ? 'rgba(167,139,250,0.6)' : '#a78bfa',
            fontWeight: 400,
            transition: 'all 0.2s',
          }}
        >
          {starting ? 'Starting...' : '+ Start PyTorch'}
        </button>
      )}

      <button
        onClick={() => setBackendMode('mock')}
        title="Use mock/demo data"
        style={{
          padding: '4px 10px', fontSize: 11, borderRadius: 4, border: 'none', cursor: 'pointer',
          background: backendMode === 'mock' ? 'rgba(245,158,11,0.2)' : 'transparent',
          color: backendMode === 'mock' ? '#f59e0b' : 'var(--text-muted)',
          fontWeight: backendMode === 'mock' ? 600 : 400,
          transition: 'all 0.2s',
        }}
      >
        Mock
      </button>
    </div>
  );
}
