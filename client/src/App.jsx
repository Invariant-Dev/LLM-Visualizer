import React, { useState, useEffect } from 'react';
import { useAppContext } from './context/AppContext';
import ErrorBoundary from './components/ErrorBoundary';
import Tour from './components/Tour';
import BackendToggle from './components/BackendToggle';
import ProgressBar from './components/ProgressBar';
import MockToggle from './components/MockToggle';
import HardwareStats from './components/HardwareStats';
import StageNav from './components/StageNav';
import Stage1_Input from './stages/Stage1_Input';
import Stage2_Tokenizer from './stages/Stage2_Tokenizer';
import Stage3_Embeddings from './stages/Stage3_Embeddings';
import Stage4_Attention from './stages/Stage4_Attention';
import Stage5_FFN from './stages/Stage5_FFN';
import Stage6_Layers from './stages/Stage6_Layers';
import Stage7_Softmax from './stages/Stage7_Softmax';
import Stage8_Generation from './stages/Stage8_Generation';
const STAGES = [
  { id: 1, name: 'Raw Input',      component: Stage1_Input },
  { id: 2, name: 'Tokenizer',      component: Stage2_Tokenizer },
  { id: 3, name: 'Embeddings',     component: Stage3_Embeddings },
  { id: 4, name: 'Attention',      component: Stage4_Attention },
  { id: 5, name: 'Feed-Forward',   component: Stage5_FFN },
  { id: 6, name: 'Layer Stack',    component: Stage6_Layers },
  { id: 7, name: 'Softmax & Temp', component: Stage7_Softmax },
  { id: 8, name: 'Generation',     component: Stage8_Generation },
];
export default function App() {
  const {
    isOllamaMocked, setIsOllamaMocked,
    backendMode, pytorchAvailable, pytorchModel,
    input, setInput,
    tokens, setTokens,
    modelInfo,
    availableModels,
    selectedModel, setSelectedModel,
    loadingModel,
  } = useAppContext();
  const [currentStage, setCurrentStage] = useState(1);
  const goToStage = (stageId) => {
    setCurrentStage(Math.max(1, Math.min(stageId, STAGES.length)));
  };
  useEffect(() => {
    const handler = (e) => {
      if (['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return;
      if (e.key === 'ArrowRight') goToStage(currentStage + 1);
      if (e.key === 'ArrowLeft')  goToStage(currentStage - 1);
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [currentStage]);
  const StageComponent = STAGES[currentStage - 1].component;
  const noModelsWarning = !isOllamaMocked && !loadingModel && availableModels.length === 0;

  const handleExport = () => {
    const dataStr = JSON.stringify({ input, tokens, modelInfo, isOllamaMocked, backendMode }, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `llm_session_stage${currentStage}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="app-container">
      <div className="app-header">
        <h1>LLM Internals Explorer</h1>
        <div className="header-controls">
          <BackendToggle />
          { }
          {backendMode === 'ollama' && !isOllamaMocked && availableModels.length > 0 && (
            <select
              className="model-picker"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              title="Select Ollama model"
            >
              {availableModels.map(m => (
                <option key={m.name} value={m.name}>{m.name}</option>
              ))}
            </select>
          )}
          {backendMode === 'pytorch' && pytorchAvailable && (
            <span style={{
              padding: '4px 10px', fontSize: 12, borderRadius: 4,
              background: 'rgba(167,139,250,0.1)', color: '#a78bfa',
              border: '1px solid rgba(167,139,250,0.2)',
            }}>
              {pytorchModel}
            </span>
          )}
          <button className="btn btn-secondary" onClick={handleExport} title="Export current session as JSON" style={{ padding: '6px 12px', fontSize: 13 }}>
            Export JSON
          </button>
          <Tour />
          <HardwareStats modelInfo={modelInfo} selectedModel={selectedModel} />
          <MockToggle isOllamaMocked={isOllamaMocked} setIsOllamaMocked={setIsOllamaMocked} />
        </div>
      </div>
      {!loadingModel && backendMode === 'ollama' && isOllamaMocked && (
        <div className="banner banner-info">
          Ollama not detected - running with demo data.
          Install Ollama and run <code>ollama serve</code> to use your own models.
        </div>
      )}
      {backendMode === 'pytorch' && !pytorchAvailable && (
        <div className="banner banner-warn">
          PyTorch backend not detected.
          Run <code>cd pytorch_server && pip install -r requirements.txt && python main.py</code>
        </div>
      )}
      {backendMode === 'pytorch' && pytorchAvailable && (
        <div className="banner" style={{ background: 'rgba(167,139,250,0.08)', borderColor: 'rgba(167,139,250,0.3)', color: '#a78bfa' }}>
          PyTorch native mode active - using real model internals from <b>{pytorchModel}</b>
        </div>
      )}
      {noModelsWarning && backendMode === 'ollama' && (
        <div className="banner banner-warn">
          Ollama is running but no models are pulled.
          Run <code>ollama pull llama3</code> (or any model) then refresh.
        </div>
      )}
      <ProgressBar currentStage={currentStage} totalStages={STAGES.length} />
      <div className="app-content">
        <StageNav currentStage={currentStage} stages={STAGES} onSelectStage={goToStage} />
        <div className="stage-container">
          <ErrorBoundary key={currentStage}>
            <StageComponent
              input={input}
              setInput={setInput}
              tokens={tokens}
              setTokens={setTokens}
              isOllamaMocked={isOllamaMocked}
              backendMode={backendMode}
              modelInfo={modelInfo}
              selectedModel={selectedModel || 'llama2'}
              goToNextStage={() => goToStage(currentStage + 1)}
            />
          </ErrorBoundary>
        </div>
      </div>
      <div className="app-footer">
        <span className="stage-indicator">Stage {currentStage} of {STAGES.length}</span>
      </div>
    </div>
  );
}