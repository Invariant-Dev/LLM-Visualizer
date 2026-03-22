import React from 'react';
export default function StageNav({ currentStage, stages, onSelectStage }) {
  return (
    <nav className="stage-nav">
      {stages.map((stage, idx) => (
        <button
          key={stage.id}
          className={`stage-nav-item ${currentStage === stage.id ? 'active' : ''}`}
          onClick={() => onSelectStage(stage.id)}
          title={stage.name}
        >
          <span className="stage-number">{stage.id}</span>
          <span className="stage-label">{stage.name}</span>
        </button>
      ))}
    </nav>
  );
}