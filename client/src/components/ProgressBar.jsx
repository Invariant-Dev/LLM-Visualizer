import React from 'react';
import './ProgressBar.css';
export default function ProgressBar({ currentStage, totalStages }) {
  const progress = (currentStage / totalStages) * 100;
  return (
    <div className="progress-bar-container">
      <div className="progress-bar">
        <div className="progress-fill" style={{ width: `${progress}%` }}></div>
      </div>
      <div className="progress-text">
        {currentStage} / {totalStages}
      </div>
    </div>
  );
}