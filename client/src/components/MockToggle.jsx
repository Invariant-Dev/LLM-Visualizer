import React from 'react';
export default function MockToggle({ isOllamaMocked, setIsOllamaMocked }) {
  return (
    <label className="mock-toggle">
      <input
        type="checkbox"
        checked={isOllamaMocked}
        onChange={(e) => setIsOllamaMocked(e.target.checked)}
      />
      <span>{isOllamaMocked ? '🔴 Mock Mode' : '🟢 Ollama Connected'}</span>
    </label>
  );
}