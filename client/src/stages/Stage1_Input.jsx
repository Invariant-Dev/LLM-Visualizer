import React, { useState } from 'react';
export default function Stage1_Input({ input, setInput, goToNextStage }) {
  const charCount = input.length;
  const byteCount = new TextEncoder().encode(input).length;
  const estimatedTokens = Math.ceil(charCount / 4);
  return (
    <div className="stage-panel">
      <h2>Stage 1: Raw Input</h2>
      <p className="stage-description">
        Everything starts with text. Let's see how it's represented at the byte level.
      </p>
      <textarea
        value={input}
        onChange={(e) => setInput(e.target.value)}
        className="input-textarea"
        placeholder="Enter your prompt..."
      />
      <div className="stats-row">
        <div className="stat">
          <div className="stat-label">Characters</div>
          <div className="stat-value">{charCount}</div>
        </div>
        <div className="stat">
          <div className="stat-label">Bytes (UTF-8)</div>
          <div className="stat-value">{byteCount}</div>
        </div>
        <div className="stat">
          <div className="stat-label">Est. Tokens</div>
          <div className="stat-value">{estimatedTokens}</div>
        </div>
      </div>
      <div className="character-chips">
        {input.split('').map((char, idx) => (
          <div key={idx} className="character-chip">
            <div className="chip-char">{char === ' ' ? '·' : char}</div>
            <div className="chip-code">U+{char.charCodeAt(0).toString(16).toUpperCase().padStart(4, '0')}</div>
          </div>
        ))}
      </div>
      <div className="explainer">
        <h3>Why does this matter?</h3>
        <p>
          LLMs don't see text as humans do. They first convert it into numeric tokens, and before that,
          the raw bytes. UTF-8 encoding means some characters span multiple bytes. This affects how the
          tokenizer splits the text.
        </p>
      </div>
    </div>
  );
}