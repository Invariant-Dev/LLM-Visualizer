import React, { useState, useEffect } from 'react';
import { useOllama } from '../hooks/useOllama';
import { tokenColors } from '../utils/colorScale';
export default function Stage2_Tokenizer({ input, setInput, tokens, setTokens, isOllamaMocked, selectedModel, goToNextStage }) {
  const [selectedToken, setSelectedToken] = useState(0);
  const { data: tokenData, loading, error } = useOllama('tokenize', { model: selectedModel || 'llama2', prompt: input }, isOllamaMocked);
  useEffect(() => {
    if (tokenData && tokenData.tokens) {
      let currentPos = 0;
      setTokens(
        tokenData.tokens.map((token, idx) => {
          const isNum = typeof token === 'number';
          let text = token;
          if (isNum) {
            const chunkSize = Math.ceil((input.length - currentPos) / (tokenData.tokens.length - idx));
            text = input.slice(currentPos, currentPos + chunkSize);
            currentPos += chunkSize;
          }
          return {
            text: text,
            id: isNum ? token : (tokenData.token_ids ? tokenData.token_ids[idx] : idx),
            index: idx
          };
        })
      );
    } else if (error && !loading) {
      const fallbackTokens = input.match(/[\w]+|[^\s\w]+|\s+/g) || [input];
      setTokens(
        fallbackTokens.map((tok, idx) => ({
          text: tok,
          id: 99000 + idx,
          index: idx
        }))
      );
    }
  }, [tokenData, error, loading, input, setTokens]);
  const getTokenColor = (tokenText) => {
    const text = String(tokenText || '');
    if (text.startsWith('<') && text.endsWith('>')) return tokenColors.special;
    if (text.match(/^[^\w\s]/)) return tokenColors.punctuation;
    if (text.startsWith(' ') || text.startsWith('Ġ') || text.startsWith(' ')) return tokenColors.subword;
    return tokenColors.fullWord;
  };
  if (loading && tokens.length === 0) {
    return <div className="stage-panel"><div className="loading">Loading tokenizer...</div></div>;
  }
  const compressionRatio = (input.length / Math.max(tokens.length, 1)).toFixed(2);
  const uniqueTokens = new Set(tokens.map(t => t.text)).size;
  return (
    <div className="stage-panel">
      <h2>Stage 2: Tokenizer</h2>
      <p className="stage-description">
        Text is split into tokens. The tokenizer uses Byte-Pair Encoding (BPE) to create
        a vocabulary of ~128k tokens. Common words are single tokens, rare words are split.
      </p>
      <div className="token-chips">
        {tokens.map((token, idx) => (
          <button
            key={idx}
            className={`token-chip ${selectedToken === idx ? 'selected' : ''}`}
            style={{ backgroundColor: getTokenColor(token.text) }}
            onClick={() => setSelectedToken(idx)}
            title={`Token ID: ${token.id}`}
          >
            <div className="token-text">{token.text || '∅'}</div>
            <div className="token-id">#{token.id}</div>
          </button>
        ))}
      </div>
      <div className="stats-row">
        <div className="stat">
          <div className="stat-label">Total Tokens</div>
          <div className="stat-value">{tokens.length}</div>
        </div>
        <div className="stat">
          <div className="stat-label">Unique Tokens</div>
          <div className="stat-value">{uniqueTokens}</div>
        </div>
        <div className="stat">
          <div className="stat-label">Compression Ratio</div>
          <div className="stat-value">{compressionRatio}x</div>
        </div>
      </div>
      <div className="token-detail" style={{ borderLeft: `4px solid ${getTokenColor(tokens[selectedToken]?.text)}` }}>
        <h4 style={{ color: 'var(--text-primary)', marginBottom: '8px' }}>Inspector: <span style={{ opacity: 0.5, fontWeight: 'normal' }}>#{tokens[selectedToken]?.id || 0}</span></h4>
        <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
          <div className="code" style={{ fontSize: '24px', padding: '12px 24px', backgroundColor: 'var(--dark-bg)' }}>
            {tokens[selectedToken]?.text || 'None'}
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Type: <b>{
              getTokenColor(tokens[selectedToken]?.text) === tokenColors.subword ? 'Subword/Suffix' :
                getTokenColor(tokens[selectedToken]?.text) === tokenColors.punctuation ? 'Punctuation' :
                  getTokenColor(tokens[selectedToken]?.text) === tokenColors.special ? 'Special Token' : 'Full Word'
            }</b></span>
            <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Length: <b>{tokens[selectedToken]?.text?.length || 0} chars</b></span>
          </div>
        </div>
      </div>
      <div className="explainer">
        <h3>Why does this matter?</h3>
        <p>
          Tokenization is the bridge between text and numbers. Better tokenization (fewer tokens for the same meaning)
          means more context fit in the LLM's attention window. It's why multilingual tokenizers are often less efficient
          for non-English text.
        </p>
      </div>
    </div>
  );
}