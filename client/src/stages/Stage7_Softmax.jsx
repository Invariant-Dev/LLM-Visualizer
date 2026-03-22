import React, { useState, useMemo } from 'react';
import { Chart as ChartJS, CategoryScale, LinearScale, BarController, BarElement, Tooltip, Legend } from 'chart.js';
import { Bar } from 'react-chartjs-2';
import * as mockData from "../mock/tokens";
ChartJS.register(CategoryScale, LinearScale, BarController, BarElement, Tooltip, Legend);
export default function Stage7_Softmax({ goToNextStage }) {
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(40);
  const [topP, setTopP] = useState(0.9);
  const logprobs = useMemo(() => mockData.getMockLogprobs(), []);
  const temperatureAdjusted = logprobs.map(item => ({
    ...item,
    logprob: item.logprob / temperature,
    probability: Math.exp(item.logprob / temperature)
  }));
  const sum = temperatureAdjusted.reduce((acc, item) => acc + item.probability, 0);
  const normalized = temperatureAdjusted.map(item => ({
    ...item,
    probability: item.probability / sum
  }));
  const filtered = normalized.slice(0, topK);
  let cumulativeProb = 0;
  const nucleusIdx = filtered.findIndex(item => {
    cumulativeProb += item.probability;
    return cumulativeProb > topP;
  });
  const chartData = {
    labels: filtered.map((item, idx) => (idx < nucleusIdx + 1 ? item.token : 'other')),
    datasets: [
      {
        label: 'Probability',
        data: filtered.map(item => item.probability * 100),
        backgroundColor: filtered.map((_, idx) => (idx < nucleusIdx + 1 ? '#3b82f6' : '#475569')),
        borderColor: '#1e293b',
        borderWidth: 1
      }
    ]
  };
  return (
    <div className="stage-panel">
      <h2>Stage 7: Softmax & Temperature</h2>
      <p className="stage-description">
        The model outputs raw logits for the next ~128k possible tokens. Softmax converts them to
        probabilities. Temperature controls how "confident" the distribution is.
      </p>
      <div className="sampling-controls">
        <div className="control-group">
          <label>Temperature: {temperature.toFixed(2)}</label>
          <input
            type="range"
            min="0.1"
            max="2.0"
            step="0.05"
            value={temperature}
            onChange={(e) => setTemperature(parseFloat(e.target.value))}
            className="slider"
          />
          <div className="control-hint">
            {temperature < 0.5 ? 'Very focused' : temperature < 1 ? 'Focused' : temperature < 1.5 ? 'Balanced' : 'Creative'}
          </div>
        </div>
        <div className="control-group">
          <label>Top-K: {topK}</label>
          <input
            type="range"
            min="1"
            max="100"
            step="1"
            value={topK}
            onChange={(e) => setTopK(parseInt(e.target.value))}
            className="slider"
          />
          <div className="control-hint">Only sample from top {topK} tokens</div>
        </div>
        <div className="control-group">
          <label>Top-P (Nucleus): {topP.toFixed(2)}</label>
          <input
            type="range"
            min="0.0"
            max="1.0"
            step="0.05"
            value={topP}
            onChange={(e) => setTopP(parseFloat(e.target.value))}
            className="slider"
          />
          <div className="control-hint">Sample from smallest set cumulating to {(topP * 100).toFixed(0)}%</div>
        </div>
      </div>
      <div className="chart-container">
        <Bar
          data={chartData}
          options={{
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              x: {
                ticks: { color: '#cbd5e1' },
                grid: { color: '#334155' }
              },
              y: {
                ticks: { color: '#cbd5e1' },
                grid: { color: '#334155' }
              }
            },
            plugins: {
              legend: { labels: { color: '#cbd5e1' } }
            }
          }}
        />
      </div>
      <div className="nucleus-info">
        <p>
          <strong>Nucleus Cutoff:</strong> {nucleusIdx + 1} tokens ({((nucleusIdx + 1) / topK * 100).toFixed(1)}% of top-K)
        </p>
        <p className="code">
          Sample from {nucleusIdx + 1} tokens with cumulative P ≥ {topP.toFixed(2)}
        </p>
      </div>
      <div className="explainer">
        <h3>Why does this matter?</h3>
        <p>
          Temperature balances diversity vs. determinism. Low temperature (0.1) makes the model repetitive but confident.
          High temperature (2.0) makes it creative but potentially incoherent. Top-K and Top-P are practical ways to avoid
          very low-probability "hallucinations."
        </p>
      </div>
    </div>
  );
}