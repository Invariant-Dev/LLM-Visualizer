const DEFAULT_PROMPT = 'Why is the sky blue?';
export function getMockTokens() {
  return {
    tokens: ['Why', ' is', ' the', ' sky', ' blue', '?'],
    token_ids: [5658, 318, 262, 6766, 4171, 30]
  };
}
export function getMockModelInfo() {
  return {
    format: 'gguf',
    model: 'llama2',
    parameters: '7B',
    quantization_level: 'Q4_K_M',
    general: {
      architecture: 'llama',
      architecture: 'llama',
      quantization_version: 2,
      file_type: 15,
      file_type_label: 'GGUF (Quantized)'
    },
    llama: {
      attention: {
        attention_layernorm_rms_epsilon: 9.99999993922529e-06,
        head_count: 32,
        head_count_kv: 32,
        rope_dimension_count: 128
      },
      block_count: 32,
      context_length: 4096,
      embedding_length: 4096,
      feed_forward_length: 11008,
      rope: {
        rope_freq_base: 10000
      }
    }
  };
}
export function getMockEmbeddings() {
  const embeddingDim = 4096;
  const tokenCount = 6;
  const embeddings = [];
  for (let t = 0; t < tokenCount; t++) {
    const vec = Array(embeddingDim)
      .fill(0)
      .map(() => (Math.random() - 0.5) * 2);
    embeddings.push(vec);
  }
  return {
    embeddings: embeddings,
    tokens: ['Why', ' is', ' the', ' sky', ' blue', '?']
  };
}
export function getMockAttentionHeads(tokenCountParam = 6) {
  const tokenCount = Math.max(1, tokenCountParam);
  const heads = [];
  for (let h = 0; h < 32; h++) {
    const matrix = Array(tokenCount)
      .fill(0)
      .map(() =>
        Array(tokenCount)
          .fill(0)
          .map(() => Math.random())
      );
    for (let i = 0; i < tokenCount; i++) {
      const sum = matrix[i].reduce((a, b) => a + b, 0);
      matrix[i] = matrix[i].map((v) => v / sum);
    }
    heads.push(matrix);
  }
  return { heads, tokenCount };
}
export function getMockLogprobs() {
  return [
    { token: 'the', logprob: 5.32, probability: 0.204 },
    { token: ',', logprob: 4.95, probability: 0.141 },
    { token: 'a', logprob: 4.82, probability: 0.123 },
    { token: 'is', logprob: 4.65, probability: 0.105 },
    { token: 'and', logprob: 4.52, probability: 0.091 },
    { token: ' ', logprob: 4.38, probability: 0.079 },
    { token: 'of', logprob: 4.12, probability: 0.061 },
    { token: 'to', logprob: 4.01, probability: 0.055 },
    { token: 'in', logprob: 3.89, probability: 0.049 },
    { token: 'that', logprob: 3.65, probability: 0.039 },
    { token: 'at', logprob: 3.42, probability: 0.030 },
    { token: 'for', logprob: 3.28, probability: 0.027 },
    { token: 'on', logprob: 3.15, probability: 0.023 },
    { token: 'we', logprob: 2.98, probability: 0.019 },
    { token: 'are', logprob: 2.85, probability: 0.017 },
    { token: 'be', logprob: 2.71, probability: 0.015 },
    { token: 'from', logprob: 2.54, probability: 0.012 },
    { token: 'by', logprob: 2.38, probability: 0.011 },
    { token: 'about', logprob: 2.21, probability: 0.009 },
    { token: 'an', logprob: 2.05, probability: 0.008 }
  ];
}