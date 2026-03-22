export function pca(data, nComponents = 2) {
  const mean = data[0].map((_, i) => {
    const sum = data.reduce((acc, row) => acc + row[i], 0);
    return sum / data.length;
  });
  const centered = data.map((row) => row.map((val, i) => val - mean[i]));
  const n = centered.length;
  const m = centered[0].length;
  const cov = Array(m)
    .fill(0)
    .map(() => Array(m).fill(0));
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < m; j++) {
      for (let k = 0; k < n; k++) {
        cov[i][j] += (centered[k][i] * centered[k][j]) / n;
      }
    }
  }
  const eigenVectors = [];
  let matrix = cov.map((row) => [...row]);
  for (let comp = 0; comp < Math.min(nComponents, m); comp++) {
    const v = Array(m)
      .fill(0)
      .map(() => Math.random());
    for (let iter = 0; iter < 50; iter++) {
      let newV = Array(m).fill(0);
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < m; j++) {
          newV[i] += matrix[i][j] * v[j];
        }
      }
      const norm = Math.sqrt(newV.reduce((acc, x) => acc + x * x, 0));
      for (let i = 0; i < m; i++) {
        v[i] = newV[i] / norm;
      }
    }
    eigenVectors.push(v);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < m; j++) {
        matrix[i][j] -= v[i] * v[j] * 10; 
      }
    }
  }
  const projected = centered.map((row) =>
    eigenVectors.map((ev) => row.reduce((acc, val, i) => acc + val * ev[i], 0))
  );
  return projected;
}