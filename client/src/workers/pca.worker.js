import { pca } from '../utils/pca';

self.onmessage = function(e) {
  const { vecs, nComponents, payload } = e.data;
  try {
    const projected = pca(vecs, nComponents);
    self.postMessage({ type: 'success', projected, payload });
  } catch (error) {
    self.postMessage({ type: 'error', error: error.message, payload });
  }
};
