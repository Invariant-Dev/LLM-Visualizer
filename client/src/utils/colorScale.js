export function createColorScale(domain, range) {
  return (value) => {
    const normalized = (value - domain[0]) / (domain[1] - domain[0]);
    const clamped = Math.max(0, Math.min(1, normalized));
    const start = hexToRgb(range[0]);
    const end = hexToRgb(range[1]);
    const r = Math.round(start[0] + (end[0] - start[0]) * clamped);
    const g = Math.round(start[1] + (end[1] - start[1]) * clamped);
    const b = Math.round(start[2] + (end[2] - start[2]) * clamped);
    return rgbToHex(r, g, b);
  };
}
function hexToRgb(hex) {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result ? [parseInt(result[1], 16), parseInt(result[2], 16), parseInt(result[3], 16)] : [0, 0, 0];
}
function rgbToHex(r, g, b) {
  return '#' + [r, g, b].map((x) => {
    const hex = x.toString(16);
    return hex.length === 1 ? '0' + hex : hex;
  }).join('');
}
export const attentionColorScale = createColorScale([0, 1], ['#09090b', '#fb923c']);
export const tokenColors = {
  fullWord: '#e2e8f0',
  subword: '#94a3b8',
  punctuation: '#f1f5f9',
  special: '#fb923c'
};