import React from 'react';

export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, info) {
    console.error('stage error:', error, info.componentStack);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          padding: 32,
          textAlign: 'center',
          color: 'var(--text-secondary)',
        }}>
          <div style={{
            fontSize: 18,
            fontWeight: 600,
            color: '#ef4444',
            marginBottom: 12,
          }}>
            something went wrong in this stage
          </div>
          <div style={{
            fontSize: 13,
            color: 'var(--text-muted)',
            marginBottom: 16,
            fontFamily: 'monospace',
          }}>
            {this.state.error?.message || 'unknown error'}
          </div>
          <button
            className="btn btn-secondary"
            onClick={() => this.setState({ hasError: false, error: null })}
          >
            try again
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
