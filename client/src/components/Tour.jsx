import React, { useState } from 'react';
import Joyride, { STATUS } from 'react-joyride';

export default function Tour() {
  const [run, setRun] = useState(false);

  const steps = [
    {
      target: '.app-header',
      content: 'Welcome to the LLM Internals Explorer! This tool visualizes how Large Language Models work under the hood.',
      disableBeacon: true,
    },
    {
      target: '.stage-nav',
      content: 'Navigate through the different stages of the transformer architecture here.',
    },
    {
      target: '.model-picker',
      content: 'Select different local Ollama models to see how their configurations differ.',
    },
    {
      target: '.mock-toggle',
      content: 'Toggle mock mode if you do not have Ollama installed locally.',
    }
  ];

  const handleJoyrideCallback = (data) => {
    const { status } = data;
    if ([STATUS.FINISHED, STATUS.SKIPPED].includes(status)) {
      setRun(false);
    }
  };

  return (
    <>
      <button 
        className="btn btn-secondary" 
        onClick={() => setRun(true)}
        title="Start Interactive Tour"
        style={{ padding: '6px 12px', fontSize: 13 }}
      >
        Take Tour
      </button>
      <Joyride
        callback={handleJoyrideCallback}
        continuous
        hideCloseButton
        run={run}
        scrollToFirstStep
        showProgress
        showSkipButton
        steps={steps}
        styles={{
          options: {
            zIndex: 10000,
            primaryColor: '#10b981',
            backgroundColor: '#27272a',
            textColor: '#f4f4f5',
            arrowColor: '#27272a',
          }
        }}
      />
    </>
  );
}
