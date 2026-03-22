import React, { createContext, useContext, useState, useEffect } from 'react';
import { fetchTags, fetchModelInfo } from '../services/api';
import axios from 'axios';

const AppContext = createContext(null);

export function useAppContext() {
  return useContext(AppContext);
}

export function AppProvider({ children }) {
  const [isOllamaMocked, setIsOllamaMocked] = useState(false);
  const [backendMode, setBackendMode] = useState('ollama');
  const [pytorchAvailable, setPytorchAvailable] = useState(false);
  const [pytorchModel, setPytorchModel] = useState('');
  const [input, setInput] = useState('Why is the sky blue?');
  const [tokens, setTokens] = useState([]);
  const [modelInfo, setModelInfo] = useState(null);
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [loadingModel, setLoadingModel] = useState(true);
  useEffect(() => {
    const init = async () => {
      try {
        const models = await fetchTags();
        if (models.length === 0) {
          setIsOllamaMocked(false);
          setAvailableModels([]);
          setSelectedModel('');
        } else {
          const firstModel = models[0].name;
          setAvailableModels(models);
          setSelectedModel(firstModel);
          setIsOllamaMocked(false);
          try {
            const info = await fetchModelInfo(firstModel);
            setModelInfo(info);
          } catch { }
        }
      } catch {
        setIsOllamaMocked(true);
      } finally {
        setLoadingModel(false);
      }
    };
    init();

    const checkPytorch = () => {
      axios.get('/api/pytorch/health', { timeout: 3000 })
        .then(res => {
          if (res.data?.status === 'ok') {
            setPytorchAvailable(true);
            setPytorchModel(res.data.model || 'gpt2');
            setBackendMode(prev => prev === 'pytorch' ? 'pytorch' : 'pytorch'); // Auto switch when available
          }
        })
        .catch(() => setPytorchAvailable(false));
    };

    checkPytorch();
    const intervalId = setInterval(checkPytorch, 3000);
    return () => clearInterval(intervalId);
  }, []);
  useEffect(() => {
    if (!selectedModel || isOllamaMocked) return;
    fetchModelInfo(selectedModel)
      .then(info => setModelInfo(info))
      .catch(() => setModelInfo(null));
  }, [selectedModel, isOllamaMocked]);
  useEffect(() => {
    if (backendMode === 'mock') {
      setIsOllamaMocked(true);
    } else if (backendMode === 'ollama') {
      setIsOllamaMocked(availableModels.length === 0);
    } else if (backendMode === 'pytorch') {
      setIsOllamaMocked(false);
    }
  }, [backendMode, availableModels.length]);
  const value = {
    isOllamaMocked, setIsOllamaMocked,
    backendMode, setBackendMode,
    pytorchAvailable, pytorchModel,
    input, setInput,
    tokens, setTokens,
    modelInfo,
    availableModels,
    selectedModel, setSelectedModel,
    loadingModel,
  };
  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}
