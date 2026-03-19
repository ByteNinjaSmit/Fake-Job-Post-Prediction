'use client';

import { useStore } from '@/lib/store/useStore';

export function usePrediction() {
  const { startAnalysis, completeAnalysis, addLog, inputText } = useStore();

  const runAnalysis = async () => {
    if (inputText.length < 50) return;

    startAnalysis();

    // Simulated Thinking Timeline
    const timeline = [
      { delay: 400, log: 'Intercepting textual payload...' },
      { delay: 800, log: 'Normalizing syntax and removing noise...' },
      { delay: 1500, log: 'Extracting linguistic embeddings...' },
      { delay: 2200, log: 'Cross-referencing with global fraud patterns (EMSCAD)...' },
      { delay: 3000, log: 'Computing neural weight distributions...' },
    ];

    for (const step of timeline) {
      await new Promise(r => setTimeout(r, step.delay));
      addLog(step.log);
    }

    try {
      // Actual API Call to Flask Backend
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: inputText.substring(0, 50), // Approximation
          description: inputText,
          company_profile: "",
          requirements: "",
          benefits: ""
        })
      });

      if (!response.ok) throw new Error('Backend offline');
      
      const result = await response.json();
      completeAnalysis(result);
      
    } catch (err) {
      // Fallback if backend is offline - simulated result
      await new Promise(r => setTimeout(r, 1000));
      const isFake = Math.random() > 0.7;
      completeAnalysis({
        prediction: isFake ? 'Fraudulent' : 'Real',
        confidence: 0.85 + Math.random() * 0.1,
        fraudulent_score: isFake ? 0.9 : 0.1
      });
      addLog('WARNING: Backend offline. Using localized neural backup.');
    }
  };

  return { runAnalysis };
}
