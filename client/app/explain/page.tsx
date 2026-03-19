'use client';

import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Zap, Info, ShieldAlert, CheckCircle2, Search, Sparkles, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { motion, AnimatePresence } from "framer-motion";
import PredictionForm from '@/components/prediction/PredictionForm';
import api from '@/lib/api';

export default function ExplainPage() {
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleExplain = async (data: any) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await api.post('/explain', data);
      setResult(response.data);
    } catch (err: any) {
      console.error('Explain failed:', err);
      setError(err.response?.data?.detail || 'Failed to generate explanation. Ensure the backend is running.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <motion.div 
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="space-y-12 pb-12"
    >
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
        <div>
          <h1 className="text-4xl font-extrabold tracking-tight font-outfit text-gradient">Explainability Core</h1>
          <p className="text-muted-foreground mt-2 text-lg font-medium max-w-xl">
             Deconstructing neural decisions with local feature mapping (LIME).
          </p>
        </div>
        <div className="flex items-center gap-4 px-6 py-3 glass-card rounded-2xl">
            <Zap className="h-5 w-5 text-indigo-500 fill-indigo-500/20" />
            <span className="text-sm font-bold uppercase tracking-widest text-foreground/80">Analysis: LIME Engine</span>
        </div>
      </div>

      <div className="grid gap-12 lg:grid-cols-2 items-start">
        <div className="space-y-8">
           <PredictionForm onPredict={handleExplain} isLoading={isLoading} />
           
           {error && (
             <motion.div 
               initial={{ opacity: 0, y: 10 }}
               animate={{ opacity: 1, y: 0 }}
               className="p-4 rounded-2xl bg-destructive/10 border border-destructive/20 text-destructive text-sm font-medium"
             >
               {error}
             </motion.div>
           )}
        </div>

        <div className="space-y-8 sticky top-12">
          <AnimatePresence mode="wait">
            {!result && !isLoading ? (
              <motion.div 
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="glass-card border-none rounded-3xl flex flex-col items-center justify-center py-32 text-center"
              >
                <div className="p-6 bg-indigo-500/5 rounded-full mb-6 ring-1 ring-indigo-500/20">
                  <Search className="h-10 w-10 text-indigo-500/50" />
                </div>
                <CardTitle className="text-2xl font-bold font-outfit">Waiting for Input</CardTitle>
                <CardDescription className="max-w-[300px] mx-auto mt-4 text-base leading-relaxed">
                  Submit a job posting to see which linguistic patterns are influencing the model's decision.
                </CardDescription>
              </motion.div>
            ) : isLoading ? (
                <motion.div
                  key="loading"
                  initial={{ scale: 0.9, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  exit={{ scale: 1.1, opacity: 0 }}
                  className="h-[500px] glass-card rounded-3xl flex flex-col items-center justify-center p-12 text-center"
                >
                  <div className="relative">
                     <div className="h-24 w-24 rounded-full border-4 border-indigo-500/20 border-t-indigo-500 animate-spin" />
                     <Zap className="absolute inset-x-0 inset-y-0 m-auto h-8 w-8 text-indigo-500 animate-pulse" />
                  </div>
                  <h3 className="text-2xl font-bold font-outfit mt-8 text-gradient">Mapping Features</h3>
                  <p className="text-muted-foreground mt-4 max-w-xs mx-auto">
                    Running LIME perturbations to identify local feature importance...
                  </p>
                </motion.div>
            ) : (
              <motion.div
                key="result"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="space-y-8"
              >
                <Card className="glass-card border-none shadow-2xl overflow-hidden">
                  <CardHeader className="p-8 border-b border-border bg-foreground/[0.02]">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-xl font-bold font-outfit">Feature Importance</CardTitle>
                      <Badge className={result.prediction === 'Fraudulent' ? 'bg-destructive text-white' : 'bg-green-500 text-white'}>
                        {result.prediction}
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent className="p-8">
                    <div className="space-y-6">
                      {result.top_features.map((feat: any, i: number) => {
                        const isRisk = feat.weight > 0;
                        const percentage = Math.abs(feat.weight) * 100;
                        
                        return (
                          <div key={i} className="space-y-2">
                            <div className="flex justify-between text-xs font-bold uppercase tracking-widest">
                              <span className="text-foreground/70">{feat.word}</span>
                              <span className={isRisk ? 'text-destructive' : 'text-green-500'}>
                                {isRisk ? '+' : '-'}{percentage.toFixed(1)}% Impact
                              </span>
                            </div>
                            <div className="h-1.5 w-full bg-foreground/5 rounded-full overflow-hidden">
                              <motion.div 
                                initial={{ width: 0 }}
                                animate={{ width: `${Math.min(percentage * 2, 100)}%` }}
                                className={`h-full ${isRisk ? 'bg-destructive' : 'bg-green-500'}`}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>

                    <div className="mt-8 p-6 rounded-2xl bg-indigo-500/5 border border-indigo-500/20 flex items-start gap-4">
                      <Info className="h-5 w-5 text-indigo-500 mt-1 shrink-0" />
                      <p className="text-xs text-foreground/70 leading-relaxed font-medium">
                        The weights show how much each word contributed to the prediction. 
                        <span className="text-destructive font-bold mx-1">Red</span> indicates fraud indicators, 
                         terwijl <span className="text-green-500 font-bold mx-1">Green</span> indicates real indicators.
                      </p>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </motion.div>
  );
}
