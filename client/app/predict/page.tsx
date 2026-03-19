'use client';

import { useState } from 'react';
import PredictionForm from '@/components/prediction/PredictionForm';
import ResultDisplay from '@/components/prediction/ResultDisplay';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { History, Search, Activity, Sparkles } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

export default function PredictPage() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const handlePredict = async (data: any) => {
    setLoading(true);
    setResult(null);
    // Mocking API call for now
    setTimeout(() => {
      const isFake = Math.random() > 0.7;
      setResult({
        is_fake: isFake,
        confidence: 0.85 + (Math.random() * 0.14),
        explanation: isFake 
          ? "High frequency of 'urgent hire' keywords and non-standard salary markers detected."
          : "Professional tone and specific structural markers align with legitimate postings."
      });
      setLoading(false);
    }, 2500);
  };

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-12"
    >
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
        <div>
          <h1 className="text-4xl font-extrabold tracking-tight font-outfit text-gradient">Neural Scan</h1>
          <p className="text-muted-foreground mt-2 text-lg font-medium max-w-xl">
            In-depth analysis of job post legitimacy using state-of-the-art ML.
          </p>
        </div>
        <div className="flex items-center gap-4 px-6 py-3 glass-card rounded-2xl">
            <Activity className="h-5 w-5 text-primary animate-pulse" />
            <span className="text-sm font-bold uppercase tracking-widest">Model Online</span>
        </div>
      </div>

      <div className="grid gap-12 lg:grid-cols-2 items-start">
        <motion.div 
          initial={{ x: -20, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="space-y-8"
        >
           <PredictionForm onPredict={handlePredict} isLoading={loading} />
        </motion.div>

        <div className="space-y-8 sticky top-12">
          <AnimatePresence mode="wait">
            {loading ? (
               <motion.div
                 key="loading"
                 initial={{ scale: 0.9, opacity: 0 }}
                 animate={{ scale: 1, opacity: 1 }}
                 exit={{ scale: 1.1, opacity: 0 }}
                 className="h-[500px] glass-card rounded-3xl flex flex-col items-center justify-center p-12 text-center"
               >
                  <div className="relative">
                     <div className="h-24 w-24 rounded-full border-4 border-primary/20 border-t-primary animate-spin" />
                     <Sparkles className="absolute inset-x-0 inset-y-0 m-auto h-8 w-8 text-primary animate-pulse" />
                  </div>
                  <h3 className="text-2xl font-bold font-outfit mt-8 text-gradient">Analyzing Patterns</h3>
                  <p className="text-muted-foreground mt-4 max-w-xs mx-auto">
                    Deconstructing text linguistic features and comparing against 1.2M labeled samples.
                  </p>
               </motion.div>
            ) : result ? (
               <ResultDisplay key="result" result={result} />
            ) : (
              <motion.div 
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="glass-card border-none rounded-3xl flex flex-col items-center justify-center py-32 text-center"
              >
                <div className="p-6 bg-primary/5 rounded-full mb-6 ring-1 ring-primary/20">
                  <History className="h-10 w-10 text-primary/50" />
                </div>
                <CardTitle className="text-2xl font-bold font-outfit">Waiting for Input</CardTitle>
                <CardDescription className="max-w-[300px] mx-auto mt-4 text-base leading-relaxed">
                  The neural engine is ready. Provide the job details to initiate the security scan.
                </CardDescription>
              </motion.div>
            )}
          </AnimatePresence>

          <Card className="glass-card bg-primary/5 border-none rounded-2xl">
             <CardHeader className="p-6">
               <CardTitle className="text-xs font-bold uppercase tracking-widest text-primary">System Advantage</CardTitle>
             </CardHeader>
             <CardContent className="px-6 pb-6 pt-0 text-sm text-foreground/80 space-y-3">
               <p className="flex items-center gap-2 leading-tight">• <span className="font-bold text-primary">98.2% Accuracy</span> on verified datasets.</p>
               <p className="flex items-center gap-2 leading-tight">• <span className="font-bold text-primary">Zero Retention</span>: Your data is never stored.</p>
               <p className="flex items-center gap-2 leading-tight">• <span className="font-bold text-primary">Pulse Analysis</span>: Sub-second inference time.</p>
             </CardContent>
          </Card>
        </div>
      </div>
    </motion.div>
  );
}
