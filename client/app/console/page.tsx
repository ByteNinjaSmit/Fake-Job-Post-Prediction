'use client';

import { AIThinkingFeed } from '@/components/ai/AIThinkingFeed';
import { LiveInputPanel } from '@/components/input/LiveInputPanel';
import { RiskSignals } from '@/components/analysis/RiskSignals';
import { usePrediction } from '@/hooks/usePrediction';
import { useStore } from '@/lib/store/useStore';
import { motion, AnimatePresence } from 'framer-motion';
import { ShieldCheck, ShieldAlert, Cpu, Activity, Zap, Terminal } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';

export default function ConsolePage() {
  const { runAnalysis } = usePrediction();
  const { isAnalyzing, prediction } = useStore();

  return (
    <div className="max-w-[1600px] mx-auto space-y-12 pb-20 px-4">
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-8 border-b border-white/5 pb-10">
        <div className="space-y-4">
          <div className="flex items-center gap-3">
             <div className="p-2 bg-primary/10 rounded-lg">
                <Terminal className="h-5 w-5 text-primary" />
             </div>
             <span className="text-[10px] font-black uppercase tracking-[0.4em] text-zinc-500">System Path: /root/intelligence/console</span>
          </div>
          <h1 className="text-5xl md:text-6xl font-black font-outfit text-white leading-tight">
             Neural <span className="text-gradient italic">Intelligence</span> Console
          </h1>
          <p className="text-zinc-400 text-lg font-medium max-w-2xl leading-relaxed">
            Orchestrate multi-layered linguistic analysis and simulate neural decision paths.
          </p>
        </div>

        <div className="hidden lg:flex items-center gap-10 px-8 py-5 glass-panel rounded-[2rem]">
           <div className="space-y-1">
              <p className="text-[9px] font-black text-zinc-500 uppercase tracking-widest">Vector Load</p>
              <div className="flex items-center gap-4">
                 <div className="h-2 w-48 bg-white/5 rounded-full overflow-hidden">
                    <motion.div 
                      animate={{ width: isAnalyzing ? '92%' : '14%' }}
                      className="h-full premium-gradient shadow-[0_0_10px_rgba(124,58,237,0.5)]" 
                    />
                 </div>
                 <span className="text-xs font-black tabular-nums text-white">{isAnalyzing ? '92%' : '14%'}</span>
              </div>
           </div>
           <div className="h-10 w-px bg-white/10" />
           <div className="space-y-1">
              <p className="text-[9px] font-black text-zinc-500 uppercase tracking-widest">Neural Sync</p>
              <div className="flex items-center gap-2">
                 <div className="h-2 w-2 rounded-full bg-green-500 shadow-[0_0_8px_#22c55e] animate-pulse" />
                 <span className="text-xs font-black uppercase tracking-tighter text-zinc-200">Synchronized</span>
              </div>
           </div>
        </div>
      </div>

      <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-8 min-h-0">
        {/* LEFT: AI Thought Engine (Simulation Feed) */}
        <div className="lg:col-span-3 h-full">
            <AIThinkingFeed />
        </div>

        {/* CENTER: Input & Prediction Core */}
        <div className="lg:col-span-6 flex flex-col space-y-8 h-full">
           <div className="flex-1 px-4">
              <LiveInputPanel onAnalyze={runAnalysis} />
           </div>
           
           <AnimatePresence>
             {prediction && (
               <motion.div
                 initial={{ y: 20, opacity: 0 }}
                 animate={{ y: 0, opacity: 1 }}
                 exit={{ y: 20, opacity: 0 }}
                 className="px-4 pb-4"
               >
                 <Card className={`overflow-hidden border-none shadow-2xl relative ${prediction.prediction === 'Fraudulent' ? 'bg-destructive/10' : 'bg-green-500/10'}`}>
                    <div className="absolute inset-x-0 top-0 h-1 premium-gradient" />
                    <CardHeader className="py-4">
                       <div className="flex items-center justify-between">
                          <div className="flex items-center gap-3">
                             {prediction.prediction === 'Fraudulent' ? <ShieldAlert className="h-6 w-6 text-destructive" /> : <ShieldCheck className="h-6 w-6 text-green-500" />}
                             <CardTitle className="text-lg font-black font-outfit uppercase italic">
                               Final Analysis Output
                             </CardTitle>
                          </div>
                          <Badge className={`${prediction.prediction === 'Fraudulent' ? 'bg-destructive' : 'bg-green-500'} text-white border-none py-1 px-4 rounded-xl font-black italic`}>
                            {prediction.prediction.toUpperCase()}
                          </Badge>
                       </div>
                    </CardHeader>
                    <CardContent>
                       <div className="flex items-center gap-8">
                          <div className="flex-1 space-y-2">
                             <div className="flex justify-between text-[10px] font-black uppercase tracking-widest opacity-60">
                                <span>Neural Confidence</span>
                                <span>{Math.round(prediction.confidence * 100)}%</span>
                             </div>
                             <Progress 
                                value={prediction.confidence * 100} 
                                className={`h-2 rounded-full bg-white/5 ${prediction.prediction === 'Fraudulent' ? '[&>div]:bg-destructive' : '[&>div]:bg-green-500'}`} 
                             />
                          </div>
                          <div className="text-center px-4 py-2 bg-white/5 rounded-2xl border border-white/5">
                             <p className="text-[10px] font-black text-muted-foreground uppercase">Stability</p>
                             <p className="text-lg font-black text-primary">0.99</p>
                          </div>
                       </div>
                    </CardContent>
                 </Card>
               </motion.div>
             )}
           </AnimatePresence>
        </div>

        {/* RIGHT: Risk Signals & Metrics */}
        <div className="lg:col-span-3 flex flex-col space-y-8 h-full">
            <div className="flex-1">
               <RiskSignals />
            </div>
            
            <Card className="glass-card border-none rounded-3xl p-6 bg-white/[0.01]">
               <div className="flex items-center justify-between mb-4">
                  <h4 className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">Vector Load</h4>
                  <Zap className="h-3 w-3 text-primary animate-pulse" />
               </div>
               <div className="space-y-4">
                  <div className="flex items-center justify-between">
                     <span className="text-[11px] font-medium text-foreground/60">Linguistic Shift</span>
                     <span className="text-[11px] font-black tabular-nums">+0.04</span>
                  </div>
                  <div className="flex items-center justify-between">
                     <span className="text-[11px] font-medium text-foreground/60">Context Weights</span>
                     <span className="text-[11px] font-black tabular-nums">0.812</span>
                  </div>
                  <div className="pt-2">
                     <div className="h-1 w-full bg-white/5 rounded-full overflow-hidden">
                        <motion.div 
                          initial={{ width: 0 }}
                          animate={{ width: '65%' }}
                          className="h-full premium-gradient" 
                        />
                     </div>
                  </div>
               </div>
            </Card>
        </div>
      </div>
    </div>
  );
}
