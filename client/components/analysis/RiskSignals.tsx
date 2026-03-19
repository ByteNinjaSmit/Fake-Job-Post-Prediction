'use client';

import { useStore } from '@/lib/store/useStore';
import { motion, AnimatePresence } from 'framer-motion';
import { ShieldAlert, AlertTriangle, ShieldCheck, Zap } from 'lucide-react';
import { Badge } from '@/components/ui/badge';

export function RiskSignals() {
  const { signals, isAnalyzing } = useStore();

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'high': return 'text-destructive';
      case 'medium': return 'text-amber-500';
      case 'low': return 'text-cyan-500';
      default: return 'text-primary';
    }
  };

  const highRiskCount = signals.filter(s => s.risk === 'high').length;

  return (
    <div className="h-full flex flex-col space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
           <div className="h-2 w-2 rounded-full bg-primary shadow-[0_0_8px_var(--primary)] animate-pulse" />
           <h3 className="text-sm font-black uppercase tracking-[0.2em] text-foreground/70">Risk Signals</h3>
        </div>
        <Badge variant="outline" className="border-border rounded-full px-3 text-[10px] font-bold">
          {signals.length} Flags Identified
        </Badge>
      </div>

      <div className="flex-1 space-y-4 overflow-y-auto scrollbar-hide pr-2">
        <AnimatePresence mode="popLayout">
          {signals.length === 0 ? (
            <motion.div 
               initial={{ opacity: 0 }}
               animate={{ opacity: 1 }}
               className="h-full flex flex-col items-center justify-center p-8 text-center glass-card border-none rounded-2xl bg-foreground/[0.01]"
            >
               <ShieldCheck className="h-10 w-10 text-green-500/20 mb-4" />
               <p className="text-xs text-muted-foreground font-medium uppercase tracking-widest italic opacity-40">No localized anomalies detected</p>
            </motion.div>
          ) : (
            signals.map((signal, idx) => (
              <motion.div
                key={signal.id}
                initial={{ x: 20, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                exit={{ x: -20, opacity: 0 }}
                transition={{ delay: idx * 0.05 }}
                className="p-4 glass-card border-none rounded-2xl bg-foreground/[0.02] hover:bg-foreground/[0.05] transition-colors relative group"
              >
                <div className={`absolute top-0 left-0 w-1 h-full rounded-full ${getRiskColor(signal.risk).replace('text', 'bg')}`} />
                <div className="flex items-start justify-between mb-2">
                   <span className="text-xs font-black uppercase tracking-widest text-primary/80">
                     {signal.keyword}
                   </span>
                   <span className={`text-[10px] font-black uppercase tracking-tighter ${getRiskColor(signal.risk)}`}>
                     {signal.risk} risk
                   </span>
                </div>
                <p className="text-[11px] text-muted-foreground leading-snug font-medium">
                  {signal.message}
                </p>
              </motion.div>
            ))
          )}
        </AnimatePresence>
      </div>

      <div className={`p-6 rounded-3xl transition-all duration-500 overflow-hidden relative group ${highRiskCount > 0 ? 'bg-destructive/10 border border-destructive/20' : 'bg-primary/10 border border-primary/20'}`}>
         <div className="absolute top-0 right-0 w-32 h-32 bg-foreground/5 rounded-full blur-[40px] -mr-16 -mt-16" />
         <div className="relative z-10 flex items-center justify-between">
            <div className="space-y-1">
               <p className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">Threat Status</p>
               <p className={`text-xl font-black font-outfit uppercase ${highRiskCount > 0 ? 'text-destructive' : 'text-primary'}`}>
                 {highRiskCount > 0 ? 'Compromised' : 'Analyzing'}
               </p>
            </div>
            {highRiskCount > 0 ? <ShieldAlert className="h-8 w-8 text-destructive animate-bounce" /> : <Zap className="h-8 w-8 text-primary animate-pulse" />}
         </div>
      </div>
    </div>
  );
}
