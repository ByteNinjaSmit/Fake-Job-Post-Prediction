'use client';

import { useEffect, useRef } from 'react';
import { useStore } from '@/lib/store/useStore';
import { motion, AnimatePresence } from 'framer-motion';
import { Terminal, Cpu, Zap, Activity } from 'lucide-react';

export function AIThinkingFeed() {
  const { thinkingLogs, isAnalyzing } = useStore();
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [thinkingLogs]);

  return (
    <div className="flex flex-col h-full glass-card border-none rounded-3xl overflow-hidden shadow-2xl">
      <div className="flex items-center justify-between px-6 py-4 bg-white/5 border-b border-white/5">
        <div className="flex items-center gap-2">
          <Terminal className="h-4 w-4 text-primary" />
          <span className="text-xs font-black uppercase tracking-widest text-foreground/80">Neural Process Log</span>
        </div>
        {isAnalyzing && (
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-bold text-primary animate-pulse uppercase">Syncing...</span>
            <Activity className="h-3 w-3 text-primary animate-pulse" />
          </div>
        )}
      </div>

      <div 
        ref={scrollRef}
        className="flex-1 p-6 overflow-y-auto font-mono text-[11px] leading-relaxed space-y-3 scrollbar-hide"
      >
        <AnimatePresence initial={false}>
          {thinkingLogs.length === 0 ? (
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-muted-foreground italic flex flex-col items-center justify-center h-full opacity-30"
            >
              <Cpu className="h-12 w-12 mb-4" />
              <p>Standing by for neural input...</p>
            </motion.div>
          ) : (
            thinkingLogs.map((log, idx) => (
              <motion.div
                key={idx}
                initial={{ x: -10, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                className="flex gap-3 group"
              >
                <span className="text-primary/50 tabular-nums">[{new Date().toLocaleTimeString([], { hour12: false, minute: '2-digit', second: '2-digit' })}]</span>
                <span className={idx === thinkingLogs.length - 1 && isAnalyzing ? 'text-primary' : 'text-foreground/90'}>
                  {log}
                  {idx === thinkingLogs.length - 1 && isAnalyzing && (
                    <motion.span
                      animate={{ opacity: [0, 1, 0] }}
                      transition={{ repeat: Infinity, duration: 0.8 }}
                      className="inline-block w-1.5 h-3 ml-1 bg-primary align-middle"
                    />
                  )}
                </span>
              </motion.div>
            ))
          )}
        </AnimatePresence>
      </div>

      <div className="p-4 bg-primary/5 border-t border-white/5 flex items-center justify-between">
        <div className="flex gap-1.5">
          {[1, 2, 3].map(i => (
            <div key={i} className={`h-1 w-8 rounded-full ${isAnalyzing ? 'bg-primary animate-pulse' : 'bg-white/10'}`} />
          ))}
        </div>
        <Zap className={`h-3 w-3 ${isAnalyzing ? 'text-primary fill-primary animate-bounce' : 'text-muted-foreground'}`} />
      </div>
    </div>
  );
}
