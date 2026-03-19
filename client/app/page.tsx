'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { ShieldAlert, Zap, Search, Activity, Cpu, Sparkles, ChevronRight, LayoutGrid, Terminal } from 'lucide-react';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

export default function Home() {
  return (
    <div className="relative min-h-[calc(100vh-8rem)] flex flex-col items-center justify-center overflow-hidden py-20 px-4">
      {/* Background Cinematic Effects - Adaptive */}
      <div className="absolute inset-0 z-0 pointer-events-none">
         <div className="absolute top-1/4 left-1/4 w-[500px] h-[500px] bg-primary/5 dark:bg-primary/20 rounded-full blur-[180px] animate-pulse opacity-40" />
         <div className="absolute bottom-1/4 right-1/4 w-[500px] h-[500px] bg-cyan-500/5 dark:bg-cyan-500/15 rounded-full blur-[180px] animate-pulse delay-1000 opacity-30" />
      </div>

      <div className="relative z-10 text-center max-w-5xl space-y-16">
        <motion.div
          initial={{ y: 30, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 1, ease: 'easeOut' }}
          className="space-y-8"
        >
          <div className="flex items-center justify-center mb-8">
             <div className="px-5 py-2 rounded-full bg-foreground/[0.03] border border-border backdrop-blur-xl flex items-center gap-3 shadow-xl">
                <div className="h-2 w-2 rounded-full bg-primary shadow-[0_0_10px_rgba(124,58,237,0.5)] animate-pulse" />
                <span className="text-[11px] font-black uppercase tracking-[0.4em] text-muted-foreground">Neural Gateway Core v2.1</span>
             </div>
          </div>
          
            <div className="space-y-4">
              <h1 className="text-7xl md:text-9xl font-black tracking-tighter font-outfit text-foreground leading-[0.85] drop-shadow-sm">
                AI FRAUD
              </h1>
              <h1 className="text-6xl md:text-8xl font-black tracking-tighter font-outfit text-gradient leading-[0.85] italic pb-4">
                INTELLIGENCE
              </h1>
            </div>
          
          <p className="text-xl md:text-2xl text-muted-foreground font-medium max-w-2xl mx-auto leading-relaxed mt-10">
            Next-generation linguistic fingerprinting for <br className="hidden md:block" />
            real-time recruitment trust verification.
          </p>
        </motion.div>

        <motion.div 
          initial={{ y: 30, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.6, duration: 1 }}
          className="flex flex-col sm:flex-row items-center justify-center gap-8 pt-8"
        >
          <Button size="lg" className="h-[80px] px-12 rounded-3xl premium-gradient text-xl font-black uppercase tracking-[0.2em] shadow-xl dark:shadow-[0_20px_60px_rgba(124,58,237,0.3)] hover:scale-105 transition-all duration-300 group border-t border-foreground/20 active:scale-95 text-white" asChild>
             <Link href="/console">
               Initialize Console <Terminal className="ml-4 h-6 w-6 group-hover:rotate-12 transition-transform" />
             </Link>
          </Button>
          
          <Button variant="outline" size="lg" className="h-[80px] px-12 rounded-3xl glass-panel text-xl font-black uppercase tracking-[0.2em] hover:bg-foreground/[0.05] text-foreground shadow-xl hover:scale-105 transition-all duration-300 active:scale-95 group border-border" asChild>
             <Link href="/scan">
               Live Scan <Scan className="ml-4 h-6 w-6 group-hover:translate-x-1 transition-transform" />
             </Link>
          </Button>
        </motion.div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-12 pt-24 border-t border-border">
           {[
             { label: 'Latency', value: '12ms', color: 'text-primary' },
             { label: 'Accuracy', value: '98.2%', color: 'text-foreground/80' },
             { label: 'Analyses', value: '1.2M+', color: 'text-cyan-600 dark:text-cyan-400' },
             { label: 'Stability', value: '100%', color: 'text-foreground/80' },
           ].map((stat, i) => (
             <motion.div 
               key={i}
               initial={{ opacity: 0, y: 20 }}
               animate={{ opacity: 1, y: 0 }}
               transition={{ delay: 1 + i * 0.1, duration: 0.8 }}
               className="text-center group"
             >
                <div className="text-[11px] font-black uppercase tracking-[0.3em] text-muted-foreground mb-2 group-hover:text-primary transition-colors">{stat.label}</div>
                <div className={cn("text-3xl font-black font-outfit tabular-nums drop-shadow-sm", stat.color)}>{stat.value}</div>
             </motion.div>
           ))}
        </div>
      </div>
    </div>
  );
}


function Scan({ className }: { className?: string }) {
    return <svg className={className} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/><line x1="7" x2="17" y1="12" y2="12"/></svg>;
}