'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { FlaskConical, Settings2, BarChart3, Binary, Zap } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';

export default function LabPage() {
  const [threshold, setThreshold] = useState(0.5);

  const getMetrics = (t: number) => {
    const TPR = Math.max(0.7, 0.95 - Math.abs(t - 0.4) * 0.5);
    const FPR = Math.max(0.01, 0.3 - (1 - t) * 0.4);
    return { TPR, FPR };
  };

  const metrics = getMetrics(threshold);

  return (
    <div className="max-w-[1400px] mx-auto space-y-16 pb-20 px-4">
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-8 border-b border-white/5 pb-10">
        <div className="space-y-4">
          <div className="flex items-center gap-3">
             <div className="p-2 bg-primary/10 rounded-lg">
                <FlaskConical className="h-5 w-5 text-primary" />
             </div>
             <span className="text-[10px] font-black uppercase tracking-[0.4em] text-zinc-500">System Path: /root/intelligence/lab</span>
          </div>
          <h1 className="text-5xl md:text-6xl font-black font-outfit text-white leading-tight">
             Model <span className="text-gradient italic">Laboratory</span>
          </h1>
          <p className="text-zinc-400 text-lg font-medium max-w-2xl leading-relaxed">
             Expose internal neural parameters and visualize classification boundary shifts with real-time feedback loops.
          </p>
        </div>
        <div className="flex items-center gap-4 px-6 py-3 glass-panel rounded-2xl border-white/10">
            <div className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-[10px] font-black uppercase tracking-widest text-zinc-300">Kernel-01 Active</span>
        </div>
      </div>

      <div className="grid gap-8 lg:grid-cols-12">
         {/* Threshold Control */}
         <Card className="lg:col-span-4 glass-card border-none rounded-3xl p-8 bg-white/[0.01]">
            <CardHeader className="p-0 mb-8">
               <div className="flex items-center gap-3 mb-2">
                  <div className="p-2 bg-primary/10 rounded-lg text-primary">
                     <Settings2 className="h-5 w-5" />
                  </div>
                  <CardTitle className="text-xl font-bold font-outfit">Decision Boundary</CardTitle>
               </div>
               <CardDescription>Adjust the neural sensitivity threshold.</CardDescription>
            </CardHeader>
            <CardContent className="p-0 space-y-12">
               <div className="space-y-6">
                  <div className="flex justify-between items-end">
                     <span className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">Threshold</span>
                     <span className="text-3xl font-black font-outfit text-primary">{threshold.toFixed(2)}</span>
                  </div>
                  <Slider 
                    value={[threshold]} 
                    onValueChange={([val]) => setThreshold(val)} 
                    max={1} 
                    step={0.01} 
                    className="[&>span:first-child]:h-2 [&>span:first-child]:bg-white/5 [&_[role=slider]]:h-6 [&_[role=slider]]:w-6 [&_[role=slider]]:bg-primary [&_[role=slider]]:border-none"
                  />
               </div>

               <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 rounded-2xl bg-white/5 border border-white/5">
                     <p className="text-[10px] font-black text-muted-foreground uppercase mb-1">True Positive</p>
                     <p className="text-xl font-black font-outfit text-green-500">{(metrics.TPR * 100).toFixed(1)}%</p>
                  </div>
                  <div className="p-4 rounded-2xl bg-white/5 border border-white/5">
                     <p className="text-[10px] font-black text-muted-foreground uppercase mb-1">False Positive</p>
                     <p className="text-xl font-black font-outfit text-destructive">{(metrics.FPR * 100).toFixed(1)}%</p>
                  </div>
               </div>

               <p className="text-[11px] text-muted-foreground leading-relaxed italic border-l-2 border-primary/20 pl-4">
                 Lowering the threshold increases recall (detections) but may increase noise in legitimate streams.
               </p>
            </CardContent>
         </Card>

         {/* Visualizations */}
         <div className="lg:col-span-8 grid gap-8 h-fit">
            <Card className="glass-card border-none rounded-3xl p-8 bg-white/[0.01]">
               <CardHeader className="p-0 mb-8 flex flex-row items-center justify-between">
                  <div>
                    <CardTitle className="text-xl font-bold font-outfit">Classification Matrix</CardTitle>
                    <CardDescription>Simulated error distribution at current threshold.</CardDescription>
                  </div>
                  <Badge className="bg-primary/10 text-primary border-none font-bold px-4 py-1.5 rounded-full uppercase tracking-tighter text-[10px]">
                     Live Computed
                  </Badge>
               </CardHeader>
               <CardContent className="p-0">
                  <div className="grid grid-cols-2 gap-px bg-white/5 rounded-2xl overflow-hidden border border-white/5">
                     <div className="bg-[#0a0a0a] p-12 flex flex-col items-center justify-center text-center">
                        <span className="text-[10px] font-black text-muted-foreground uppercase tracking-widest mb-4">True Neg</span>
                        <div className="text-4xl font-black font-outfit text-white/40">12,402</div>
                     </div>
                     <div className="bg-[#0a0a0a] p-12 flex flex-col items-center justify-center text-center">
                        <span className="text-[10px] font-black text-destructive uppercase tracking-widest mb-4 font-bold">False pos</span>
                        <div className="text-4xl font-black font-outfit text-destructive/40">342</div>
                     </div>
                     <div className="bg-[#0a0a0a] p-12 flex flex-col items-center justify-center text-center">
                        <span className="text-[10px] font-black text-amber-500 uppercase tracking-widest mb-4 font-bold">False neg</span>
                        <div className="text-4xl font-black font-outfit text-amber-500/40">128</div>
                     </div>
                     <div className="bg-[#0a0a0a] p-12 flex flex-col items-center justify-center text-center">
                        <span className="text-[10px] font-black text-green-500 uppercase tracking-widest mb-4 font-bold">True Pos</span>
                        <div className="text-4xl font-black font-outfit text-green-500/80">3,204</div>
                     </div>
                  </div>
               </CardContent>
            </Card>

            <div className="flex gap-8">
               <Card className="flex-1 glass-card border-none rounded-3xl p-6 relative overflow-hidden group">
                  <div className="absolute inset-0 premium-gradient opacity-0 group-hover:opacity-10 transition-opacity" />
                  <div className="flex items-center gap-3 mb-4">
                     <Zap className="h-4 w-4 text-primary" />
                     <h4 className="text-xs font-black uppercase tracking-widest">Weight Entropy</h4>
                  </div>
                  <div className="h-8 w-full flex items-end gap-1">
                     {[30, 50, 40, 80, 60, 45, 90, 70].map((h, i) => (
                       <motion.div key={i} animate={{ height: `${h}%` }} className="flex-1 bg-primary/20 rounded-t-sm" />
                     ))}
                  </div>
               </Card>
               <Card className="flex-1 glass-card border-none rounded-3xl p-6 relative overflow-hidden group">
                  <div className="absolute inset-0 bg-cyan-500/20 opacity-0 group-hover:opacity-10 transition-opacity" />
                  <div className="flex items-center gap-3 mb-4">
                     <Binary className="h-4 w-4 text-cyan-500" />
                     <h4 className="text-xs font-black uppercase tracking-widest">Vector Drift</h4>
                  </div>
                  <div className="flex items-center justify-between">
                     <span className="text-2xl font-black font-outfit">0.024</span>
                     <Badge className="bg-cyan-500 text-white border-none font-black text-[9px] uppercase tracking-tighter">Normal</Badge>
                  </div>
               </Card>
            </div>
         </div>
      </div>
    </div>
  );
}
