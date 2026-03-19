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
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-8 border-b border-border pb-10">
        <div className="space-y-4">
          <div className="flex items-center gap-3">
             <div className="p-2 bg-primary/10 rounded-lg">
                <FlaskConical className="h-5 w-5 text-primary" />
             </div>
             <span className="text-[10px] font-black uppercase tracking-[0.4em] text-muted-foreground">System Path: /root/intelligence/lab</span>
          </div>
          <h1 className="text-5xl md:text-6xl font-black font-outfit text-foreground leading-tight">
             Model <span className="text-gradient italic">Laboratory</span>
          </h1>
          <p className="text-muted-foreground text-lg font-medium max-w-2xl leading-relaxed">
             Expose internal neural parameters and visualize classification boundary shifts with real-time feedback loops.
          </p>
        </div>
        <div className="flex items-center gap-4 px-6 py-3 glass-panel rounded-2xl border-border">
            <div className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">Kernel-01 Active</span>
        </div>
      </div>

      <div className="grid gap-8">
         <Card className="glass-card border-none rounded-3xl bg-foreground/[0.01] overflow-hidden min-h-[800px] relative">
            <div className="absolute inset-0 flex items-center justify-center -z-10">
               <div className="flex flex-col items-center gap-4 text-muted-foreground">
                  <FlaskConical className="h-12 w-12 animate-pulse" />
                  <p className="font-outfit font-bold">Initializing Prometheus Kernel...</p>
               </div>
            </div>
            <iframe 
              src="http://localhost:9090/graph" 
              className="w-full h-[800px] border-none"
              title="Prometheus UI"
            />
         </Card>
      </div>

      <div className="grid gap-8 md:grid-cols-3">
         {[
           { icon: Zap, label: 'Weight Entropy', value: 'Dynamic' },
           { icon: Binary, label: 'Vector Drift', value: '0.024' },
           { icon: Settings2, label: 'Kernel Version', value: 'v1.0.4-stable' },
         ].map((item, i) => (
           <div key={i} className="glass-card border-none p-6 rounded-3xl flex items-center gap-4 bg-foreground/[0.01]">
              <div className="p-3 bg-foreground/5 rounded-2xl text-primary">
                 <item.icon className="h-5 w-5" />
              </div>
              <div>
                 <p className="text-[10px] font-black uppercase text-muted-foreground tracking-widest">{item.label}</p>
                 <p className="text-sm font-bold">{item.value}</p>
              </div>
           </div>
         ))}
      </div>
    </div>
  );
}
