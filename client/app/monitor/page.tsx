'use client';

import { motion } from 'framer-motion';
import { Activity, Cpu, Server, Database, Globe, Zap, Clock } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { cn } from '@/lib/utils';
import { Badge } from '@/components/ui/badge';

export default function MonitorPage() {
  const hardware = [
    { label: 'Neural Throughput', value: '4.2 Gbps', status: 'Stable', usage: 65, color: 'bg-primary' },
    { label: 'Vector DB Latency', value: '1.2ms', status: 'Optimal', usage: 12, color: 'bg-cyan-500' },
    { label: 'Memory Pressure', value: '18.4 GB', status: 'Healthy', usage: 45, color: 'bg-indigo-500' },
    { label: 'Inference Queue', value: '0.0ms', status: 'Empty', usage: 5, color: 'bg-emerald-500' },
  ];

  const logs = [
    { time: '14:22:04', event: 'New inference request for JobID: 9822', status: 'OK' },
    { time: '14:22:05', event: 'Vector embedding generated (384 dimensions)', status: 'OK' },
    { time: '14:22:05', event: 'Classification complete: REAL (0.92 conf)', status: 'OK' },
    { time: '14:22:12', event: 'Periodic dataset refresh initiated', status: 'INFO' },
    { time: '14:22:15', event: 'Anomalous linguistic pattern detected in Stream-X', status: 'WARN' },
  ];

  return (
    <div className="max-w-[1600px] mx-auto space-y-16 pb-20 px-4">
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-8 border-b border-border pb-10">
        <div className="space-y-4">
          <div className="flex items-center gap-3">
             <div className="p-2 bg-primary/10 rounded-lg">
                <Activity className="h-5 w-5 text-primary" />
             </div>
             <span className="text-[10px] font-black uppercase tracking-[0.4em] text-muted-foreground">System Path: /root/intelligence/monitor</span>
          </div>
          <h1 className="text-5xl md:text-6xl font-black font-outfit text-foreground leading-tight">
             System <span className="text-gradient italic">Telemetry</span>
          </h1>
          <p className="text-muted-foreground text-lg font-medium max-w-2xl leading-relaxed">
             Real-time infrastructure health and neural processing throughput across the global cluster.
          </p>
        </div>
        <div className="flex items-center gap-4 px-6 py-3 glass-panel rounded-2xl border-border">
            <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
            <span className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">All Clusters Green</span>
        </div>
      </div>

      <div className="grid gap-8">
         <Card className="glass-card border-none rounded-3xl bg-foreground/[0.01] overflow-hidden min-h-[800px] relative">
            <div className="absolute inset-0 flex items-center justify-center -z-10">
               <div className="flex flex-col items-center gap-4 text-muted-foreground">
                  <Activity className="h-12 w-12 animate-pulse" />
                  <p className="font-outfit font-bold">Connecting to Grafana Engine...</p>
               </div>
            </div>
            <iframe 
              src="http://localhost:3001/d/system-telemetry/system-telemetry?orgId=1&refresh=5s&kiosk" 
              className="w-full h-[800px] border-none"
              title="Grafana Dashboard"
            />
         </Card>
      </div>

      <div className="grid gap-8 md:grid-cols-3">
         {[
           { icon: Cpu, label: 'Node Distribution', value: 'Eu-West (Primary)' },
           { icon: Database, label: 'Model Version', value: 'Neural_Core_v2.1' },
           { icon: Globe, label: 'Traffic Origin', value: '84 Cities globally' },
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


