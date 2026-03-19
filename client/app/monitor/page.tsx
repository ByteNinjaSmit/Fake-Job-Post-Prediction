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
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-8 border-b border-white/5 pb-10">
        <div className="space-y-4">
          <div className="flex items-center gap-3">
             <div className="p-2 bg-primary/10 rounded-lg">
                <Activity className="h-5 w-5 text-primary" />
             </div>
             <span className="text-[10px] font-black uppercase tracking-[0.4em] text-zinc-500">System Path: /root/intelligence/monitor</span>
          </div>
          <h1 className="text-5xl md:text-6xl font-black font-outfit text-white leading-tight">
             System <span className="text-gradient italic">Telemetry</span>
          </h1>
          <p className="text-zinc-400 text-lg font-medium max-w-2xl leading-relaxed">
             Real-time infrastructure health and neural processing throughput across the global cluster.
          </p>
        </div>
        <div className="flex items-center gap-4 px-6 py-3 glass-panel rounded-2xl border-white/10">
            <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
            <span className="text-[10px] font-black uppercase tracking-widest text-zinc-300">All Clusters Green</span>
        </div>
      </div>

      <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-4">
         {hardware.map((item, i) => (
           <Card key={i} className="glass-card border-none rounded-3xl p-6 bg-white/[0.01] hover:bg-white/[0.03] transition-all group">
              <div className="flex items-center justify-between mb-6">
                 <div className="p-2 bg-white/5 rounded-xl text-primary group-hover:scale-110 transition-transform">
                    <Activity className="h-4 w-4" />
                 </div>
                 <Badge className="bg-white/10 text-white border-none font-bold px-2 py-0.5 rounded-lg text-[9px] uppercase tracking-tighter">
                   {item.status}
                 </Badge>
              </div>
              <p className="text-[10px] font-black uppercase text-muted-foreground tracking-widest mb-1">{item.label}</p>
              <h4 className="text-2xl font-black font-outfit mb-4">{item.value}</h4>
              <Progress value={item.usage} className={`h-1 bg-white/5 [&>div]:bg-${item.color}`} />
           </Card>
         ))}
      </div>

      <div className="grid gap-8 lg:grid-cols-3">
         <Card className="lg:col-span-2 glass-card border-none rounded-3xl bg-white/[0.01] overflow-hidden">
            <CardHeader className="p-8 border-b border-white/5 flex flex-row items-center justify-between bg-white/[0.02]">
               <CardTitle className="text-xl font-bold font-outfit">Live Telemetry</CardTitle>
               <div className="flex gap-2">
                  <Badge variant="outline" className="border-white/10 text-xs py-1">Requests: 142/min</Badge>
                  <Badge variant="outline" className="border-white/10 text-xs py-1">Avg Latency: 84ms</Badge>
               </div>
            </CardHeader>
            <CardContent className="p-0">
               <div className="h-80 w-full p-8 flex items-end gap-2">
                  {[...Array(40)].map((_, i) => (
                    <motion.div 
                       key={i} 
                       initial={{ height: 0 }}
                       animate={{ height: `${20 + Math.random() * 80}%` }}
                       transition={{ repeat: Infinity, duration: 2, repeatType: 'reverse', delay: i * 0.05 }}
                       className="flex-1 bg-primary/20 rounded-t-sm" 
                    />
                  ))}
               </div>
            </CardContent>
         </Card>

         <Card className="glass-card border-none rounded-3xl bg-white/[0.01] flex flex-col">
            <CardHeader className="p-8 border-b border-white/5">
                <div className="flex items-center gap-3">
                  <Server className="h-5 w-5 text-primary" />
                  <CardTitle className="text-xl font-bold font-outfit">Cluster Logs</CardTitle>
                </div>
            </CardHeader>
            <CardContent className="flex-1 p-8 space-y-6 overflow-y-auto font-mono text-[10px] scrollbar-hide">
               {logs.map((log, i) => (
                 <div key={i} className="flex gap-4 border-l border-white/10 pl-4 relative group">
                    <div className="absolute -left-0.5 top-0 h-2 w-1 bg-primary rounded-full opacity-0 group-hover:opacity-100 transition-opacity" />
                    <span className="text-muted-foreground tabular-nums">{log.time}</span>
                    <span className="flex-1 text-foreground/80">{log.event}</span>
                    <span className={`font-black ${log.status === 'OK' ? 'text-green-500' : log.status === 'WARN' ? 'text-amber-500' : 'text-primary'}`}>
                      {log.status}
                    </span>
                 </div>
               ))}
               <div className="flex items-center gap-2 pt-4 opacity-30">
                  <div className="h-1.5 w-1.5 rounded-full bg-primary animate-pulse" />
                  <span className="italic">Monitoring live streams...</span>
               </div>
            </CardContent>
         </Card>
      </div>

      <div className="grid gap-8 md:grid-cols-3">
         {[
           { icon: Cpu, label: 'Node Distribution', value: 'Eu-West (Primary)' },
           { icon: Database, label: 'Model Version', value: 'Neural_Core_v2.1' },
           { icon: Globe, label: 'Traffic Origin', value: '84 Cities globally' },
         ].map((item, i) => (
           <div key={i} className="glass-card border-none p-6 rounded-3xl flex items-center gap-4 bg-white/[0.01]">
              <div className="p-3 bg-white/5 rounded-2xl text-primary">
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


