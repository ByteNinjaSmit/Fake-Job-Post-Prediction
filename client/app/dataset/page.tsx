'use client';

import { motion } from 'framer-motion';
import { Database, TrendingUp, PieChart, Users, ArrowRight, Table as TableIcon } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  Cell,
  PieChart as RePieChart,
  Pie
} from 'recharts';

const data = [
  { name: 'Admin', count: 4200, fraud: 1200 },
  { name: 'Tech', count: 5800, fraud: 150 },
  { name: 'Sales', count: 3100, fraud: 800 },
  { name: 'Eng', count: 2900, fraud: 40 },
  { name: 'Service', count: 1880, fraud: 600 },
];

const pieData = [
  { name: 'Real Posts', value: 14600, color: '#7c3aed' },
  { name: 'Fraudulent', value: 1280, color: '#ef4444' },
];

export default function DatasetExplorer() {
  return (
    <div className="max-w-[1600px] mx-auto space-y-16 pb-20 px-4">
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-8 border-b border-border pb-10">
        <div className="space-y-4">
          <div className="flex items-center gap-3">
             <div className="p-2 bg-primary/10 rounded-lg">
                <Database className="h-5 w-5 text-primary" />
             </div>
             <span className="text-[10px] font-black uppercase tracking-[0.4em] text-muted-foreground">System Path: /root/intelligence/dataset</span>
          </div>
          <h1 className="text-5xl md:text-6xl font-black font-outfit text-foreground leading-tight">
             Corpus <span className="text-gradient italic">Explorer</span>
          </h1>
          <p className="text-muted-foreground text-lg font-medium max-w-2xl leading-relaxed">
             Deep-dive into the EMSCAD global threat repository and linguistic distribution statistics.
          </p>
        </div>
        <div className="flex items-center gap-4 px-6 py-3 glass-panel rounded-2xl border-border">
            <div className="h-2 w-2 rounded-full bg-primary animate-pulse" />
            <span className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">Archive: EMSCAD-G</span>
        </div>
      </div>

      <div className="grid gap-8 lg:grid-cols-3">
         <Card className="glass-card border-none rounded-3xl p-8 bg-foreground/[0.01]">
            <CardHeader className="p-0 mb-8">
               <CardTitle className="text-xl font-bold font-outfit">Class Distribution</CardTitle>
               <CardDescription>Overall label distribution in training set.</CardDescription>
            </CardHeader>
            <CardContent className="p-0 h-64">
               <ResponsiveContainer width="100%" height="100%">
                  <RePieChart>
                    <Pie
                      data={pieData}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={80}
                      paddingAngle={5}
                      dataKey="value"
                    >
                      {pieData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip 
                      contentStyle={{ backgroundColor: 'var(--popover)', border: '1px solid var(--border)', borderRadius: '12px' }}
                      itemStyle={{ color: 'var(--foreground)' }}
                    />
                  </RePieChart>
               </ResponsiveContainer>
               <div className="mt-4 space-y-2">
                  {pieData.map((d, i) => (
                    <div key={i} className="flex items-center justify-between">
                       <div className="flex items-center gap-2">
                          <div className="h-2 w-2 rounded-full" style={{ backgroundColor: d.color }} />
                          <span className="text-xs text-muted-foreground">{d.name}</span>
                       </div>
                       <span className="text-xs font-bold">{d.value}</span>
                    </div>
                  ))}
               </div>
            </CardContent>
         </Card>

         <Card className="lg:col-span-2 glass-card border-none rounded-3xl p-8 bg-foreground/[0.01]">
            <CardHeader className="p-0 mb-8">
               <CardTitle className="text-xl font-bold font-outfit">Department Analysis</CardTitle>
               <CardDescription>Correlation between job sectors and fraud density.</CardDescription>
            </CardHeader>
            <CardContent className="p-0 h-64">
               <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                    <XAxis 
                      dataKey="name" 
                      axisLine={false} 
                      tickLine={false} 
                      tick={{ fill: 'rgba(255,255,255,0.4)', fontSize: 10, fontWeight: 800 }} 
                    />
                    <YAxis 
                      axisLine={false} 
                      tickLine={false} 
                      tick={{ fill: 'rgba(255,255,255,0.4)', fontSize: 10, fontWeight: 800 }} 
                    />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#0a0a0a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '12px' }}
                      cursor={{ fill: 'rgba(255,255,255,0.02)' }}
                    />
                    <Bar dataKey="count" fill="rgba(124, 58, 237, 0.2)" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="fraud" fill="#ef4444" radius={[4, 4, 0, 0]} />
                  </BarChart>
               </ResponsiveContainer>
            </CardContent>
         </Card>
      </div>

      <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-4">
         {[
           { icon: Users, label: 'Data Nodes', value: '17,880', trend: '+12%' },
           { icon: TrendingUp, label: 'Vocabulary', value: '42.4k', trend: 'Stable' },
           { icon: PieChart, label: 'Imbalance Ratio', value: '1:11', trend: 'Corrected' },
           { icon: TableIcon, label: 'Fields/Entry', value: '14', trend: 'Full' },
         ].map((stat, i) => (
           <Card key={i} className="glass-card border-none p-6 rounded-2xl bg-foreground/[0.01] hover:bg-foreground/[0.03] transition-all group">
              <div className="p-2 bg-foreground/5 rounded-xl text-primary w-fit mb-4 group-hover:scale-110 transition-transform">
                 <stat.icon className="h-4 w-4" />
              </div>
              <p className="text-[10px] font-black uppercase text-muted-foreground tracking-widest">{stat.label}</p>
              <div className="flex items-center justify-between mt-1">
                 <span className="text-xl font-black font-outfit">{stat.value}</span>
                 <span className="text-[9px] font-black text-primary px-1.5 py-0.5 bg-primary/10 rounded-md">{stat.trend}</span>
              </div>
           </Card>
         ))}
      </div>

      <div className="glass-card border-none rounded-3xl p-10 bg-primary/10 relative overflow-hidden group">
         <div className="absolute top-0 right-0 w-64 h-64 bg-primary/20 rounded-full blur-[80px] -mr-32 -mt-32" />
         <div className="relative z-10 space-y-4 max-w-2xl">
            <h3 className="text-2xl font-black font-outfit">Ready to contribute findings?</h3>
            <p className="text-muted-foreground font-medium leading-relaxed">
               All analysis data is anonymized and contributed back to the Global Threat Network to improve LSTM weights across the cluster.
            </p>
            <div className="pt-4">
               <button className="flex items-center gap-2 px-8 py-3 premium-gradient rounded-2xl font-black text-xs uppercase tracking-widest shadow-2xl shadow-primary/30 group">
                  Submit Anomaly Case <ArrowRight className="h-4 w-4 group-hover:translate-x-1 transition-transform" />
               </button>
            </div>
         </div>
      </div>
    </div>
  );
}
