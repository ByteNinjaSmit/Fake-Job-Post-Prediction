'use client';

import { motion } from 'framer-motion';
import { Database, TrendingUp, PieChart, Users, ArrowRight, Table as TableIcon, Loader2, AlertCircle } from 'lucide-react';
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
import { useEffect, useState } from 'react';

interface DatasetInfo {
  total_jobs: number;
  real_posts: number;
  fraudulent_posts: number;
  departments: { name: string; count: number; fraud: number }[];
  distribution: { name: string; value: number; color: string }[];
  locations: { name: string; count: number; fraud: number }[];
  vocabulary_size: number;
  fields_per_entry: number;
  imbalance_ratio: string;
}

export default function DatasetExplorer() {
  const [data, setData] = useState<DatasetInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchDatasetInfo = async () => {
      try {
        const response = await fetch('http://localhost:8000/dataset/info');
        if (!response.ok) {
          throw new Error('Failed to fetch dataset information');
        }
        const result = await response.json();
        setData(result);
      } catch (err: any) {
        setError(err.message || 'An error occurred while fetching data');
      } finally {
        setLoading(false);
      }
    };

    fetchDatasetInfo();
  }, []);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] space-y-4">
        <Loader2 className="h-12 w-12 text-primary animate-spin" />
        <p className="text-muted-foreground font-medium animate-pulse">Initializing Corpus Explorer...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] space-y-4">
        <AlertCircle className="h-12 w-12 text-destructive" />
        <p className="text-destructive font-bold text-xl">Connection Error</p>
        <p className="text-muted-foreground">{error}</p>
        <button 
          onClick={() => window.location.reload()}
          className="px-6 py-2 bg-primary text-primary-foreground rounded-xl text-sm font-bold uppercase tracking-widest mt-4"
        >
          Retry Connection
        </button>
      </div>
    );
  }

  if (!data) return null;

  const stats = [
    { icon: Users, label: 'Data Nodes', value: data.total_jobs.toLocaleString(), trend: '+12%' },
    { icon: TrendingUp, label: 'Vocabulary', value: (data.vocabulary_size / 1000).toFixed(1) + 'k', trend: 'Stable' },
    { icon: PieChart, label: 'Imbalance Ratio', value: data.imbalance_ratio, trend: 'Corrected' },
    { icon: TableIcon, label: 'Fields/Entry', value: data.fields_per_entry.toString(), trend: 'Full' },
  ];

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
                      data={data.distribution}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={80}
                      paddingAngle={5}
                      dataKey="value"
                    >
                      {data.distribution.map((entry, index) => (
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
                  {data.distribution.map((d, i) => (
                    <div key={i} className="flex items-center justify-between">
                       <div className="flex items-center gap-2">
                          <div className="h-2 w-2 rounded-full" style={{ backgroundColor: d.color }} />
                          <span className="text-xs text-muted-foreground">{d.name}</span>
                       </div>
                       <span className="text-xs font-bold">{d.value.toLocaleString()}</span>
                    </div>
                  ))}
               </div>
            </CardContent>
         </Card>

         <Card className="glass-card border-none rounded-3xl p-8 bg-foreground/[0.01]">
            <CardHeader className="p-0 mb-8">
               <CardTitle className="text-xl font-bold font-outfit">Department Analysis</CardTitle>
               <CardDescription>Correlation between job sectors and fraud density.</CardDescription>
            </CardHeader>
            <CardContent className="p-0 h-64">
               <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data.departments}>
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

         <Card className="glass-card border-none rounded-3xl p-8 bg-foreground/[0.01]">
            <CardHeader className="p-0 mb-8">
               <CardTitle className="text-xl font-bold font-outfit">Geographic Distribution</CardTitle>
               <CardDescription>Top threat origin nodes by country code.</CardDescription>
            </CardHeader>
            <CardContent className="p-0 h-64">
               <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data.locations} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
                    <XAxis type="number" hide />
                    <YAxis 
                      dataKey="name" 
                      type="category"
                      axisLine={false} 
                      tickLine={false} 
                      tick={{ fill: 'rgba(255,255,255,0.6)', fontSize: 10, fontWeight: 800 }} 
                    />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#0a0a0a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '12px' }}
                      cursor={{ fill: 'rgba(255,255,255,0.02)' }}
                    />
                    <Bar dataKey="count" fill="rgba(6, 182, 212, 0.2)" radius={[0, 4, 4, 0]} />
                    <Bar dataKey="fraud" fill="#ef4444" radius={[0, 4, 4, 0]} />
                  </BarChart>
               </ResponsiveContainer>
            </CardContent>
         </Card>
      </div>

      <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-4">
         {stats.map((stat, i) => (
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
