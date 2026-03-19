'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  Legend,
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis
} from 'recharts';
import { BarChart3, TrendingUp, Cpu, PieChart, Activity } from 'lucide-react';
import { motion } from 'framer-motion';

const modelData = [
  { name: 'Neural Net (Transformer)', accuracy: 0.99, precision: 0.98, recall: 0.97, f1: 0.985 },
  { name: 'XGBoost Ensemble', accuracy: 0.98, precision: 0.97, recall: 0.95, f1: 0.96 },
  { name: 'Random Forest', accuracy: 0.97, precision: 0.96, recall: 0.94, f1: 0.95 },
  { name: 'Logistic Baseline', accuracy: 0.91, precision: 0.90, recall: 0.88, f1: 0.89 },
];

const chartData = [
  { metric: 'Accuracy', nn: 99, xg: 98, rf: 97, lb: 91 },
  { metric: 'Precision', nn: 98.4, xg: 97.2, rf: 96.5, lb: 90 },
  { metric: 'Recall', nn: 97.5, xg: 95.8, rf: 94.2, lb: 88 },
  { metric: 'F1 Score', nn: 98.2, xg: 96.5, rf: 95.3, lb: 89 },
];

const radarData = [
  { subject: 'NLP Precision', A: 99, B: 92, fullMark: 100 },
  { subject: 'Prediction Latency', A: 82, B: 98, fullMark: 100 },
  { subject: 'Interpretability', A: 75, B: 95, fullMark: 100 },
  { subject: 'Concept Drift', A: 94, B: 88, fullMark: 100 },
  { subject: 'Data Coverage', A: 98, B: 94, fullMark: 100 },
];

export default function ModelsPage() {
  return (
    <motion.div 
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      className="space-y-12 pb-12"
    >
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
        <div>
          <h1 className="text-4xl font-extrabold tracking-tight font-outfit text-gradient">Model Benchmarks</h1>
          <p className="text-muted-foreground mt-2 text-lg font-medium max-w-xl">
             In-depth performance analysis across the JobGuard model spectrum.
          </p>
        </div>
        <div className="flex items-center gap-4 px-6 py-3 glass-card rounded-2xl">
            <TrendingUp className="h-5 w-5 text-indigo-500" />
            <span className="text-sm font-bold uppercase tracking-widest text-foreground/80">Active Benchmark: Mar 19</span>
        </div>
      </div>

      <div className="grid gap-8 lg:grid-cols-2">
        <Card className="glass-card border-none shadow-2xl relative overflow-hidden group">
          <div className="absolute top-0 right-0 w-32 h-32 bg-indigo-500/10 rounded-full blur-3xl -mr-16 -mt-16 group-hover:scale-150 transition-transform duration-1000" />
          <CardHeader>
            <CardTitle className="text-xl font-bold font-outfit">Core Metric Spectrum</CardTitle>
            <CardDescription>Comparative analysis of weighted F1 and Accuracy scores.</CardDescription>
          </CardHeader>
          <CardContent className="h-[400px] pt-4">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
                <defs>
                   <linearGradient id="colorNN" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="var(--primary)" stopOpacity={1}/>
                      <stop offset="95%" stopColor="var(--primary)" stopOpacity={0.6}/>
                   </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="metric" axisLine={false} tickLine={false} tick={{ fill: 'rgba(255,255,255,0.4)', fontSize: 12 }} />
                <YAxis domain={[80, 100]} axisLine={false} tickLine={false} tick={{ fill: 'rgba(255,255,255,0.4)', fontSize: 12 }} />
                <Tooltip 
                  cursor={{ fill: 'rgba(255,255,255,0.03)' }}
                  contentStyle={{ backgroundColor: 'rgba(15,23,42,0.9)', backdropFilter: 'blur(10px)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '16px', boxShadow: '0 20px 40px rgba(0,0,0,0.4)' }}
                />
                <Legend iconType="circle" wrapperStyle={{ paddingTop: '20px' }} />
                <Bar dataKey="nn" name="Neural Core" fill="url(#colorNN)" radius={[6, 6, 0, 0]} />
                <Bar dataKey="xg" name="XGBoost" fill="oklch(0.65 0.22 190)" radius={[6, 6, 0, 0]} />
                <Bar dataKey="rf" name="Random Forest" fill="oklch(0.7 0.15 150)" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card className="glass-card border-none shadow-2xl relative overflow-hidden group">
          <CardHeader>
            <CardTitle className="text-xl font-bold font-outfit">Architecture Profile</CardTitle>
            <CardDescription>Neural core (Primary) vs XGBoost (Secondary) structural advantages.</CardDescription>
          </CardHeader>
          <CardContent className="h-[400px] pt-4 flex items-center justify-center">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarData}>
                <PolarGrid stroke="rgba(255,255,255,0.1)" />
                <PolarAngleAxis dataKey="subject" tick={{ fill: 'rgba(255,255,255,0.6)', fontSize: 11 }} />
                <PolarRadiusAxis angle={30} domain={[0, 100]} axisLine={false} tick={false} />
                <Radar name="Neural Core" dataKey="A" stroke="var(--primary)" fill="var(--primary)" fillOpacity={0.6} />
                <Radar name="XGBoost" dataKey="B" stroke="oklch(0.65 0.22 190)" fill="oklch(0.65 0.22 190)" fillOpacity={0.4} />
                <Legend iconType="diamond" wrapperStyle={{ paddingTop: '20px' }} />
              </RadarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      <Card className="glass-card border-none shadow-2xl overflow-hidden">
        <CardHeader className="p-8 border-b border-border bg-foreground/5">
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-2xl font-bold font-outfit">Validation Analytics (EMSCAD)</CardTitle>
                <CardDescription className="text-base">Comprehensive performance scoring across the master training set.</CardDescription>
              </div>
              <Badge className="premium-gradient font-bold px-4 py-1.5 rounded-full border-none">Production Ready</Badge>
            </div>
        </CardHeader>
        <CardContent className="p-0">
          <Table>
            <TableHeader className="bg-foreground/5">
              <TableRow className="hover:bg-transparent border-border">
                <TableHead className="w-[300px] pl-8 font-bold text-foreground/70 tracking-widest uppercase text-[10px]">Model Architecture</TableHead>
                <TableHead className="font-bold text-foreground/70 tracking-widest uppercase text-[10px]">Accuracy</TableHead>
                <TableHead className="font-bold text-foreground/70 tracking-widest uppercase text-[10px]">Precision</TableHead>
                <TableHead className="font-bold text-foreground/70 tracking-widest uppercase text-[10px]">Recall</TableHead>
                <TableHead className="font-bold text-foreground/70 tracking-widest uppercase text-[10px]">F1 Score</TableHead>
                <TableHead className="text-right pr-8 font-bold text-foreground/70 tracking-widest uppercase text-[10px]">Health</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {modelData.map((model, idx) => (
                <TableRow key={model.name} className="border-border group hover:bg-foreground/5 transition-colors">
                  <TableCell className="font-bold pl-8 py-5 text-base font-outfit group-hover:text-primary transition-colors">{model.name}</TableCell>
                  <TableCell className="font-mono text-sm">{(model.accuracy * 100).toFixed(1)}%</TableCell>
                  <TableCell className="font-mono text-sm">{(model.precision * 100).toFixed(1)}%</TableCell>
                  <TableCell className="font-mono text-sm">{(model.recall * 100).toFixed(1)}%</TableCell>
                  <TableCell className="font-mono text-sm font-bold text-primary">{(model.f1 * 100).toFixed(1)}%</TableCell>
                  <TableCell className="text-right pr-8">
                     <div className="flex items-center justify-end gap-2 text-xs font-bold text-green-500">
                        <Activity className="h-3 w-3 animate-pulse" /> SYNCED
                     </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <div className="grid gap-8 md:grid-cols-3">
         <motion.div whileHover={{ y: -5 }}>
           <Card className="glass-card border-none p-2 overflow-hidden relative group">
             <div className="absolute top-0 right-0 w-24 h-24 bg-primary/20 rounded-full blur-3xl -mr-12 -mt-12 group-hover:scale-150 transition-transform" />
             <CardHeader className="pb-2">
               <CardTitle className="text-xs font-black uppercase tracking-widest text-primary flex items-center gap-2">
                 <Cpu className="h-3 w-3" /> Hardware Engine
               </CardTitle>
             </CardHeader>
             <CardContent>
               <div className="text-3xl font-extrabold font-outfit">T4 Tensor</div>
               <p className="text-xs text-muted-foreground mt-2 font-medium">Mixed-precision accelerated inference.</p>
             </CardContent>
           </Card>
         </motion.div>
         
         <motion.div whileHover={{ y: -5 }}>
           <Card className="glass-card border-none p-2 overflow-hidden relative group">
             <div className="absolute top-0 right-0 w-24 h-24 bg-cyan-500/20 rounded-full blur-3xl -mr-12 -mt-12 group-hover:scale-150 transition-transform" />
             <CardHeader className="pb-2">
               <CardTitle className="text-xs font-black uppercase tracking-widest text-cyan-500 flex items-center gap-2">
                 <Activity className="h-3 w-3" /> Training Cycle
               </CardTitle>
             </CardHeader>
             <CardContent>
               <div className="text-3xl font-extrabold font-outfit">14.2 min</div>
               <p className="text-xs text-muted-foreground mt-2 font-medium">Distributed training over 17k nodes.</p>
             </CardContent>
           </Card>
         </motion.div>

         <motion.div whileHover={{ y: -5 }}>
           <Card className="glass-card border-none p-2 overflow-hidden relative group">
             <div className="absolute top-0 right-0 w-24 h-24 bg-indigo-500/20 rounded-full blur-3xl -mr-12 -mt-12 group-hover:scale-150 transition-transform" />
             <CardHeader className="pb-2">
               <CardTitle className="text-xs font-black uppercase tracking-widest text-indigo-500 flex items-center gap-2">
                 <PieChart className="h-3 w-3" /> Data Volume
               </CardTitle>
             </CardHeader>
             <CardContent>
               <div className="text-3xl font-extrabold font-outfit">17,880</div>
               <p className="text-xs text-muted-foreground mt-2 font-medium">Job descriptions in master corpus.</p>
             </CardContent>
           </Card>
         </motion.div>
      </div>
    </motion.div>
  );
}
