'use client';

import { motion } from 'framer-motion';
import { FileText, Download, Share2, ShieldCheck, Zap, ArrowRight, Printer } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';

export default function ReportPage() {
  const recentReports = [
    { name: 'Weekly Fraud Audit - Mar 19', date: '2026-03-19', cases: 142, risk: 'Moderate' },
    { name: 'EMSCAD Sync Analysis', date: '2026-03-18', cases: 84, risk: 'Low' },
    { name: 'Critical Anomaly Export', date: '2026-03-15', cases: 12, risk: 'High' },
  ];

  return (
    <div className="max-w-[1400px] mx-auto space-y-16 pb-20 px-4">
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-8 border-b border-white/5 pb-10">
        <div className="space-y-4">
          <div className="flex items-center gap-3">
             <div className="p-2 bg-primary/10 rounded-lg">
                <FileText className="h-5 w-5 text-primary" />
             </div>
             <span className="text-[10px] font-black uppercase tracking-[0.4em] text-zinc-500">System Path: /root/intelligence/reports</span>
          </div>
          <h1 className="text-5xl md:text-6xl font-black font-outfit text-white leading-tight">
             Neural <span className="text-gradient italic">Archives</span>
          </h1>
          <p className="text-zinc-400 text-lg font-medium max-w-2xl leading-relaxed">
             Generate cryptographically signed audit logs and system performance insights.
          </p>
        </div>
        <div className="flex items-center gap-4">
           <Button className="rounded-2xl premium-gradient px-8 font-black uppercase tracking-widest text-[10px] h-14 shadow-2xl shadow-primary/30 border-t border-white/20 active:scale-95 transition-all">
              <Zap className="mr-3 h-4 w-4" /> Global Intelligence Export
           </Button>
        </div>
      </div>

      <div className="grid gap-8 lg:grid-cols-3">
         <Card className="lg:col-span-2 glass-card border-none rounded-3xl p-10 bg-white/[0.01]">
            <CardHeader className="p-0 mb-12">
               <div className="flex items-center gap-4 mb-4">
                  <div className="p-3 premium-gradient rounded-2xl shadow-xl">
                     <FileText className="h-6 w-6 text-white" />
                  </div>
                  <div>
                    <CardTitle className="text-2xl font-black font-outfit">Report Configuration</CardTitle>
                    <CardDescription>Select metrics and dimensions for your intelligence export.</CardDescription>
                  </div>
               </div>
            </CardHeader>
            <CardContent className="p-0 space-y-10">
               <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                  {[
                    { title: 'Neural Logs', desc: 'Include raw step-by-step thinking feed.', enabled: true },
                    { title: 'Vector Maps', desc: 'Linguistic distribution visualizations.', enabled: true },
                    { title: 'Risk Signatures', desc: 'Detailed rule-engine trigger data.', enabled: false },
                    { title: 'Cluster Telemetry', desc: 'Server health and queue metrics.', enabled: true },
                  ].map((field, i) => (
                    <div key={i} className="flex items-start gap-4 p-6 glass-card border-none rounded-2xl bg-white/[0.02] hover:bg-white/[0.05] transition-colors cursor-pointer group">
                       <div className={`mt-1 h-5 w-5 rounded-md border-2 border-primary/20 flex items-center justify-center transition-colors ${field.enabled ? 'bg-primary border-primary' : 'bg-transparent'}`}>
                          {field.enabled && <ShieldCheck className="h-3 w-3 text-white" />}
                       </div>
                       <div>
                          <p className="font-bold text-sm mb-1">{field.title}</p>
                          <p className="text-xs text-muted-foreground leading-relaxed">{field.desc}</p>
                       </div>
                    </div>
                  ))}
               </div>

               <div className="pt-8 border-t border-white/5">
                  <Button className="w-full h-16 rounded-3xl bg-white text-black hover:bg-white/90 font-black uppercase tracking-widest text-xs group">
                     Build Intel Package <ArrowRight className="ml-2 h-4 w-4 group-hover:translate-x-1 transition-transform" />
                  </Button>
               </div>
            </CardContent>
         </Card>

         <div className="space-y-8">
            <Card className="glass-card border-none rounded-3xl p-8 bg-white/[0.01]">
               <CardHeader className="p-0 mb-6">
                  <CardTitle className="text-sm font-black uppercase tracking-widest text-muted-foreground">Recent Archives</CardTitle>
               </CardHeader>
               <CardContent className="p-0 space-y-4">
                  {recentReports.map((report, i) => (
                    <div key={i} className="p-4 rounded-2xl bg-white/5 border border-white/5 flex items-center justify-between group hover:border-primary/20 transition-colors">
                       <div className="space-y-1">
                          <p className="text-xs font-bold truncate max-w-[120px]">{report.name}</p>
                          <p className="text-[10px] text-muted-foreground">{report.date} • {report.cases} cases</p>
                       </div>
                       <Button size="icon" variant="ghost" className="rounded-xl hover:bg-primary/20 hover:text-primary">
                          <Download className="h-4 w-4" />
                       </Button>
                    </div>
                  ))}
               </CardContent>
            </Card>

            <Card className="glass-card border-none rounded-3xl p-8 bg-[#7c3aed] text-white relative overflow-hidden group">
               <div className="absolute top-0 right-0 w-32 h-32 bg-white/20 rounded-full blur-[40px] -mr-16 -mt-16 group-hover:scale-150 transition-transform duration-1000" />
               <div className="relative z-10 flex flex-col h-full space-y-6">
                  <div className="p-2 bg-white/20 rounded-lg w-fit">
                     <Share2 className="h-4 w-4" />
                  </div>
                  <div className="space-y-2">
                     <h4 className="text-lg font-black font-outfit uppercase italic">Share Intelligence</h4>
                     <p className="text-xs text-white/70 leading-relaxed font-medium">Securely distribute findings to peer security organizations via neural-tunnel.</p>
                  </div>
                  <Button variant="secondary" className="w-full rounded-xl font-bold text-xs uppercase tracking-widest bg-white text-primary">
                     Connect Peers
                  </Button>
               </div>
            </Card>
         </div>
      </div>

      <div className="flex items-center justify-center gap-12 pt-12 opacity-30">
         <Printer className="h-8 w-8" />
         <div className="h-px flex-1 bg-gradient-to-r from-transparent via-white to-transparent" />
         <span className="text-[10px] font-black uppercase tracking-[0.5em] whitespace-nowrap">End of Audit Stream</span>
         <div className="h-px flex-1 bg-gradient-to-r from-white via-transparent to-transparent" />
      </div>
    </div>
  );
}
