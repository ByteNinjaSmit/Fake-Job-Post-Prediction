'use client';

import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { 
  LayoutGrid, 
  Upload, 
  FileJson, 
  Play, 
  CheckCircle2, 
  AlertTriangle, 
  Activity, 
  Trash2,
  FileSpreadsheet
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { motion, AnimatePresence } from "framer-motion";
import { Textarea } from "@/components/ui/textarea";
import { 
  Table, 
  TableBody, 
  TableCell, 
  TableHead, 
  TableHeader, 
  TableRow 
} from "@/components/ui/table";
import api from '@/lib/api';

const SAMPLE_BATCH = {
  "posts": [
    {
      "title": "Data Scientist - Remote",
      "company_profile": "Leading AI research firm.",
      "description": "We are seeking a senior data scientist with expertise in deep learning.",
      "requirements": "PhD in CS or related field, 5+ years experience.",
      "benefits": "Competitive salary, health insurance."
    },
    {
      "title": "URGENT: Admin Assistant Needed",
      "company_profile": "Confidential hiring.",
      "description": "Earn $5000/week working from home. No experience required. Data entry only.",
      "requirements": "Must have a computer and internet.",
      "benefits": "Daily payments via wire transfer."
    }
  ]
};

export default function BatchPage() {
  const [jsonInput, setJsonInput] = useState(JSON.stringify(SAMPLE_BATCH, null, 2));
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);

  const handleBatchPredict = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const payload = JSON.parse(jsonInput);
      if (!payload.posts || !Array.isArray(payload.posts)) {
        throw new Error("Invalid format: 'posts' must be an array.");
      }
      const response = await api.post('/batch', payload);
      setResults(response.data.results);
    } catch (err: any) {
      console.error('Batch failed:', err);
      setError(err.message || 'Batch analysis failed. Check JSON format.');
    } finally {
      setIsLoading(false);
    }
  };

  const fraudulentCount = results.filter(r => r.prediction === 'Fraudulent').length;

  return (
    <div className="max-w-[1400px] mx-auto space-y-12 pb-20">
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
        <div>
          <h1 className="text-4xl font-extrabold tracking-tight font-outfit text-gradient">Batch Intelligence</h1>
          <p className="text-muted-foreground mt-2 text-lg font-medium max-w-xl">
             High-throughput linguistic analysis for large-scale recruitment datasets.
          </p>
        </div>
        <div className="flex items-center gap-4 px-6 py-3 glass-card rounded-2xl">
            <LayoutGrid className="h-5 w-5 text-primary" />
            <span className="text-sm font-bold uppercase tracking-widest text-foreground/80">Mode: Parallel Processing</span>
        </div>
      </div>

      <div className="grid gap-12 lg:grid-cols-5 items-start">
        <Card className="lg:col-span-2 glass-card border-none shadow-2xl h-full flex flex-col overflow-hidden">
          <CardHeader className="p-8 border-b border-border bg-foreground/[0.02]">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <FileJson className="h-5 w-5 text-primary" />
                <CardTitle className="text-xl font-bold font-outfit">Input Stream</CardTitle>
              </div>
              <Button 
                variant="ghost" 
                size="sm" 
                onClick={() => setJsonInput(JSON.stringify(SAMPLE_BATCH, null, 2))}
                className="text-[10px] font-black uppercase tracking-widest"
              >
                Reset Sample
              </Button>
            </div>
          </CardHeader>
          <CardContent className="p-0 flex-1 relative min-h-[500px]">
            <Textarea 
              value={jsonInput}
              onChange={(e) => setJsonInput(e.target.value)}
              className="w-full h-full min-h-[500px] bg-transparent border-none p-8 font-mono text-sm leading-relaxed resize-none focus-visible:ring-0 scrollbar-hide"
              placeholder='{"posts": [{"title": "...", "description": "..."}]}'
            />
            <div className="absolute bottom-6 right-6 flex gap-4">
               <Button 
                 onClick={handleBatchPredict} 
                 disabled={isLoading}
                 className="h-14 px-8 rounded-2xl premium-gradient text-white font-bold shadow-xl hover:scale-105 active:scale-95 transition-all"
               >
                 {isLoading ? <Activity className="h-5 w-5 animate-spin" /> : <Play className="h-5 w-5 mr-3" />}
                 {isLoading ? 'Processing...' : 'Execute Batch'}
               </Button>
            </div>
          </CardContent>
        </Card>

        <div className="lg:col-span-3 space-y-8 h-full">
           {results.length > 0 ? (
             <motion.div 
               initial={{ opacity: 0, y: 20 }}
               animate={{ opacity: 1, y: 0 }}
               className="space-y-8 h-full"
             >
                <div className="grid grid-cols-3 gap-6">
                   <Card className="glass-card border-none p-6 rounded-2xl flex flex-col items-center justify-center text-center">
                      <span className="text-[10px] font-black uppercase tracking-widest text-muted-foreground mb-1">Total Analysed</span>
                      <span className="text-3xl font-black font-outfit">{results.length}</span>
                   </Card>
                   <Card className="glass-card border-none p-6 rounded-2xl flex flex-col items-center justify-center text-center">
                      <span className="text-[10px] font-black uppercase tracking-widest text-destructive/80 mb-1">Anomalies</span>
                      <span className="text-3xl font-black font-outfit text-destructive">{fraudulentCount}</span>
                   </Card>
                   <Card className="glass-card border-none p-6 rounded-2xl flex flex-col items-center justify-center text-center">
                      <span className="text-[10px] font-black uppercase tracking-widest text-green-500/80 mb-1">Verified Real</span>
                      <span className="text-3xl font-black font-outfit text-green-500">{results.length - fraudulentCount}</span>
                   </Card>
                </div>

                <Card className="glass-card border-none shadow-2xl overflow-hidden h-full">
                   <div className="max-h-[600px] overflow-auto scrollbar-hide">
                     <Table>
                        <TableHeader className="bg-foreground/[0.03] sticky top-0 z-10">
                          <TableRow className="border-border hover:bg-transparent">
                            <TableHead className="w-[80px] text-[10px] font-black uppercase text-muted-foreground py-6 pl-8">Index</TableHead>
                            <TableHead className="text-[10px] font-black uppercase text-muted-foreground">Status</TableHead>
                            <TableHead className="text-[10px] font-black uppercase text-muted-foreground">Confidence</TableHead>
                            <TableHead className="text-[10px] font-black uppercase text-muted-foreground text-right pr-8">Actions</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {results.map((result, idx) => (
                            <TableRow key={idx} className="border-border hover:bg-foreground/[0.01] transition-colors">
                              <TableCell className="font-mono text-[11px] py-4 pl-8 opacity-40">#{idx + 1}</TableCell>
                              <TableCell>
                                <div className="flex items-center gap-2">
                                  <div className={`h-1.5 w-1.5 rounded-full ${result.prediction === 'Fraudulent' ? 'bg-destructive animate-pulse shadow-[0_0_8px_rgba(239,68,68,0.5)]' : 'bg-green-500'}`} />
                                  <span className={`text-[11px] font-black uppercase tracking-widest ${result.prediction === 'Fraudulent' ? 'text-destructive' : 'text-green-500'}`}>
                                    {result.prediction}
                                  </span>
                                </div>
                              </TableCell>
                              <TableCell className="font-outfit font-bold tabular-nums">
                                {(result.confidence * 100).toFixed(1)}%
                              </TableCell>
                              <TableCell className="text-right pr-8">
                                <Button variant="ghost" size="sm" className="h-8 w-8 p-0 rounded-full hover:bg-foreground/5">
                                   <Trash2 className="h-3 w-3 text-muted-foreground" />
                                </Button>
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                     </Table>
                   </div>
                </Card>
             </motion.div>
           ) : (
             <div className="glass-card border-none rounded-3xl h-full min-h-[500px] flex flex-col items-center justify-center p-12 text-center opacity-70">
                <div className="p-8 bg-foreground/[0.03] rounded-full mb-8 ring-1 ring-border">
                  <LayoutGrid className="h-12 w-12 text-muted-foreground" />
                </div>
                <h3 className="text-2xl font-bold font-outfit mb-4">Neural Buffer Empty</h3>
                <p className="text-muted-foreground max-w-[300px] mx-auto text-sm leading-relaxed">
                  Supply a valid JSON payload in the input stream to initiate bulk linguistic evaluation.
                </p>
                <div className="mt-8 flex gap-4">
                   <div className="flex items-center gap-2 px-4 py-2 rounded-xl bg-foreground/5 text-xs font-bold">
                      <FileJson className="h-4 w-4 text-primary" /> JSON
                   </div>
                   <div className="flex items-center gap-2 px-4 py-2 rounded-xl bg-foreground/5 text-xs font-bold opacity-30">
                      <FileSpreadsheet className="h-4 w-4" /> CSV (Coming Soon)
                   </div>
                </div>
             </div>
           )}
        </div>
      </div>

      {error && (
        <motion.div 
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-6 rounded-3xl bg-destructive/10 border border-destructive/20 flex items-center gap-4 text-destructive"
        >
          <AlertTriangle className="h-6 w-6 shrink-0" />
          <div className="flex-1">
            <h4 className="text-sm font-black uppercase tracking-widest">Analysis Fault</h4>
            <p className="text-xs font-medium mt-1">{error}</p>
          </div>
        </motion.div>
      )}
    </div>
  );
}
