'use client';

import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Zap, Info, ShieldAlert, CheckCircle2, Search, ArrowRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { motion, Variants } from "framer-motion";

const container: Variants = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1
    }
  }
};

const item: Variants = {
  hidden: { y: 20, opacity: 0 },
  show: { y: 0, opacity: 1, transition: { type: 'spring', stiffness: 300, damping: 24 } }
};

export default function ExplainPage() {
  const sampleText = [
    { word: "Entry", weight: 0.1 },
    { word: "Level", weight: 0.1 },
    { word: "Admin", weight: 0.2 },
    { word: "Position", weight: 0.05 },
    { word: "with", weight: 0 },
    { word: "Urgent", weight: 0.82 },
    { word: "Hire", weight: 0.78 },
    { word: "potential.", weight: 0.12 },
    { word: "Earn", weight: 0.45 },
    { word: "$5000/week", weight: 0.98 },
    { word: "working", weight: 0.08 },
    { word: "from", weight: 0 },
    { word: "home.", weight: 0.65 },
    { word: "No", weight: 0.15 },
    { word: "experience", weight: 0.22 },
    { word: "required.", weight: 0.75 },
    { word: "Apply", weight: 0.1 },
    { word: "Fast", weight: 0.3 }
  ];

  return (
    <motion.div 
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="space-y-12 pb-12"
    >
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
        <div>
          <h1 className="text-4xl font-extrabold tracking-tight font-outfit text-gradient">Explainability Core</h1>
          <p className="text-muted-foreground mt-2 text-lg font-medium max-w-xl">
             Deconstructing neural decisions with local feature mapping.
          </p>
        </div>
        <div className="flex items-center gap-4 px-6 py-3 glass-card rounded-2xl">
            <Zap className="h-5 w-5 text-indigo-500 fill-indigo-500/20" />
            <span className="text-sm font-bold uppercase tracking-widest text-foreground/80">Analysis: LIME Engine</span>
        </div>
      </div>

      <div className="grid gap-8 lg:grid-cols-3 items-start">
        <Card className="lg:col-span-2 glass-card border-none shadow-2xl relative overflow-hidden group">
          <div className="absolute top-0 left-0 w-64 h-64 bg-primary/10 rounded-full blur-[100px] pointer-events-none" />
          <CardHeader className="p-10 border-b border-white/5 relative z-10">
            <CardTitle className="text-2xl font-bold font-outfit">Linguistic Importance Mapping</CardTitle>
            <CardDescription className="text-base mt-2">
              The model identifies specific tokens that shift the prediction probability.
            </CardDescription>
          </CardHeader>
          <CardContent className="p-10 relative z-10">
            <div className="p-10 rounded-3xl bg-white/5 border border-white/10 text-2xl leading-[1.8] font-outfit tracking-tight">
              {sampleText.map((item, idx) => {
                const isPositive = item.weight > 0.4;
                const isVeryPositive = item.weight > 0.7;
                
                return (
                  <motion.span 
                    key={idx}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 + idx * 0.05 }}
                    className={`inline-block mx-1.5 px-2 rounded-xl border-b-2 font-medium transition-all duration-300 relative group cursor-default ${
                        isVeryPositive 
                          ? 'bg-destructive/20 text-destructive border-destructive/50 shadow-lg shadow-destructive/10 -rotate-1' 
                          : isPositive 
                          ? 'bg-destructive/10 text-destructive/80 border-destructive/30'
                          : 'text-foreground/80 border-transparent'
                    }`}
                  >
                    {item.word}
                    {isVeryPositive && (
                        <span className="absolute -top-3 -right-2 px-1.5 py-0.5 bg-destructive text-white rounded-full text-[8px] font-black tracking-tighter opacity-0 group-hover:opacity-100 transition-opacity">
                            RISK
                        </span>
                    )}
                  </motion.span>
                );
              })}
            </div>

            <div className="mt-8 p-6 rounded-2xl bg-indigo-500/5 border border-indigo-500/20 flex items-start gap-4">
               <Info className="h-6 w-6 text-indigo-500 mt-1 shrink-0" />
               <p className="text-sm text-foreground/70 leading-relaxed font-medium">
                  The term <strong>"$5000/week"</strong> was identified as the highest outlier, contributing <strong>42.8%</strong> of the "Fake" classification probability. This aligns with statistical outliers in the EMSCAD salary distribution.
               </p>
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card border-none shadow-2xl sticky top-12">
          <CardHeader className="p-8 pb-4">
            <CardTitle className="text-sm font-black uppercase tracking-[0.2em] text-muted-foreground">Pulse Legend</CardTitle>
          </CardHeader>
          <CardContent className="p-8 pt-0 space-y-8">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-2 text-sm font-bold"><ShieldAlert className="h-4 w-4 text-destructive" /> Fake Impact</span>
                <Badge className="bg-destructive text-white border-none text-[10px] font-black px-3 rounded-full">SEVERE</Badge>
              </div>
              <div className="h-2.5 w-full bg-destructive/10 rounded-full overflow-hidden">
                <motion.div 
                  initial={{ width: 0 }}
                  animate={{ width: '85%' }}
                  transition={{ duration: 1, delay: 0.5 }}
                  className="h-full bg-destructive shadow-[0_0_15px_rgba(239,68,68,0.5)]" 
                />
              </div>
              <p className="text-[10px] text-muted-foreground font-bold tracking-tight">Anomalous patterns linked to known fraud vectors.</p>
            </div>

            <div className="space-y-3 pt-4 border-t border-white/5">
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-2 text-sm font-bold"><CheckCircle2 className="h-4 w-4 text-green-500" /> Real Impact</span>
                <Badge className="bg-green-500 text-white border-none text-[10px] font-black px-3 rounded-full">SECURE</Badge>
              </div>
              <div className="h-2.5 w-full bg-green-500/10 rounded-full overflow-hidden">
                <motion.div 
                    initial={{ width: 0 }}
                    animate={{ width: '70%' }}
                    transition={{ duration: 1, delay: 0.7 }}
                    className="h-full bg-green-500 shadow-[0_0_15px_rgba(34,197,94,0.5)]" 
                />
              </div>
              <p className="text-[10px] text-muted-foreground font-bold tracking-tight">Standard linguistic markers from verified organizations.</p>
            </div>

            <div className="pt-8 mt-8 border-t border-white/5 space-y-4">
               <h4 className="text-xs font-black uppercase tracking-widest text-foreground/50">Core Engine</h4>
               <p className="text-xs text-muted-foreground leading-relaxed font-medium italic">
                  "LIME creates a locally faithful explanation by perturbing inputs and observing black-box outputs."
               </p>
               <Button variant="outline" className="w-full rounded-2xl glass-card h-11 font-bold text-xs" asChild>
                   <a href="https://github.com/marcotcr/lime" target="_blank">Documentation <Search className="ml-2 h-3.5 w-3.5" /></a>
               </Button>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 md:grid-cols-4">
         {[
           { word: "$5000/week", impact: "+42%", type: "Fake", trend: "Severe" },
           { word: "Urgent Hire", impact: "+35%", type: "Fake", trend: "High" },
           { word: "No experience", impact: "+18%", type: "Fake", trend: "Medium" },
           { word: "Engineering", impact: "-22%", type: "Real", trend: "Secure" },
         ].map((feat, i) => (
           <motion.div 
             key={i} 
             whileHover={{ scale: 1.05 }}
             className="glass-card p-6 rounded-3xl flex flex-col items-center justify-center text-center space-y-3 group"
           >
              <span className="font-outfit text-xl font-bold group-hover:text-primary transition-colors">{feat.word}</span>
              <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-white/5 border border-white/5">
                 <span className={`text-[10px] font-black ${feat.type === 'Fake' ? 'text-destructive' : 'text-green-500'}`}>{feat.impact}</span>
                 <div className={`h-1.5 w-1.5 rounded-full ${feat.type === 'Fake' ? 'bg-destructive' : 'bg-green-500'}`} />
              </div>
              <p className="text-[9px] uppercase font-black tracking-[0.2em] text-muted-foreground">{feat.trend}</p>
           </motion.div>
         ))}
      </div>
    </motion.div>
  );
}
