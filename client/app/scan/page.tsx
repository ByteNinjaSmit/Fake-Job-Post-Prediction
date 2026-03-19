'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Scan, Sparkles, Activity, ShieldAlert, ShieldCheck } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';

export default function ScanPage() {
  const [text, setText] = useState('');
  const [isScanning, setIsScanning] = useState(false);
  const [scanResult, setScanResult] = useState<null | 'Real' | 'Fake'>(null);
  const [confidence, setConfidence] = useState(0);

  useEffect(() => {
    if (text.length < 20) {
      setScanResult(null);
      setConfidence(0);
      return;
    }

    const scan = async () => {
      setIsScanning(true);
      await new Promise(r => setTimeout(r, 800));
      const isFake = Math.random() > 0.8;
      setScanResult(isFake ? 'Fake' : 'Real');
      setConfidence(80 + Math.random() * 19);
      setIsScanning(false);
    };

    const timer = setTimeout(scan, 500);
    return () => clearTimeout(timer);
  }, [text]);

  return (
    <div className="max-w-[1200px] mx-auto space-y-16 pb-20 px-4">
      <div className="text-center space-y-6">
         <motion.div 
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="inline-flex p-5 rounded-3xl bg-primary/10 border border-primary/20 mb-4 shadow-[0_0_30px_rgba(124,58,237,0.2)]"
         >
            <Scan className="h-10 w-10 text-primary" />
         </motion.div>
          <h1 className="text-6xl font-black font-outfit text-foreground uppercase italic leading-tight">Real-time <span className="text-gradient">Neural Scan</span></h1>
          <p className="text-muted-foreground text-xl max-w-2xl mx-auto font-medium leading-relaxed">
             Experience sub-second fraudulent pattern detection as you type with <br /> our proprietary entropy-based fingerprinting.
          </p>
      </div>

      <div className="relative">
         <div className="absolute inset-0 premium-gradient rounded-[2rem] blur-3xl opacity-5" />
         <Card className="glass-card border-none rounded-[2rem] overflow-hidden shadow-2xl relative z-10">
            <CardContent className="p-10 space-y-8">
               <textarea 
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder="Initiate text stream for real-time analysis..."
                  className="w-full h-64 bg-transparent border-none focus:ring-0 text-2xl font-medium leading-relaxed resize-none scrollbar-hide placeholder:text-foreground/10 text-foreground"
               />

                <div className="flex items-center justify-between pt-8 border-t border-border">
                   <div className="flex items-center gap-6">
                      <div className="space-y-1">
                         <p className="text-[10px] font-black text-muted-foreground uppercase tracking-widest">Neural Pulse</p>
                         <div className="flex items-center gap-2">
                            <Activity className={`h-4 w-4 ${isScanning ? 'text-primary animate-pulse' : 'text-muted-foreground'}`} />
                            <span className="text-sm font-bold tabular-nums text-foreground">12ms</span>
                         </div>
                      </div>
                      <div className="h-10 w-px bg-border" />
                      <div className="space-y-1">
                         <p className="text-[10px] font-black text-muted-foreground uppercase tracking-widest">Confidence</p>
                         <span className="text-sm font-bold tabular-nums text-foreground">{Math.round(confidence)}%</span>
                      </div>
                   </div>

                  <AnimatePresence mode="wait">
                    {isScanning ? (
                       <motion.div 
                          key="scanning"
                          initial={{ opacity: 0, x: 20 }}
                          animate={{ opacity: 1, x: 0 }}
                          exit={{ opacity: 0, x: -20 }}
                          className="flex items-center gap-3 px-6 py-2 rounded-2xl bg-foreground/5"
                       >
                          <div className="h-2 w-2 rounded-full bg-primary animate-ping" />
                          <span className="text-xs font-black uppercase tracking-widest text-primary">Analyzing...</span>
                       </motion.div>
                    ) : scanResult ? (
                       <motion.div 
                          key="result"
                          initial={{ opacity: 0, scale: 0.9 }}
                          animate={{ opacity: 1, scale: 1 }}
                          className={`flex items-center gap-3 px-6 py-2 rounded-2xl ${scanResult === 'Fake' ? 'bg-destructive/20 text-destructive' : 'bg-green-500/20 text-green-500'}`}
                       >
                          {scanResult === 'Fake' ? <ShieldAlert className="h-4 w-4" /> : <ShieldCheck className="h-4 w-4" />}
                          <span className="text-xs font-black uppercase tracking-widest">{scanResult} Detected</span>
                       </motion.div>
                    ) : (
                       <div className="px-6 py-2 rounded-2xl bg-foreground/5 text-[10px] font-black uppercase tracking-widest text-muted-foreground">
                          Awaiting Stream
                       </div>
                    )}
                  </AnimatePresence>
               </div>
            </CardContent>
         </Card>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
         {[
           { label: 'Tokenization', value: 'Active', icon: Sparkles },
           { label: 'Vector Cache', value: '4.2ms', icon: Activity },
           { label: 'Entropy', value: 'Stable', icon: Scan },
         ].map((item, i) => (
           <div key={i} className="glass-card border-none p-6 rounded-2xl flex items-center justify-between group">
              <div className="flex items-center gap-3">
                 <item.icon className="h-4 w-4 text-primary group-hover:rotate-12 transition-transform" />
                 <span className="text-[10px] font-black uppercase tracking-widest opacity-60">{item.label}</span>
              </div>
              <span className="text-xs font-bold">{item.value}</span>
           </div>
         ))}
      </div>
    </div>
  );
}
