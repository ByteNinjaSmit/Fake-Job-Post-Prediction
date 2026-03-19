'use client';

import { useStore } from '@/lib/store/useStore';
import { motion } from 'framer-motion';
import { Search, Sparkles, Clipboard, Trash2 } from 'lucide-react';
import { Button } from '@/components/ui/button';

export function LiveInputPanel({ onAnalyze }: { onAnalyze: () => void }) {
  const { inputText, setInputText, isAnalyzing, reset } = useStore();

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
         <div className="flex items-center gap-3">
            <div className="p-2 premium-gradient rounded-xl shadow-lg">
                <Search className="h-5 w-5 text-white" />
            </div>
            <div>
              <h2 className="text-xl font-extrabold font-outfit">Neural Input Matrix</h2>
              <p className="text-xs text-muted-foreground font-medium">Paste job descriptions for linguistic fingerprinting.</p>
            </div>
         </div>
         <div className="flex gap-2">
            <Button 
                variant="ghost" 
                size="icon" 
                onClick={reset}
                className="rounded-xl hover:bg-destructive/10 hover:text-destructive group"
            >
              <Trash2 className="h-4 w-4 transition-transform group-hover:scale-110" />
            </Button>
            <Button 
                variant="ghost" 
                size="icon" 
                className="rounded-xl hover:bg-primary/10 hover:text-primary group"
                onClick={async () => {
                   const text = await navigator.clipboard.readText();
                   setInputText(text);
                }}
            >
              <Clipboard className="h-4 w-4 transition-transform group-hover:scale-110" />
            </Button>
         </div>
      </div>

      <div className="relative group">
        <div className="absolute inset-0 premium-gradient rounded-3xl blur-2xl opacity-0 group-hover:opacity-10 transition-opacity duration-1000" />
        <textarea
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Enter job description, organizational requirements, or role benefits..."
          disabled={isAnalyzing}
          className="w-full h-[400px] glass-card rounded-3xl p-8 text-base leading-relaxed bg-foreground/[0.02] border-border focus:bg-white/[0.04] focus:border-primary/20 focus:ring-1 focus:ring-primary/20 outline-none transition-all scrollbar-hide resize-none font-medium placeholder:text-muted-foreground/30"
        />
        
        <div className="absolute bottom-6 right-6 flex items-center gap-4">
           <div className="px-3 py-1.5 rounded-full bg-foreground/5 border border-border text-[10px] font-black tracking-widest text-muted-foreground uppercase">
              {inputText.split(/\s+/).filter(Boolean).length} Words Detectable
           </div>
           
           <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <Button 
                onClick={onAnalyze}
                disabled={isAnalyzing || inputText.length < 50}
                className="rounded-2xl premium-gradient px-8 py-6 h-auto shadow-2xl shadow-primary/20 font-bold group"
              >
                {isAnalyzing ? (
                   <span className="flex items-center gap-2">
                     <span className="h-4 w-4 border-2 border-foreground/20 border-t-white rounded-full animate-spin" />
                     Processing...
                   </span>
                ) : (
                  <span className="flex items-center gap-2">
                    Execute Scan <Sparkles className="h-4 w-4 group-hover:rotate-12 transition-transform" />
                  </span>
                )}
              </Button>
           </motion.div>
        </div>
      </div>
    </div>
  );
}
