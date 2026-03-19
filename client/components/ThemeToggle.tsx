'use client';

import * as React from 'react';
import { Moon, Sun } from 'lucide-react';
import { useTheme } from 'next-themes';
import { motion, AnimatePresence } from 'framer-motion';

export function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = React.useState(false);

  React.useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) return <div className="h-10 w-10 min-w-10 rounded-xl bg-foreground/5 animate-pulse" />;

  const isDark = theme === 'dark' || (theme === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches);

  return (
    <button
      onClick={() => setTheme(isDark ? 'light' : 'dark')}
      className="relative h-12 w-full flex items-center gap-3 px-4 rounded-2xl bg-foreground/[0.03] border border-border hover:bg-foreground/[0.08] transition-all group overflow-hidden"
      aria-label="Toggle theme"
    >
      <div className="flex items-center justify-center h-6 w-6 rounded-lg bg-primary/10 text-primary group-hover:scale-110 transition-transform">
        <AnimatePresence mode="wait">
          {isDark ? (
            <motion.div
              key="moon"
              initial={{ rotate: -90, opacity: 0 }}
              animate={{ rotate: 0, opacity: 1 }}
              exit={{ rotate: 90, opacity: 0 }}
              transition={{ duration: 0.2 }}
            >
              <Moon className="h-4 w-4" />
            </motion.div>
          ) : (
            <motion.div
              key="sun"
              initial={{ rotate: -90, opacity: 0 }}
              animate={{ rotate: 0, opacity: 1 }}
              exit={{ rotate: 90, opacity: 0 }}
              transition={{ duration: 0.2 }}
            >
              <Sun className="h-4 w-4" />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
      
      <span className="text-[10px] font-black uppercase tracking-widest text-muted-foreground group-hover:text-foreground transition-colors">
        {isDark ? 'System: Dark' : 'System: Light'}
      </span>

      <div className="absolute right-4 h-1.5 w-1.5 rounded-full bg-primary shadow-[0_0_8px_rgba(124,58,237,0.5)]" />
    </button>
  );
}
