'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { cn } from '@/lib/utils';
import { 
  Terminal, 
  Scan, 
  FlaskConical, 
  Activity, 
  Database,
  ShieldAlert,
  ChevronRight,
  LayoutGrid,
  FileText
} from 'lucide-react';
import { motion } from 'framer-motion';

const navigation = [
  { name: 'Intelligence Console', href: '/console', icon: Terminal },
  { name: 'Real-time Scan', href: '/scan', icon: Scan },
  { name: 'Model Lab', href: '/lab', icon: FlaskConical },
  { name: 'System Monitor', href: '/monitor', icon: Activity },
  { name: 'Dataset Explorer', href: '/dataset', icon: Database },
  { name: 'Export Reports', href: '/report', icon: FileText },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <div className="flex h-full w-72 flex-col glass-sidebar m-4 rounded-[2rem] overflow-hidden shadow-[0_0_50px_rgba(0,0,0,0.5)] border border-white/5 bg-[#080808]/90">
      <div className="flex h-24 shrink-0 items-center px-8 border-b border-white/5 bg-white/[0.02]">
        <motion.div 
          whileHover={{ rotate: 5, scale: 1.1 }}
          className="p-3 premium-gradient rounded-2xl shadow-[0_0_20px_rgba(124,58,237,0.4)] ring-1 ring-white/30"
        >
          <ShieldAlert className="h-6 w-6 text-white" />
        </motion.div>
        <span className="ml-4 text-2xl font-black tracking-tight text-white italic drop-shadow-md">JobGuard</span>
      </div>
      
      <nav className="flex-1 space-y-2 px-4 py-10 overflow-y-auto scrollbar-hide">
        {navigation.map((item) => {
          const isActive = pathname === item.href;
          return (
            <Link
              key={item.name}
              href={item.href}
              className="relative group block"
            >
              <div
                className={cn(
                  'flex items-center rounded-2xl px-5 py-4 text-[10px] font-black uppercase tracking-[0.2em] transition-all duration-300 relative z-10',
                  isActive
                    ? 'text-white'
                    : 'text-zinc-500 hover:text-white'
                )}
              >
                {isActive && (
                  <motion.div
                    layoutId="sidebar-active"
                    className="absolute inset-0 premium-gradient rounded-2xl -z-10 shadow-[0_0_30px_rgba(124,58,237,0.3)]"
                    transition={{ type: 'spring', bounce: 0.2, duration: 0.6 }}
                  />
                )}
                
                <item.icon
                  className={cn(
                    'mr-4 h-5 w-5 shrink-0 transition-transform duration-300 group-hover:scale-110',
                    isActive ? 'text-white' : 'text-zinc-500 group-hover:text-primary'
                  )}
                />
                <span className="flex-1">{item.name}</span>
                {isActive && <motion.div layoutId="pulse" className="h-1.5 w-1.5 rounded-full bg-white shadow-[0_0_8px_white] animate-pulse" />}
              </div>
            </Link>
          );
        })}
      </nav>

      <div className="p-8 border-t border-white/5 bg-white/[0.01]">
        <div className="rounded-3xl bg-white/[0.03] backdrop-blur-md p-5 border border-white/5 overflow-hidden relative group">
          <div className="absolute inset-0 bg-gradient-to-br from-primary/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
          <p className="text-[9px] font-black text-primary uppercase tracking-[0.3em] mb-2 relative z-10">Neural Hub</p>
          <div className="flex items-center justify-between relative z-10">
            <div className="flex items-center gap-2">
              <div className="h-2 w-2 rounded-full bg-green-500 shadow-[0_0_10px_#22c55e] animate-pulse" />
              <span className="text-[10px] font-black text-zinc-400 uppercase">Integrity 100%</span>
            </div>
            <Activity className="h-3 w-3 text-zinc-600" />
          </div>
        </div>
      </div>
    </div>
  );
}

