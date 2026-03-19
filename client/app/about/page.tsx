import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Info, Github, BookOpen, Database, Shield, Globe, Star, Users, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function AboutPage() {
  return (
    <div className="space-y-16 pb-16">
      <div className="relative text-center max-w-3xl mx-auto pt-10">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-48 h-48 bg-primary/20 rounded-full blur-[100px] -z-10" />
        <h1 className="text-5xl font-black tracking-tight font-outfit text-gradient mb-4">JobGuard Intelligence</h1>
        <p className="text-xl text-muted-foreground font-medium leading-relaxed">
           Decentralizing trust in recruitment through adversarial neural mapping and linguistic fingerprinting.
        </p>
      </div>

      <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-3">
          <Card className="glass-card border-none shadow-2xl p-4 group hover:-translate-y-2 transition-transform duration-500">
             <div className="p-4 bg-indigo-500/10 rounded-3xl w-fit mb-6 ring-1 ring-indigo-500/20 group-hover:scale-110 transition-transform">
                <Shield className="h-8 w-8 text-indigo-500" />
             </div>
             <h3 className="text-2xl font-bold font-outfit mb-4">Core Mission</h3>
             <p className="text-muted-foreground leading-relaxed font-medium">
                To neutralize the threat of digital employment fraud by providing job seekers with military-grade linguistic analysis tools. 
                We believe security should be localized and accessible.
             </p>
          </Card>

          <Card className="glass-card border-none shadow-2xl p-4 group hover:-translate-y-2 transition-transform duration-500">
             <div className="p-4 bg-cyan-500/10 rounded-3xl w-fit mb-6 ring-1 ring-cyan-500/20 group-hover:scale-110 transition-transform">
                <Database className="h-8 w-8 text-cyan-500" />
             </div>
             <h3 className="text-2xl font-bold font-outfit mb-4">Dataset Corpus</h3>
             <p className="text-muted-foreground leading-relaxed font-medium">
                Our model is refined on the <strong>EMSCAD</strong> corpus—a gold standard dataset of 17,880 recruitment advertisements 
                meticulously labeled by cybersecurity experts.
             </p>
             <div className="mt-6 flex flex-wrap gap-2">
                <Badge variant="outline" className="border-white/10 px-3">17.8k nodes</Badge>
                <Badge variant="outline" className="border-white/10 px-3">Fraud: 32%</Badge>
             </div>
          </Card>

          <Card className="premium-gradient border-none shadow-2xl p-8 relative overflow-hidden group">
             <div className="absolute top-0 right-0 w-32 h-32 bg-white/20 rounded-full blur-3xl -mr-16 -mt-16 group-hover:scale-150 transition-transform duration-700" />
             <div className="relative z-10 text-white flex flex-col h-full">
                <Star className="h-10 w-10 text-white/50 mb-6" />
                <h3 className="text-2xl font-bold font-outfit mb-4">Community Driven</h3>
                <p className="text-white/80 leading-relaxed font-medium mb-8">
                   Contribute to the security engine or deploy your own local sidecar. Star us on GitHub.
                </p>
                <div className="mt-auto">
                    <Button variant="secondary" className="w-full rounded-2xl h-12 font-bold shadow-2xl">
                        <Github className="mr-2 h-5 w-5" /> GitHub Repository
                    </Button>
                </div>
             </div>
          </Card>
      </div>

      <div className="space-y-8">
        <h2 className="text-3xl font-bold font-outfit text-center">Inference Stack</h2>
        <div className="grid gap-4 md:grid-cols-4">
           {[
             { name: "Scikit Learning", icon: Globe, status: "Active" },
             { name: "TensorFlow 2.x", icon: Zap, status: "Active" },
             { name: "FastAPI 0.109", icon: Users, status: "Operational" },
             { name: "Next.js 15", icon: Info, status: "Latest" },
           ].map((stack, i) => (
             <div key={i} className="glass-card p-6 rounded-2xl flex items-center justify-between group">
                <div className="flex items-center gap-3">
                   <stack.icon className="h-5 w-5 text-primary group-hover:rotate-12 transition-transform" />
                   <span className="font-bold text-sm tracking-tight">{stack.name}</span>
                </div>
                <div className="h-1.5 w-1.5 rounded-full bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.6)]" />
             </div>
           ))}
        </div>
      </div>
    </div>
  );
}
