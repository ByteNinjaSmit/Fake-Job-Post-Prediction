import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { CheckCircle2, AlertTriangle, ShieldCheck, ShieldAlert, ArrowRight, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { motion } from "framer-motion";

interface ResultDisplayProps {
  result: {
    is_fake: boolean;
    confidence: number;
    explanation?: string;
  };
}

export default function ResultDisplay({ result }: ResultDisplayProps) {
  const isFake = result.is_fake;
  const confidencePercent = Math.round(result.confidence * 100);

  return (
    <motion.div
      initial={{ scale: 0.95, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className="w-full"
    >
      <Card className={`glass-card border-none shadow-2xl relative overflow-hidden`}>
        <div className={`absolute inset-0 bg-gradient-to-br ${isFake ? 'from-destructive/10 to-transparent' : 'from-green-500/10 to-transparent'} pointer-events-none`} />
        
        <CardHeader className="relative z-10">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-4">
              <div className={`p-4 rounded-2xl ${isFake ? 'bg-destructive/10 text-destructive' : 'bg-green-500/10 text-green-500'}`}>
                {isFake ? <ShieldAlert className="h-8 w-8" /> : <ShieldCheck className="h-8 w-8" />}
              </div>
              <div>
                <CardTitle className="text-2xl font-extrabold font-outfit">
                  {isFake ? "Insecure Content" : "Legitimate Post"}
                </CardTitle>
                <CardDescription className="text-base">
                  Analysis complete with {confidencePercent}% confidence.
                </CardDescription>
              </div>
            </div>
            <Badge className={`${isFake ? 'bg-destructive text-destructive-foreground' : 'bg-green-500 text-white'} text-xl px-6 py-2 rounded-2xl shadow-xl font-bold border-none`}>
               {isFake ? "FAKE" : "REAL"}
            </Badge>
          </div>
        </CardHeader>
        
        <CardContent className="relative z-10 space-y-8 pt-4">
          <div className="space-y-3">
            <div className="flex justify-between text-sm font-bold uppercase tracking-widest text-muted-foreground">
              <span>Security Confidence</span>
              <span className={isFake ? 'text-destructive' : 'text-green-500'}>{confidencePercent}%</span>
            </div>
            <Progress 
              value={confidencePercent} 
              className={`h-3 rounded-full bg-foreground/5 ${isFake ? '[&>div]:bg-destructive' : '[&>div]:bg-green-500'}`} 
            />
          </div>

          <div className="rounded-2xl bg-foreground/5 p-6 border border-border space-y-4">
            <h4 className="font-bold font-outfit text-lg flex items-center gap-2">
              {isFake ? <AlertTriangle className="h-5 w-5 text-destructive" /> : <CheckCircle2 className="h-5 w-5 text-green-500" />}
              Linguistic Evidence
            </h4>
            <p className="text-muted-foreground leading-relaxed italic">
              "{result.explanation || (isFake 
                ? "Neural patterns detected a high concentration of deceptive linguistic traits, including artificial urgency and non-standard salary-to-role ratios." 
                : "The job posting exhibits consistent professional syntax and follows standard organizational announcement structures.")}"
            </p>
          </div>

          <div className="flex flex-col sm:flex-row gap-4">
            <Button variant="outline" className="flex-1 h-12 rounded-2xl glass-card font-bold text-base" asChild>
              <Link href="/explain">
                 Pulse Analysis <Zap className="ml-2 h-4 w-4 fill-primary text-primary" />
              </Link>
            </Button>
            <Button className="flex-1 h-12 rounded-2xl premium-gradient font-bold text-base shadow-lg shadow-primary/20 transition-transform active:scale-95 text-white" onClick={() => window.location.reload()}>
              Reset Engine
            </Button>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}
