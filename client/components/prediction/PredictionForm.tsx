'use client';

import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import * as z from "zod";
import { Button } from "@/components/ui/button";
import {
  Field,
  FieldContent,
  FieldDescription,
  FieldLabel,
  FieldError,
} from "@/components/ui/field";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Loader2, Search, Sparkles, Building2, MapPin, Briefcase } from "lucide-react";
import { motion } from "framer-motion";

const formSchema = z.object({
  title: z.string().min(2, {
    message: "Job title must be at least 2 characters.",
  }),
  location: z.string().optional(),
  department: z.string().optional(),
  salary_range: z.string().optional(),
  description: z.string().min(10, {
    message: "Job description must be at least 10 characters.",
  }),
  requirements: z.string().optional(),
  benefits: z.string().optional(),
});

type FormValues = z.infer<typeof formSchema>;

interface PredictionFormProps {
  onPredict: (data: FormValues) => void;
  isLoading: boolean;
}

export default function PredictionForm({ onPredict, isLoading }: PredictionFormProps) {
  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<FormValues>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      title: "",
      location: "",
      department: "",
      salary_range: "",
      description: "",
      requirements: "",
      benefits: "",
    },
  });

  const onSubmit = (values: FormValues) => {
    onPredict(values);
  };

  return (
    <Card className="glass-card border-none shadow-2xl relative overflow-hidden group">
      <div className="absolute inset-0 bg-gradient-to-br from-primary/5 to-transparent pointer-events-none" />
      <CardHeader className="relative z-10">
        <div className="flex items-center gap-3 mb-2">
            <div className="p-2 premium-gradient rounded-lg shadow-lg">
                <Briefcase className="h-5 w-5 text-white" />
            </div>
            <CardTitle>Job Post Details</CardTitle>
        </div>
        <CardDescription>Enter as much information as possible for a more accurate prediction.</CardDescription>
      </CardHeader>
      <CardContent className="relative z-10">
        <form onSubmit={handleSubmit(onSubmit)} className="space-y-8">
          <div className="grid gap-6 md:grid-cols-2">
            <Field className={errors.title ? "data-[invalid=true]" : ""}>
              <FieldLabel htmlFor="title" className="flex items-center gap-2"><Sparkles className="h-3 w-3 text-primary" /> Job Title</FieldLabel>
              <FieldContent>
                <div className="relative">
                   <Input 
                    id="title" 
                    placeholder="e.g. Software Engineer" 
                    {...register("title")} 
                    aria-invalid={!!errors.title}
                    className="rounded-xl border-white/10 bg-white/5 pl-9 h-11 focus-visible:ring-primary/50"
                  />
                  <Building2 className="absolute left-3 top-3.5 h-4 w-4 text-muted-foreground" />
                </div>
              </FieldContent>
              <FieldError errors={[errors.title]} />
            </Field>

            <Field className={errors.location ? "data-[invalid=true]" : ""}>
              <FieldLabel htmlFor="location" className="flex items-center gap-2"><MapPin className="h-3 w-3 text-cyan-500" /> Location</FieldLabel>
              <FieldContent>
                 <div className="relative">
                    <Input 
                      id="location" 
                      placeholder="e.g. New York, NY" 
                      {...register("location")} 
                      aria-invalid={!!errors.location}
                      className="rounded-xl border-white/10 bg-white/5 pl-9 h-11 focus-visible:ring-primary/50"
                    />
                    <MapPin className="absolute left-3 top-3.5 h-4 w-4 text-muted-foreground" />
                 </div>
              </FieldContent>
              <FieldError errors={[errors.location]} />
            </Field>
          </div>

          <Field className={errors.description ? "data-[invalid=true]" : ""}>
            <FieldLabel htmlFor="description">Job Description</FieldLabel>
            <FieldContent>
              <Textarea 
                id="description"
                placeholder="Paste the full job description here..." 
                className="min-h-[200px] rounded-2xl border-white/10 bg-white/5 focus-visible:ring-primary/50 p-4 leading-relaxed"
                {...register("description")} 
                aria-invalid={!!errors.description}
              />
            </FieldContent>
            <FieldDescription className="text-[10px] uppercase font-bold tracking-widest text-primary/70 mt-2">
              Neural input: Higher word count improves model precision.
            </FieldDescription>
            <FieldError errors={[errors.description]} />
          </Field>

          <Button type="submit" className="w-full h-12 rounded-2xl premium-gradient font-bold text-lg shadow-xl shadow-primary/20 hover:scale-[1.02] active:scale-[0.98] transition-all" disabled={isLoading}>
            {isLoading ? (
              <>
                <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                Processing Patterns...
              </>
            ) : (
              <>
                <Search className="mr-2 h-5 w-5" />
                Execute Neural Scan
              </>
            )}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}
