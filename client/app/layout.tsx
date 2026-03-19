import type { Metadata } from "next";
import { Geist, Geist_Mono, Outfit, Inter } from "next/font/google";
import "./globals.css";
import { cn } from "@/lib/utils";
import { Sidebar } from "@/components/Sidebar";

const outfit = Outfit({ subsets: ["latin"], variable: "--font-outfit" });
const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "JobGuard AI | Premium ML Protection",
  description: "Experience the next generation of fraudulent job post detection.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={cn(
        "h-full", 
        "antialiased", 
        outfit.variable, 
        inter.variable, 
        geistSans.variable, 
        geistMono.variable
      )}
    >
      <body className="h-full bg-background text-foreground font-sans selection:bg-primary/20 selection:text-primary overflow-hidden">
        <div className="flex h-full relative">
          {/* Animated Background Mesh */}
          <div className="absolute inset-0 -z-10 pointer-events-none">
            <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-primary/10 rounded-full blur-[120px] animate-pulse" />
            <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-cyan-500/10 rounded-full blur-[120px] animate-pulse [animation-delay:2s]" />
          </div>

          <Sidebar />
          
          <main className="flex-1 overflow-y-auto p-4 md:p-8 lg:p-12 relative">
            <div className="max-w-7xl mx-auto min-h-full">
              {children}
            </div>
          </main>
        </div>
      </body>
    </html>
  );
}
