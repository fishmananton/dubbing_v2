"use client"

import { Button } from "@/components/ui/button"
import Link from "next/link"
import Image from "next/image"

export function HeroSection() {
  return (
    <section className="relative min-h-screen flex flex-col items-center justify-center px-4 py-20 bg-gradient-to-br from-background via-background to-zinc-950">
      {/* Background decoration */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-primary/5 rounded-full blur-3xl" />
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-primary/5 rounded-full blur-3xl" />
      </div>

      <div className="relative z-10 max-w-4xl mx-auto text-center space-y-8">
        {/* Logo */}
        <div className="flex flex-col items-center mb-6 gap-3">
          <div className="relative w-24 h-24">
            <Image
              src="/logo.png"
              alt="verbox.ai logo"
              width={96}
              height={96}
              className="w-full h-full object-contain"
            />
          </div>
          <span className="text-2xl font-bold tracking-tight text-foreground">verbox.ai</span>
        </div>

        {/* Headline */}
        <div className="space-y-4">
          <h1 className="text-5xl md:text-6xl lg:text-7xl font-bold tracking-tight text-foreground">
            Translate Your Videos{" "}
            <span className="block text-transparent bg-clip-text bg-gradient-to-r from-primary to-brand-end">
              In Minutes
            </span>
          </h1>
          
          <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto leading-relaxed">
            Professional AI-powered video translation and dubbing. Reach global audiences with crystal-clear audio in major languages.
          </p>
        </div>

        {/* CTA Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 justify-center pt-4">
          <Link href="/app">
            <Button size="lg" className="text-base px-8 h-12">
              Start Translating
            </Button>
          </Link>
        </div>

        {/* Trial Credit Badge */}
        <div className="pt-4">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-border/50 bg-background/50 backdrop-blur-sm">
            <span className="text-sm text-muted-foreground">
              ✨ New users get <span className="font-semibold text-foreground">$5 trial credit</span>
            </span>
          </div>
        </div>
      </div>
    </section>
  )
}
