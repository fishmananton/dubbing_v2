"use client"

import { Button } from "@/components/ui/button"
import Link from "next/link"

export function TrialSection() {
  return (
    <section className="py-20 px-4 bg-background">
      <div className="max-w-4xl mx-auto">
        <div className="rounded-2xl border border-border bg-gradient-to-br from-background to-zinc-950 p-12 text-center space-y-8">
          <div className="space-y-4">
            <h2 className="text-4xl md:text-5xl font-bold text-foreground">
              Ready to Go Global?
            </h2>
            <p className="text-lg text-zinc-300 max-w-2xl mx-auto">
              Start with $5 in trial credits. No credit card required. Get instant access to professional-grade video translation.
            </p>
          </div>

          <div className="flex flex-col sm:flex-row gap-4 justify-center pt-4">
            <Link href="/app">
              <Button size="lg" className="text-base px-8 h-12 w-full sm:w-auto">
                Get $5 Trial Credit
              </Button>
            </Link>
          </div>

          <div className="pt-4 text-sm text-zinc-300">
            ✓ Instant activation • ✓ No card required • ✓ Full feature access
          </div>
        </div>
      </div>
    </section>
  )
}
