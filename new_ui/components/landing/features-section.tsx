"use client"

import { Card } from "@/components/ui/card"
import { Check } from "lucide-react"

export function FeaturesSection() {
  const features = [
    {
      title: "Lightning Fast",
      description: "Translate and dub videos in minutes, not hours. Process multiple languages simultaneously.",
    },
    {
      title: "Crystal Clear Quality",
      description: "Natural-sounding AI voices that match the original tone and emotion of your content.",
    },
    {
      title: "Easy to Use",
      description: "Upload, configure, and publish. No technical expertise required.",
    },
    {
      title: "Major Languages",
      description: "Support for all major languages and continuous expansion of language support.",
    },
    {
      title: "Multi-speaker Ready",
      description: "Automatically detect and preserve multiple speakers with distinct voice characteristics.",
    },
    {
      title: "Global Reach",
      description: "Connect with audiences worldwide regardless of their language or location.",
    },
  ]

  return (
    <section className="py-20 px-4 bg-background">
      <div className="max-w-6xl mx-auto">
        <div className="text-center space-y-4 mb-16">
          <h2 className="text-4xl md:text-5xl font-bold text-foreground">
            Why creators love verbox.ai
          </h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Everything you need to reach global audiences with professional-quality translations.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, idx) => (
            <Card key={idx} className="p-6 border border-border hover:border-primary/50 transition-colors">
              <div className="space-y-3">
                <div className="w-10 h-10 rounded-lg flex items-center justify-center" style={{background: "rgba(114,71,237,0.1)"}}>
                  <Check className="w-5 h-5" style={{color: "#7247ED"}} />
                </div>
                <h3 className="text-lg font-semibold text-foreground">{feature.title}</h3>
                <p className="text-muted-foreground">{feature.description}</p>
              </div>
            </Card>
          ))}
        </div>
      </div>
    </section>
  )
}
