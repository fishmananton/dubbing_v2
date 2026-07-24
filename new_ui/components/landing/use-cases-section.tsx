"use client"

import { Card } from "@/components/ui/card"
import { Users, BookOpen, TrendingUp, Zap } from "lucide-react"

export function UseCasesSection() {
  const useCases = [
    {
      icon: Users,
      title: "Content Creators",
      description: "Expand to international audiences. YouTube, TikTok, Twitch—all platforms supported.",
    },
    {
      icon: BookOpen,
      title: "Educators",
      description: "Make learning accessible globally. Courses, lectures, and training materials in any language.",
    },
    {
      icon: TrendingUp,
      title: "Businesses",
      description: "Localize marketing, sales, and training content for international markets.",
    },
    {
      icon: Zap,
      title: "Agencies",
      description: "Scale video production for multiple clients across different regions efficiently.",
    },
  ]

  return (
    <section className="py-20 px-4 bg-zinc-950">
      <div className="max-w-6xl mx-auto">
        <div className="text-center space-y-4 mb-16">
          <h2 className="text-4xl md:text-5xl font-bold text-muted-foreground">
            Built for Every Use Case
          </h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Whether you're a solo creator or enterprise, we have the tools you need.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          {useCases.map((useCase, idx) => {
            const Icon = useCase.icon
            return (
              <Card key={idx} className="p-8 border border-border/50 bg-background/50 backdrop-blur-sm hover:border-primary/50 transition-colors">
                <div className="space-y-4">
                  <div className="w-12 h-12 rounded-lg flex items-center justify-center" style={{background: "rgba(114,71,237,0.1)"}}>
                    <Icon className="w-6 h-6" style={{color: "#7247ED"}} />
                  </div>
                  <h3 className="text-xl font-semibold text-foreground">{useCase.title}</h3>
                  <p className="text-zinc-300">{useCase.description}</p>
                </div>
              </Card>
            )
          })}
        </div>
      </div>
    </section>
  )
}
