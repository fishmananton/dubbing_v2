import type { Metadata } from "next"
import { HeroSection } from "@/components/landing/hero-section"
import { FeaturesSection } from "@/components/landing/features-section"
import { UseCasesSection } from "@/components/landing/use-cases-section"
import { TrialSection } from "@/components/landing/trial-section"
import { AuthRedirect } from "@/components/landing/auth-redirect"

export const metadata: Metadata = {
  title: "verbox.ai — AI Video Translation & Dubbing",
  description:
    "Professional AI-powered video translation and dubbing. Reach global audiences in different languages. Fast, high-quality, easy to use.",
  openGraph: {
    title: "verbox.ai — AI Video Translation & Dubbing",
    description:
      "Professional AI-powered video translation and dubbing. Reach global audiences in different languages.",
    url: "https://verbox.ai",
    siteName: "verbox.ai",
    images: [{ url: "https://verbox.ai/og-image.jpg", width: 1200, height: 630 }],
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "verbox.ai — AI Video Translation & Dubbing",
    description: "Professional AI-powered video translation and dubbing. Reach global audiences in different languages.",
    images: ["https://verbox.ai/og-image.jpg"],
  },
}

export default function LandingPage() {
  return (
    <main className="min-h-screen bg-background">
      <AuthRedirect />
      <HeroSection />
      <FeaturesSection />
      <UseCasesSection />
      <TrialSection />
    </main>
  )
}
