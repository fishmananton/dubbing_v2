"use client"

import { useState, Suspense } from "react"
import { useSearchParams } from "next/navigation"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { 
  ArrowLeft, 
  Search, 
  Upload, 
  Languages, 
  Play, 
  Download, 
  Settings, 
  CreditCard,
  MessageCircle,
  Mail,
  ChevronDown,
  ChevronUp,
  Zap
} from "lucide-react"
import { cn } from "@/lib/utils"

interface FAQItem {
  question: string
  answer: string
}

const FAQ_CATEGORIES = [
  {
    title: "Getting Started",
    icon: Upload,
    items: [
      {
        question: "How do I create a new project?",
        answer: "Click the 'New Project' button in the header, enter a name for your project, and you're ready to upload your video. You can have multiple projects to organize different videos or clients."
      },
      {
        question: "What video formats are supported?",
        answer: "verbox.ai supports most common video formats including MP4, MOV, AVI, WebM, and MKV. For best results, we recommend using MP4 with H.264 encoding. Maximum file size is 2GB."
      },
      {
        question: "How long does processing take?",
        answer: "Processing time depends on video length and complexity. A typical 10-minute video takes 3-5 minutes to process. Longer videos with multiple speakers may take longer."
      }
    ]
  },
  {
    title: "Translation & Dubbing",
    icon: Languages,
    items: [
      {
        question: "Which languages are supported?",
        answer: "We support 15 languages including Spanish, French, German, Portuguese, Italian, Japanese, Korean, Chinese, Arabic, Hindi, and many more. New languages are added regularly."
      },
      {
        question: "What's the difference between Overlay and New Voice modes?",
        answer: "Overlay mode keeps the original audio at a lower volume while adding the translated voice on top. New Voice mode completely replaces the original audio with the translated version."
      },
      {
        question: "How accurate is the translation?",
        answer: "Our AI translation is highly accurate and context-aware. However, we recommend reviewing subtitles before final export, especially for specialized content. You can edit any subtitle directly in the editor."
      }
    ]
  },
  {
    title: "Editing & Export",
    icon: Play,
    items: [
      {
        question: "Can I edit the translated subtitles?",
        answer: "Yes! Click on any subtitle in the editor to make changes. Your edits are saved automatically. After editing, use 'Save & Regenerate' to update the audio with your changes."
      },
      {
        question: "What export formats are available?",
        answer: "You can export your translated video in various resolutions (480p, 720p, 1080p, or Original quality). Audio-only export is also available in WAV format."
      }
    ]
  },
  {
    title: "Billing & Account",
    icon: CreditCard,
    items: [
      {
        question: "How does billing work?",
        answer: "verbox.ai uses a prepaid credit system. You add funds to your account and are charged per minute of video processed. Regeneration rates may vary depending on the selected voice mode."
      },
      {
        question: "What happens if I run out of credits?",
        answer: "Your projects are saved and you can add more funds at any time. Processing will pause until you have sufficient balance."
      },
      {
        question: "Can I get a refund?",
        answer: "We offer refunds for unused credits within 30 days of purchase. Contact support for assistance with refund requests."
      }
    ]
  }
]

function HelpPageInner() {
  const searchParams = useSearchParams()
  const backHref = searchParams.get("project") ? `/app?project=${searchParams.get("project")}` : "/app"
  const [searchQuery, setSearchQuery] = useState("")
  const [expandedItems, setExpandedItems] = useState<Set<string>>(new Set())

  const toggleItem = (id: string) => {
    const newExpanded = new Set(expandedItems)
    if (newExpanded.has(id)) {
      newExpanded.delete(id)
    } else {
      newExpanded.add(id)
    }
    setExpandedItems(newExpanded)
  }

  const filteredCategories = FAQ_CATEGORIES.map(category => ({
    ...category,
    items: category.items.filter(
      item => 
        item.question.toLowerCase().includes(searchQuery.toLowerCase()) ||
        item.answer.toLowerCase().includes(searchQuery.toLowerCase())
    )
  })).filter(category => category.items.length > 0)

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-border bg-card shadow-sm">
        <div className="mx-auto max-w-4xl px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link href={backHref}>
              <Button variant="ghost" size="sm" className="gap-2">
                <ArrowLeft className="h-4 w-4" />
                Back to App
              </Button>
            </Link>
            <div className="flex items-center gap-2">
              <img src="/logo.png" alt="verbox.ai" className="h-8 w-8 rounded-lg object-contain" />
              <span className="text-base font-semibold text-foreground">verbox.ai Help</span>
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-4xl px-4 py-8">
        {/* Hero Section */}
        <div className="text-center mb-10">
          <h1 className="text-3xl font-bold text-foreground mb-3">How can we help you?</h1>
          <p className="text-muted-foreground mb-6">Search our help center or browse categories below</p>
          
          {/* Search */}
          <div className="relative max-w-lg mx-auto">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search for help..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10"
            />
          </div>
        </div>

        {/* FAQ Categories */}
        <div className="space-y-8">
          {filteredCategories.map((category) => (
            <div key={category.title} className="rounded-xl border border-border bg-card p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                  <category.icon className="h-5 w-5 text-primary" />
                </div>
                <h2 className="text-lg font-semibold text-foreground">{category.title}</h2>
              </div>
              
              <div className="space-y-2">
                {category.items.map((item, idx) => {
                  const itemId = `${category.title}-${idx}`
                  const isExpanded = expandedItems.has(itemId)
                  
                  return (
                    <div key={idx} className="border-b border-border last:border-b-0">
                      <button
                        onClick={() => toggleItem(itemId)}
                        className="w-full flex items-center justify-between py-3 text-left hover:text-primary transition-colors"
                      >
                        <span className="text-sm font-medium text-foreground pr-4">{item.question}</span>
                        {isExpanded ? (
                          <ChevronUp className="h-4 w-4 text-muted-foreground shrink-0" />
                        ) : (
                          <ChevronDown className="h-4 w-4 text-muted-foreground shrink-0" />
                        )}
                      </button>
                      {isExpanded && (
                        <div className="pb-3">
                          <p className="text-sm text-muted-foreground leading-relaxed">{item.answer}</p>
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            </div>
          ))}
        </div>

        {/* Contact Section */}
        <div className="mt-12 rounded-xl border border-border bg-card p-8 text-center">
          <h2 className="text-xl font-semibold text-foreground mb-2">Still need help?</h2>
          <p className="text-muted-foreground mb-6">Our support team is here to assist you.</p>
          
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <a href="mailto:anton@verbox.ai?subject=Support Request&body=Describe your issue...">
              <Button className="gap-2 w-full sm:w-auto">
                <Mail className="h-4 w-4" />
                Email Support
              </Button>
            </a>
          </div>
          
          <p className="text-xs text-muted-foreground mt-4">
            Average response time: Less than 2 hours
          </p>
        </div>
      </main>
    </div>
  )
}

export default function HelpPage() {
  return (
    <Suspense fallback={null}>
      <HelpPageInner />
    </Suspense>
  )
}
