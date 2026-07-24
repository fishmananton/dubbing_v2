"use client"

import { useSearchParams, useRouter } from "next/navigation"
import { Zap, CheckCircle2, XCircle, ArrowRight } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Suspense } from "react"

function VerifyEmailContent() {
  const searchParams = useSearchParams()
  const router = useRouter()
  const status = searchParams.get("status")
  const success = status === "success"

  return (
    <div className="min-h-screen flex">
      {/* Left side - Branding */}
      <div className="hidden lg:flex lg:w-1/2 bg-gradient-to-br from-primary to-brand-end items-center justify-center p-12">
        <div className="max-w-md text-center">
          <div className="flex items-center justify-center mb-8">
            <img src="/logo.png" alt="verbox.ai" className="h-12 w-12 rounded-xl object-contain" />
          </div>
          <h1 className="text-2xl font-semibold text-primary-foreground mb-4">
            Translate and dub your videos with AI
          </h1>
          <p className="text-primary-foreground/80">
            Reach global audiences with professional-quality translations in over 15 languages.
            Powered by cutting-edge AI for natural-sounding results.
          </p>
        </div>
      </div>

      {/* Right side */}
      <div className="flex-1 flex items-center justify-center p-8 bg-background">
        <div className="w-full max-w-md text-center">

          {/* Mobile logo */}
          <div className="lg:hidden flex items-center justify-center mb-8">
            <img src="/logo.png" alt="verbox.ai" className="h-8 w-8 rounded-lg object-contain" />
          </div>

          {success ? (
            <>
              <div className="mx-auto mb-6 flex h-20 w-20 items-center justify-center rounded-full bg-success/10">
                <CheckCircle2 className="h-10 w-10 text-success" />
              </div>
              <h2 className="text-2xl font-bold text-foreground mb-2">Email verified!</h2>
              <p className="text-sm text-muted-foreground mb-8">
                Your email has been confirmed. You can now sign in to your account.
              </p>
              <Button className="w-full h-11 gap-2" onClick={() => router.push("/login")}>
                Go to Sign In
                <ArrowRight className="h-4 w-4" />
              </Button>
            </>
          ) : (
            <>
              <div className="mx-auto mb-6 flex h-20 w-20 items-center justify-center rounded-full bg-destructive/10">
                <XCircle className="h-10 w-10 text-destructive" />
              </div>
              <h2 className="text-2xl font-bold text-foreground mb-2">Verification failed</h2>
              <p className="text-sm text-muted-foreground mb-8">
                The verification link is invalid or has expired. Request a new one from the sign-in page.
              </p>
              <Button className="w-full h-11 gap-2" onClick={() => router.push("/login")}>
                Back to Sign In
                <ArrowRight className="h-4 w-4" />
              </Button>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

export default function VerifyEmailPage() {
  return (
    <Suspense>
      <VerifyEmailContent />
    </Suspense>
  )
}
