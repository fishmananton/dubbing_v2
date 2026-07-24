"use client"

import { useState, useEffect, useRef, Suspense } from "react"
import { useSearchParams } from "next/navigation"
import { Turnstile } from "@marsidev/react-turnstile"

declare global {
  interface Window {
    google?: {
      accounts: {
        id: {
          initialize: (config: { client_id: string; callback: (response: { credential: string }) => void }) => void
          renderButton: (element: HTMLElement, config: { theme: string; size: string; width?: string; text?: string }) => void
        }
      }
    }
  }
}


import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog"
import { Mail, Lock, User, Eye, EyeOff, Loader2, ArrowRight } from "lucide-react"
import { cn } from "@/lib/utils"

type AuthMode = "login" | "signup" | "forgot"

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000"

function LoginContent() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const [mode, setMode] = useState<AuthMode>(
    () => (searchParams.get("mode") === "signup" ? "signup" : "login")
  )
  const [isLoading, setIsLoading] = useState(false)
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirmPassword, setShowConfirmPassword] = useState(false)
  const [error, setError] = useState("")
  const [success, setSuccess] = useState("")
  const [showResetSentDialog, setShowResetSentDialog] = useState(false)
  const [resetEmail, setResetEmail] = useState("")
  const [showSignUpSuccessDialog, setShowSignUpSuccessDialog] = useState(false)
  const [showPasswordChangedDialog, setShowPasswordChangedDialog] = useState(
    () => searchParams.get("reset") === "success"
  )

  // Form state
  const [email, setEmail] = useState("")
  const [showResendVerification, setShowResendVerification] = useState(false)
  const [resendingVerification, setResendingVerification] = useState(false)
  const [password, setPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")
  const [firstName, setFirstName] = useState("")
  const [lastName, setLastName] = useState("")
  const [username, setUsername] = useState("")
  const [turnstileToken, setTurnstileToken] = useState("")
  const turnstileRef = useRef<{ reset: () => void }>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError("")
    setSuccess("")

    if (mode === "signup" && password !== confirmPassword) {
      setError("Passwords do not match")
      return
    }

    if (mode === "signup" && password.length < 8) {
      setError("Password must be at least 8 characters")
      return
    }

    if (mode === "signup" && !turnstileToken) {
      setError("Please complete the CAPTCHA")
      return
    }

    setIsLoading(true)

    try {
      if (mode === "signup") {
        // Register
        const res = await fetch(`${API_BASE}/auth/register/local`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          credentials: "include",
          body: JSON.stringify({
            user_name: username,
            email,
            password,
            first_name: firstName,
            last_name: lastName,
            cf_turnstile_token: turnstileToken,
          }),
        })

        if (!res.ok) {
          const error = await res.json()
          turnstileRef.current?.reset()
          setTurnstileToken("")
          throw new Error(error.detail || "Registration failed")
        }

        setMode("login")
        resetForm()
        setShowSignUpSuccessDialog(true)
      } else if (mode === "forgot") {
        // Password reset request
        const res = await fetch(`${API_BASE}/auth/request-password-reset`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          credentials: "include",
          body: JSON.stringify({ email }),
        })

        if (!res.ok) {
          throw new Error("Failed to send reset email")
        }

        setResetEmail(email)
        setShowResetSentDialog(true)
        setMode("login")
        resetForm()
      } else {
        // Login
        const res = await fetch(`${API_BASE}/auth/login/local`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          credentials: "include",
          body: JSON.stringify({
            login: email,
            password,
          }),
        })

        if (!res.ok) {
          const error = await res.json()
          if (res.status === 403) {
            setError("Email not verified. Check your email for verification link.")
            setShowResendVerification(true)
          } else {
            throw new Error(error.detail || "Login failed")
          }
          setIsLoading(false)
          return
        }

        // Login successful, redirect
        router.push("/app")
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred")
    } finally {
      setIsLoading(false)
    }
  }

  const handleResendVerification = async () => {
    if (!email) return
    setResendingVerification(true)
    try {
      const res = await fetch(`${API_BASE}/auth/resend-email-verification`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ email }),
      })
      if (res.ok) {
        setSuccess("Verification email sent! Check your inbox.")
        setShowResendVerification(false)
      } else {
        const error = await res.json()
        setError(error.detail || "Failed to resend verification email")
      }
    } catch (e) {
      setError("Failed to resend verification email")
    } finally {
      setResendingVerification(false)
    }
  }

  const handleGoogleSignIn = async (credential: string) => {
    setError("")
    setIsLoading(true)
    try {
      const res = await fetch(`${API_BASE}/auth/google`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ credential }),
      })
      if (!res.ok) {
        const error = await res.json()
        throw new Error(error.detail || "Google sign-in failed")
      }
      router.push("/app")
    } catch (err) {
      setError(err instanceof Error ? err.message : "Google sign-in failed")
    } finally {
      setIsLoading(false)
    }
  }

  // Initialize Google Sign-In
  useEffect(() => {
    const clientId = process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID
    if (!clientId) return

    const script = document.createElement("script")
    script.src = "https://accounts.google.com/gsi/client"
    script.async = true
    script.defer = true
    script.onload = () => {
      if (window.google) {
        window.google.accounts.id.initialize({
          client_id: clientId,
          callback: (response: { credential: string }) => {
            handleGoogleSignIn(response.credential)
          },
        })
        // Render the button
        const buttonDiv = document.getElementById("google-signin-button")
        if (buttonDiv) {
          window.google.accounts.id.renderButton(buttonDiv, {
            theme: "outline",
            size: "large",
            text: "continue_with",
          })
        }
      }
    }
    document.body.appendChild(script)
    return () => {
      document.body.removeChild(script)
    }
  }, [])

  const resetForm = () => {
    setEmail("")
    setPassword("")
    setConfirmPassword("")
    setFirstName("")
    setLastName("")
    setUsername("")
    setError("")
    setSuccess("")
    setTurnstileToken("")
    turnstileRef.current?.reset()
  }

  return (
    <>
    <div className="min-h-screen flex">
      {/* Left side - Branding */}
      <div className="hidden lg:flex lg:w-1/2 bg-gradient-to-br from-primary to-brand-end items-center justify-center p-12">
        <div className="max-w-md text-center">
          <div className="flex flex-col items-center gap-3 mb-8">
            <img src="/logo.png" alt="verbox.ai" className="h-20 w-20 rounded-2xl object-contain" style={{filter: "brightness(0) invert(1)"}} />
            <span className="text-3xl font-bold text-white">verbox.ai</span>
          </div>
          <h1 className="text-2xl font-semibold text-white mb-4">
            Translate and dub your videos with AI
          </h1>
          <p className="text-white/90">
            Reach global audiences with professional-quality translations in over 15 languages.
            Powered by cutting-edge AI for natural-sounding results.
          </p>

          <div className="mt-12 grid grid-cols-3 gap-4 text-center">
            <div className="rounded-lg bg-primary-foreground/10 p-4">
              <p className="text-2xl font-bold text-white">15+</p>
              <p className="text-xs text-white/80">Languages</p>
            </div>
            <div className="rounded-lg bg-primary-foreground/10 p-4">
              <p className="text-2xl font-bold text-white">5M+</p>
              <p className="text-xs text-white/80">Videos Processed</p>
            </div>
            <div className="rounded-lg bg-primary-foreground/10 p-4">
              <p className="text-2xl font-bold text-white">99%</p>
              <p className="text-xs text-white/80">Accuracy</p>
            </div>
          </div>
        </div>
      </div>

      {/* Right side - Auth form */}
      <div className="flex-1 flex items-center justify-center p-8 bg-background">
        <div className="w-full max-w-md">
          {/* Mobile logo */}
          <div className="lg:hidden flex flex-col items-center gap-2 mb-8">
            <img src="/logo.png" alt="verbox.ai" className="h-14 w-14 rounded-xl object-contain" />
            <span className="text-xl font-bold text-foreground">verbox.ai</span>
          </div>

          <div className="text-center mb-8">
            <h2 className="text-2xl font-bold text-foreground">
              {mode === "login" && "Welcome back"}
              {mode === "signup" && "Create your account"}
              {mode === "forgot" && "Reset your password"}
            </h2>
            <p className="text-sm text-muted-foreground mt-1">
              {mode === "login" && "Sign in to continue to verbox.ai"}
              {mode === "signup" && "Start translating videos in minutes"}
              {mode === "forgot" && "We'll send you a reset link"}
            </p>
          </div>

          {/* Google Sign In - always mounted so the SDK button survives mode switches */}
          <div className={mode === "forgot" ? "hidden" : ""}>
            <div id="google-signin-button" className="w-full flex justify-center" />

            <div className="relative my-6">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-border" />
              </div>
              <div className="relative flex justify-center text-xs uppercase">
                <span className="bg-background px-2 text-muted-foreground">Or continue with email</span>
              </div>
            </div>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            {mode === "signup" && (
              <>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <Label className="text-xs text-muted-foreground">First Name</Label>
                    <div className="relative mt-1">
                      <User className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                      <Input
                        value={firstName}
                        onChange={(e) => setFirstName(e.target.value)}
                        placeholder="John"
                        className="pl-9"
                        required
                      />
                    </div>
                  </div>
                  <div>
                    <Label className="text-xs text-muted-foreground">Last Name</Label>
                    <div className="relative mt-1">
                      <User className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                      <Input
                        value={lastName}
                        onChange={(e) => setLastName(e.target.value)}
                        placeholder="Doe"
                        className="pl-9"
                        required
                      />
                    </div>
                  </div>
                </div>

                <div>
                  <Label className="text-xs text-muted-foreground">Username</Label>
                  <div className="relative mt-1">
                    <User className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      value={username}
                      onChange={(e) => setUsername(e.target.value)}
                      placeholder="johnDoe"
                      className="pl-9"
                      required
                    />
                  </div>
                </div>
              </>
            )}

            <div>
              <Label className="text-xs text-muted-foreground">Email</Label>
              <div className="relative mt-1">
                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@example.com"
                  className="pl-9"
                  required
                />
              </div>
            </div>

            {mode !== "forgot" && (
              <div>
                <Label className="text-xs text-muted-foreground">Password</Label>
                <div className="relative mt-1">
                  <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    type={showPassword ? "text" : "password"}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="Enter your password"
                    className="pl-9 pr-9"
                    required
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                  >
                    {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  </button>
                </div>
                {mode === "signup" && (
                  <p className="text-[10px] text-muted-foreground mt-1">Must be at least 8 characters</p>
                )}
              </div>
            )}

            {mode === "signup" && (
              <div>
                <Label className="text-xs text-muted-foreground">Confirm Password</Label>
                <div className="relative mt-1">
                  <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    type={showConfirmPassword ? "text" : "password"}
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    placeholder="Confirm your password"
                    className="pl-9 pr-9"
                    required
                  />
                  <button
                    type="button"
                    onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                  >
                    {showConfirmPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  </button>
                </div>
              </div>
            )}

            {mode === "login" && (
              <div className="flex justify-end">
                <button
                  type="button"
                  onClick={() => { setMode("forgot"); resetForm() }}
                  className="text-xs text-primary hover:underline"
                >
                  Forgot password?
                </button>
              </div>
            )}

            {error && (
              <div className="text-sm text-destructive bg-destructive/10 rounded-md p-2">
                <p>{error}</p>
                {showResendVerification && (
                  <button
                    type="button"
                    onClick={handleResendVerification}
                    disabled={resendingVerification}
                    className="mt-2 text-xs text-primary hover:underline disabled:opacity-50"
                  >
                    {resendingVerification ? "Sending..." : "Resend verification email"}
                  </button>
                )}
              </div>
            )}

            {success && (
              <p className="text-sm text-success bg-success/10 rounded-md p-2">{success}</p>
            )}

            {mode === "signup" && (
              <Turnstile
                ref={turnstileRef}
                siteKey={process.env.NEXT_PUBLIC_CF_TURNSTILE_SITE_KEY!}
                onSuccess={(token) => setTurnstileToken(token)}
                onExpire={() => setTurnstileToken("")}
                onError={() => setTurnstileToken("")}
                options={{ theme: "auto" }}
              />
            )}

            <Button type="submit" className="w-full h-11 gap-2" disabled={isLoading}>
              {isLoading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  {mode === "login" && "Signing in..."}
                  {mode === "signup" && "Creating account..."}
                  {mode === "forgot" && "Sending link..."}
                </>
              ) : (
                <>
                  {mode === "login" && "Sign In"}
                  {mode === "signup" && "Create Account"}
                  {mode === "forgot" && "Send Reset Link"}
                  <ArrowRight className="h-4 w-4" />
                </>
              )}
            </Button>
          </form>

          <div className="mt-6 text-center text-sm">
            {mode === "login" && (
              <p className="text-muted-foreground">
                Don&apos;t have an account?{" "}
                <button
                  onClick={() => { setMode("signup"); resetForm() }}
                  className="text-primary font-medium hover:underline"
                >
                  Sign up
                </button>
              </p>
            )}
            {mode === "signup" && (
              <p className="text-muted-foreground">
                Already have an account?{" "}
                <button
                  onClick={() => { setMode("login"); resetForm() }}
                  className="text-primary font-medium hover:underline"
                >
                  Sign in
                </button>
              </p>
            )}
            {mode === "forgot" && (
              <button
                onClick={() => { setMode("login"); resetForm() }}
                className="text-primary font-medium hover:underline"
              >
                Back to sign in
              </button>
            )}
          </div>

          {(mode === "signup" || mode === "login") && (
            <p className="mt-6 text-[10px] text-center text-muted-foreground">
              By using verbox.ai, you agree to our{" "}
              <a href={`/terms?from=${mode}`} target="_blank" className="underline">Terms of Service</a> and{" "}
              <a href={`/privacy?from=${mode}`} target="_blank" className="underline">Privacy Policy</a>.
            </p>
          )}
        </div>
      </div>
    </div>

    <Dialog open={showResetSentDialog} onOpenChange={setShowResetSentDialog}>
      <DialogContent className="sm:max-w-sm text-center">
        <DialogHeader>
          <div className="flex justify-center mb-2">
            <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center">
              <Mail className="h-6 w-6 text-primary" />
            </div>
          </div>
          <DialogTitle>Check your email</DialogTitle>
          <DialogDescription className="mt-2">
            If <span className="font-medium text-foreground">{resetEmail}</span>{" "}is registered, you&apos;ll receive a password reset link shortly.
          </DialogDescription>
        </DialogHeader>
        <Button className="w-full mt-2" onClick={() => setShowResetSentDialog(false)}>
          Back to Sign In
        </Button>
      </DialogContent>
    </Dialog>

    <Dialog open={showPasswordChangedDialog} onOpenChange={setShowPasswordChangedDialog}>
      <DialogContent className="sm:max-w-sm text-center">
        <DialogHeader>
          <div className="flex justify-center mb-2">
            <div className="h-12 w-12 rounded-full bg-success/10 flex items-center justify-center">
              <Lock className="h-6 w-6 text-success" />
            </div>
          </div>
          <DialogTitle>Password changed</DialogTitle>
          <DialogDescription className="mt-2">
            Your password has been updated successfully. You can now sign in with your new password.
          </DialogDescription>
        </DialogHeader>
        <Button className="w-full mt-2" onClick={() => setShowPasswordChangedDialog(false)}>
          Sign In
        </Button>
      </DialogContent>
    </Dialog>

    <Dialog open={showSignUpSuccessDialog} onOpenChange={setShowSignUpSuccessDialog}>
      <DialogContent className="sm:max-w-sm text-center">
        <DialogHeader>
          <div className="flex justify-center mb-2">
            <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center">
              <Mail className="h-6 w-6 text-primary" />
            </div>
          </div>
          <DialogTitle>Verify your email</DialogTitle>
          <DialogDescription className="mt-2">
            Your account has been created. Please check your inbox and click the verification link to activate your account.
          </DialogDescription>
        </DialogHeader>
        <Button className="w-full mt-2" onClick={() => setShowSignUpSuccessDialog(false)}>
          Got it
        </Button>
      </DialogContent>
    </Dialog>
    </>
  )
}

export default function LoginPage() {
  return (
    <Suspense>
      <LoginContent />
    </Suspense>
  )
}
