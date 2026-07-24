"use client"

import { useState, useEffect } from "react"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { CreditCard, Plus, Loader2, ExternalLink } from "lucide-react"
import { cn } from "@/lib/utils"
import { createCheckoutSession } from "@/lib/api"

interface BillingModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  currentBalance?: number
}

const PRESET_AMOUNTS = [10, 25, 50, 100]

export function BillingModal({ open, onOpenChange, currentBalance = 0 }: BillingModalProps) {
  const [selectedAmount, setSelectedAmount] = useState<number | null>(25)
  const [customAmount, setCustomAmount] = useState("")
  const [isProcessing, setIsProcessing] = useState(false)
  const [error, setError] = useState("")

  useEffect(() => {
    const handlePageShow = (e: PageTransitionEvent) => {
      if (e.persisted) setIsProcessing(false)
    }
    window.addEventListener("pageshow", handlePageShow)
    return () => window.removeEventListener("pageshow", handlePageShow)
  }, [])

  const getAmount = () => {
    if (customAmount) return parseFloat(customAmount) || 0
    return selectedAmount || 0
  }

  const isFormValid = getAmount() >= 5

  const handleAddFunds = async () => {
    const amount = getAmount()
    if (amount < 5) return

    setIsProcessing(true)
    setError("")
    try {
      const amountCents = Math.round(amount * 100)
      const checkoutUrl = await createCheckoutSession(amountCents)
      window.location.href = checkoutUrl
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to start checkout")
      setIsProcessing(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <CreditCard className="h-5 w-5 text-primary" />
            Billing & Balance
          </DialogTitle>
          <DialogDescription>
            Add funds to your account to continue using verbox.ai services.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6 py-2">
          {/* Current Balance */}
          <div className="rounded-lg border border-border bg-secondary/50 p-4">
            <p className="text-xs text-muted-foreground mb-1">Current Balance</p>
            <p className="text-2xl font-bold text-foreground">${(currentBalance / 100).toFixed(2)}</p>
          </div>

          {/* Amount Selection */}
          <div className="space-y-3">
            <Label className="text-sm font-medium">Add Funds</Label>
            <div className="grid grid-cols-4 gap-2">
              {PRESET_AMOUNTS.map((amount) => (
                <button
                  key={amount}
                  onClick={() => { setSelectedAmount(amount); setCustomAmount("") }}
                  className={cn(
                    "rounded-lg border py-2 text-sm font-medium transition-colors",
                    selectedAmount === amount && !customAmount
                      ? "border-[#7247ED] bg-gradient-to-r from-primary/10 to-brand-end/10 text-primary"
                      : "border-border bg-secondary hover:bg-muted text-foreground"
                  )}
                >
                  ${amount}
                </button>
              ))}
            </div>
            <div className="relative">
              <span className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground">$</span>
              <Input
                type="number"
                placeholder="Custom amount (min $5)"
                value={customAmount}
                onChange={(e) => { setCustomAmount(e.target.value); setSelectedAmount(null) }}
                className="pl-7"
                min="5"
              />
            </div>
          </div>

          {error && (
            <p className="text-xs text-destructive">{error}</p>
          )}

          {/* Submit Button */}
          <Button
            onClick={handleAddFunds}
            disabled={!isFormValid || isProcessing}
            className="w-full gap-2"
          >
            {isProcessing ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Redirecting to Stripe...
              </>
            ) : (
              <>
                <ExternalLink className="h-4 w-4" />
                Pay ${getAmount().toFixed(2)} with Stripe
              </>
            )}
          </Button>

          <p className="text-[10px] text-center text-muted-foreground">
            You will be redirected to Stripe's secure checkout page.
          </p>
        </div>
      </DialogContent>
    </Dialog>
  )
}
