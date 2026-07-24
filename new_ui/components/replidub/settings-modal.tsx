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
import { Settings, User, Lock, Check, Loader2, Eye, EyeOff } from "lucide-react"
import { cn } from "@/lib/utils"
import { updateProfile, changePassword } from "@/lib/api"

interface SettingsModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  user?: {
    firstName: string
    lastName: string
    email: string
    authProvider: string
  }
}

type SettingsTab = "profile" | "password"

export function SettingsModal({
  open,
  onOpenChange,
  user = { firstName: "", lastName: "", email: "", authProvider: "local" },
}: SettingsModalProps) {
  const [activeTab, setActiveTab] = useState<SettingsTab>("profile")
  const isGoogle = user.authProvider === "google"

  // Profile state
  const [firstName, setFirstName] = useState(user.firstName)
  const [lastName, setLastName] = useState(user.lastName)
  const [email, setEmail] = useState(user.email)
  const [isSavingProfile, setIsSavingProfile] = useState(false)
  const [profileSaved, setProfileSaved] = useState(false)
  const [profileError, setProfileError] = useState("")

  // Password state
  const [currentPassword, setCurrentPassword] = useState("")
  const [newPassword, setNewPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")
  const [showCurrentPassword, setShowCurrentPassword] = useState(false)
  const [showNewPassword, setShowNewPassword] = useState(false)
  const [showConfirmPassword, setShowConfirmPassword] = useState(false)
  const [isSavingPassword, setIsSavingPassword] = useState(false)
  const [passwordSaved, setPasswordSaved] = useState(false)
  const [passwordError, setPasswordError] = useState("")

  // Sync fields when user prop changes (e.g. after data loads)
  useEffect(() => {
    setFirstName(user.firstName)
    setLastName(user.lastName)
    setEmail(user.email)
  }, [user.firstName, user.lastName, user.email])

  const handleSaveProfile = async () => {
    if (!firstName.trim() || !lastName.trim()) return
    setProfileError("")
    setIsSavingProfile(true)
    try {
      await updateProfile({
        first_name: firstName.trim(),
        last_name: lastName.trim(),
        ...(isGoogle ? {} : { email: email.trim() }),
      })
      setProfileSaved(true)
      setTimeout(() => setProfileSaved(false), 2000)
    } catch (e) {
      setProfileError(e instanceof Error ? e.message : "Failed to save profile")
    } finally {
      setIsSavingProfile(false)
    }
  }

  const handleChangePassword = async () => {
    setPasswordError("")
    if (newPassword.length < 8) {
      setPasswordError("Password must be at least 8 characters")
      return
    }
    if (newPassword !== confirmPassword) {
      setPasswordError("Passwords do not match")
      return
    }
    setIsSavingPassword(true)
    try {
      await changePassword(currentPassword, newPassword)
      setPasswordSaved(true)
      setCurrentPassword("")
      setNewPassword("")
      setConfirmPassword("")
      setTimeout(() => setPasswordSaved(false), 2000)
    } catch (e) {
      setPasswordError(e instanceof Error ? e.message : "Failed to change password")
    } finally {
      setIsSavingPassword(false)
    }
  }

  const isProfileValid = firstName.trim().length > 0 && lastName.trim().length > 0 &&
    (isGoogle || email.trim().length > 0)
  const isPasswordValid = currentPassword.length > 0 && newPassword.length >= 8 && confirmPassword.length > 0

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Settings className="h-5 w-5 text-primary" />
            Account Settings
          </DialogTitle>
          <DialogDescription>
            Manage your profile information and security settings.
          </DialogDescription>
        </DialogHeader>

        <div className="flex gap-4 py-2">
          {/* Tabs */}
          <div className="flex flex-col gap-1 w-32 shrink-0">
            <button
              onClick={() => setActiveTab("profile")}
              className={cn(
                "flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition-colors text-left",
                activeTab === "profile"
                  ? "bg-gradient-to-r from-primary/10 to-brand-end/10 text-primary"
                  : "text-muted-foreground hover:bg-secondary hover:text-foreground"
              )}
            >
              <User className="h-4 w-4" />
              Profile
            </button>
            <button
              onClick={() => !isGoogle && setActiveTab("password")}
              disabled={isGoogle}
              className={cn(
                "flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition-colors text-left",
                isGoogle
                  ? "opacity-40 cursor-not-allowed text-muted-foreground"
                  : activeTab === "password"
                  ? "bg-gradient-to-r from-primary/10 to-brand-end/10 text-primary"
                  : "text-muted-foreground hover:bg-secondary hover:text-foreground"
              )}
            >
              <Lock className="h-4 w-4" />
              Password
            </button>
          </div>

          {/* Content */}
          <div className="flex-1 min-w-0">
            {activeTab === "profile" && (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <Label className="text-xs text-muted-foreground">First Name</Label>
                    <Input
                      value={firstName}
                      onChange={(e) => setFirstName(e.target.value)}
                      className="mt-1"
                    />
                  </div>
                  <div>
                    <Label className="text-xs text-muted-foreground">Last Name</Label>
                    <Input
                      value={lastName}
                      onChange={(e) => setLastName(e.target.value)}
                      className="mt-1"
                    />
                  </div>
                </div>

                <div>
                  <Label className="text-xs text-muted-foreground">Email</Label>
                  <Input
                    value={isGoogle ? user.email : email}
                    onChange={(e) => !isGoogle && setEmail(e.target.value)}
                    disabled={isGoogle}
                    className="mt-1"
                  />
                  {isGoogle && (
                    <p className="text-[10px] text-muted-foreground mt-1">
                      Email is managed by Google and cannot be changed here.
                    </p>
                  )}
                </div>

                {profileError && (
                  <p className="text-xs text-destructive">{profileError}</p>
                )}

                <Button
                  onClick={handleSaveProfile}
                  disabled={!isProfileValid || isSavingProfile}
                  className="w-full gap-2"
                >
                  {isSavingProfile ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Saving...
                    </>
                  ) : profileSaved ? (
                    <>
                      <Check className="h-4 w-4" />
                      Saved!
                    </>
                  ) : (
                    "Save Changes"
                  )}
                </Button>
              </div>
            )}

            {activeTab === "password" && !isGoogle && (
              <div className="space-y-4">
                <div>
                  <Label className="text-xs text-muted-foreground">Current Password</Label>
                  <div className="relative mt-1">
                    <Input
                      type={showCurrentPassword ? "text" : "password"}
                      value={currentPassword}
                      onChange={(e) => setCurrentPassword(e.target.value)}
                      placeholder="Enter current password"
                    />
                    <button
                      type="button"
                      onClick={() => setShowCurrentPassword(!showCurrentPassword)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                    >
                      {showCurrentPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </button>
                  </div>
                </div>

                <div>
                  <Label className="text-xs text-muted-foreground">New Password</Label>
                  <div className="relative mt-1">
                    <Input
                      type={showNewPassword ? "text" : "password"}
                      value={newPassword}
                      onChange={(e) => setNewPassword(e.target.value)}
                      placeholder="Enter new password"
                    />
                    <button
                      type="button"
                      onClick={() => setShowNewPassword(!showNewPassword)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                    >
                      {showNewPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </button>
                  </div>
                  <p className="text-[10px] text-muted-foreground mt-1">
                    Must be at least 8 characters long.
                  </p>
                </div>

                <div>
                  <Label className="text-xs text-muted-foreground">Confirm New Password</Label>
                  <div className="relative mt-1">
                    <Input
                      type={showConfirmPassword ? "text" : "password"}
                      value={confirmPassword}
                      onChange={(e) => setConfirmPassword(e.target.value)}
                      placeholder="Confirm new password"
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

                {passwordError && (
                  <p className="text-xs text-destructive">{passwordError}</p>
                )}

                <Button
                  onClick={handleChangePassword}
                  disabled={!isPasswordValid || isSavingPassword}
                  className="w-full gap-2"
                >
                  {isSavingPassword ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Updating...
                    </>
                  ) : passwordSaved ? (
                    <>
                      <Check className="h-4 w-4" />
                      Password Updated!
                    </>
                  ) : (
                    "Change Password"
                  )}
                </Button>
              </div>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
