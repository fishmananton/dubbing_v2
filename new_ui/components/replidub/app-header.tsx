"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Zap, Plus, ChevronDown, Settings, LogOut, HelpCircle, CreditCard, Moon, Sun, Trash2 } from "lucide-react"
import { BillingModal } from "./billing-modal"
import { SettingsModal } from "./settings-modal"
import Link from "next/link"

export interface Project {
  id: number
  name: string
}

interface AppHeaderProps {
  onNewProject: (name: string) => void
  onSelectProject?: (projectId: number) => void
  onDeleteProject?: (projectId: number) => void
  projects?: Project[]
  selectedProjectId?: number | null
  transferring?: boolean
  darkMode?: boolean
  onDarkModeChange?: (darkMode: boolean) => void
  onLogout?: () => void
  balance?: number
  currentUser?: {
    firstName: string
    lastName: string
    email: string
    authProvider: string
  }
  billingOpen?: boolean
  onBillingOpenChange?: (open: boolean) => void
}

export function AppHeader({
  onNewProject,
  onSelectProject,
  onDeleteProject,
  projects = [],
  selectedProjectId,
  transferring = false,
  darkMode = false,
  onDarkModeChange,
  onLogout,
  balance,
  currentUser,
  billingOpen,
  onBillingOpenChange,
}: AppHeaderProps) {
  const [showNewProjectDialog, setShowNewProjectDialog] = useState(false)
  const [newProjectName, setNewProjectName] = useState("")
  const [_showBillingModal, _setShowBillingModal] = useState(false)
  const showBillingModal = billingOpen !== undefined ? billingOpen : _showBillingModal
  const setShowBillingModal = (open: boolean) => {
    _setShowBillingModal(open)
    onBillingOpenChange?.(open)
  }
  const [showSettingsModal, setShowSettingsModal] = useState(false)
  const [showDeleteDialog, setShowDeleteDialog] = useState(false)

  const selectedProject = projects.find((p) => p.id === selectedProjectId)

  const handleConfirmDelete = () => {
    if (selectedProjectId && onDeleteProject) {
      onDeleteProject(selectedProjectId)
    }
    setShowDeleteDialog(false)
  }

  const handleCreateProject = () => {
    if (newProjectName.trim()) {
      onNewProject(newProjectName.trim())
      setShowNewProjectDialog(false)
      setNewProjectName("")
    }
  }

  // Apply dark mode to document
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add("dark")
    } else {
      document.documentElement.classList.remove("dark")
    }
  }, [darkMode])

  return (
    <>
      <header className="sticky top-0 z-50 flex h-14 items-center justify-between border-b border-border bg-card px-3 sm:px-6 shadow-sm gap-2">
        {/* Left: Logo */}
        <div className="flex items-center gap-2 shrink-0">
          <img src="/logo.png" alt="verbox.ai" className="h-8 w-8 rounded-lg object-contain" />
          <span className="hidden sm:inline text-base font-semibold tracking-tight text-foreground">verbox.ai</span>
        </div>

        {/* Center: Project selector */}
        <div className="flex flex-1 sm:flex-none items-center gap-1 sm:gap-2 min-w-0">
          <Select
            value={selectedProjectId?.toString() || ""}
            onValueChange={(val) => {
              if (val && onSelectProject) onSelectProject(Number(val))
            }}
            disabled={transferring}
          >
            <SelectTrigger className="h-8 flex-1 sm:w-48 min-w-0 border-border bg-secondary text-sm font-medium">
              <SelectValue placeholder="Select project" />
            </SelectTrigger>
            <SelectContent>
              {projects.map((p) => (
                <SelectItem key={p.id} value={p.id.toString()}>
                  {p.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => selectedProjectId && !transferring && setShowDeleteDialog(true)}
            disabled={!selectedProjectId || transferring}
            className="flex h-8 w-8 p-0 text-muted-foreground hover:text-destructive hover:bg-destructive/10"
            title="Delete project"
          >
            <Trash2 className="h-3.5 w-3.5" />
          </Button>
          <Button
            size="sm"
            onClick={() => setShowNewProjectDialog(true)}
            disabled={transferring}
            className="h-8 w-8 sm:w-auto sm:gap-1.5 sm:px-3 p-0 bg-gradient-to-r from-primary to-brand-end text-primary-foreground hover:opacity-90 shrink-0"
          >
            <Plus className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">New Project</span>
          </Button>
        </div>

        {/* Right: Credits + Avatar */}
        <div className="flex items-center gap-2 sm:gap-3 shrink-0">
          <button
            onClick={() => setShowBillingModal(true)}
            className="flex items-center gap-1 sm:gap-1.5 rounded-full border border-border bg-accent px-2 sm:px-3 py-1 hover:bg-accent/80 transition-colors"
          >
            <CreditCard className="h-3.5 w-3.5 text-accent-foreground" />
            <span className="text-xs font-semibold text-accent-foreground">
              ${balance !== undefined ? (balance / 100).toFixed(2) : "—"}
            </span>
          </button>

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button className="flex items-center gap-1.5 rounded-full outline-none ring-ring focus-visible:ring-2">
                <Avatar className="h-8 w-8">
                  <AvatarFallback className="bg-gradient-to-br from-primary to-brand-end text-primary-foreground text-xs">
                    {currentUser ? `${currentUser.firstName[0] ?? ""}${currentUser.lastName[0] ?? ""}`.toUpperCase() : "?"}
                  </AvatarFallback>
                </Avatar>
                <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-48">
              <div className="px-2 py-1.5">
                <p className="text-sm font-medium">
                  {currentUser ? `${currentUser.firstName} ${currentUser.lastName}`.trim() : "—"}
                </p>
                <p className="text-xs text-muted-foreground">{currentUser?.email ?? ""}</p>
              </div>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={() => onDarkModeChange?.(!darkMode)}>
                {darkMode ? <Sun className="mr-2 h-4 w-4" /> : <Moon className="mr-2 h-4 w-4" />}
                {darkMode ? "Light Mode" : "Dark Mode"}
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={() => setShowBillingModal(true)}>
                <CreditCard className="mr-2 h-4 w-4" /> Billing
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => setShowSettingsModal(true)}>
                <Settings className="mr-2 h-4 w-4" /> Settings
              </DropdownMenuItem>
              <DropdownMenuItem asChild>
                <Link href={selectedProjectId ? `/help?project=${selectedProjectId}` : "/help"}><HelpCircle className="mr-2 h-4 w-4" /> Help</Link>
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem className="text-destructive" onClick={onLogout}><LogOut className="mr-2 h-4 w-4" /> Sign out</DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </header>

      {/* Billing Modal */}
      <BillingModal open={showBillingModal} onOpenChange={setShowBillingModal} currentBalance={balance} />

      {/* Settings Modal */}
      <SettingsModal
        open={showSettingsModal}
        onOpenChange={setShowSettingsModal}
        user={currentUser}
      />

      {/* Delete Project Confirmation */}
      <Dialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Delete Project</DialogTitle>
            <DialogDescription>
              Delete &ldquo;{selectedProject?.name}&rdquo;? All files will be permanently removed and this cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowDeleteDialog(false)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={handleConfirmDelete}>
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* New Project Dialog */}
      <Dialog open={showNewProjectDialog} onOpenChange={setShowNewProjectDialog}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Create New Project</DialogTitle>
            <DialogDescription>
              Enter a name for your new video translation project.
            </DialogDescription>
          </DialogHeader>
          <div className="py-4">
            <Label htmlFor="project-name" className="text-sm font-medium">
              Project Name
            </Label>
            <Input
              id="project-name"
              placeholder="e.g., YouTube — Spanish Dub"
              value={newProjectName}
              onChange={(e) => setNewProjectName(e.target.value)}
              className="mt-2"
              onKeyDown={(e) => e.key === "Enter" && handleCreateProject()}
            />
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowNewProjectDialog(false)}>
              Cancel
            </Button>
            <Button onClick={handleCreateProject} disabled={!newProjectName.trim()}>
              Create Project
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}
