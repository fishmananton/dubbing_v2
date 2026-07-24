"use client"

import { useEffect } from "react"
import { useRouter } from "next/navigation"

export function AuthRedirect() {
  const router = useRouter()

  useEffect(() => {
    fetch(`${process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000"}/auth/me`, {
      credentials: "include",
    })
      .then((res) => {
        if (res.ok) router.replace("/app")
      })
      .catch(() => {})
  }, [router])

  return null
}
