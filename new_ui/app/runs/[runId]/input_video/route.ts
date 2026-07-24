import { type NextRequest, NextResponse } from "next/server"

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000"

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ runId: string }> }
) {
  const { runId } = await params
  const upstream = `${API_BASE_URL}/runs/${runId}/input_video`

  const headers: HeadersInit = {
    Cookie: request.headers.get("cookie") ?? "",
  }

  const range = request.headers.get("range")
  if (range) {
    headers["Range"] = range
  }

  const res = await fetch(upstream, { headers })

  if (!res.ok && res.status !== 206) {
    return new NextResponse(null, { status: res.status })
  }

  const proxyHeaders = new Headers()
  for (const key of ["content-type", "content-length", "content-range", "accept-ranges"]) {
    const val = res.headers.get(key)
    if (val) proxyHeaders.set(key, val)
  }

  return new NextResponse(res.body, {
    status: res.status,
    headers: proxyHeaders,
  })
}
