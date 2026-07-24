import type { MetadataRoute } from "next"

export default function robots(): MetadataRoute.Robots {
  return {
    rules: {
      userAgent: "*",
      allow: ["/", "/login", "/help"],
      disallow: ["/app", "/runs", "/verify-email"],
    },
    sitemap: "https://verbox.ai/sitemap.xml",
  }
}
