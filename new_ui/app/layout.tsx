import type { Metadata } from 'next'
import { Geist, Geist_Mono } from 'next/font/google'
import { Analytics } from '@vercel/analytics/next'
import './globals.css'

const _geist = Geist({ subsets: ["latin"] });
const _geistMono = Geist_Mono({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: 'verbox.ai — AI Video Translation & Dubbing',
  description: 'Professional AI-powered video translation and dubbing. Reach global audiences in different languages. Fast, high-quality, easy to use.',

}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
    <head>
        <link rel="icon" type="image/png" sizes="32x32" href="/favicon-32x32.png"/>
        <link rel="icon" type="image/png" sizes="16x16" href="/favicon-16x16.png"/>
        <link rel="apple-touch-icon" sizes="180x180" href="/apple-touch-icon.png"/>
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              "@context": "https://schema.org",
              "@type": "Organization",
              "name": "verbox.ai",
              "url": "https://verbox.ai",
              "logo": {
                "@type": "ImageObject",
                "url": "https://verbox.ai/logo.png",
                "width": 256,
                "height": 257
              }
            })
          }}
        />
    </head>

    <body className="font-sans antialiased">
    {children}
    <Analytics/>
    </body>
    </html>
  )
}
