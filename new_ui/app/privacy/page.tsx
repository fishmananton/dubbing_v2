"use client"

import { useRouter, useSearchParams } from "next/navigation"
import { Button } from "@/components/ui/button"
import { ArrowLeft } from "lucide-react"
import { Suspense } from "react"

function PrivacyContent() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const from = searchParams.get("from")
  const backUrl = from === "signup" ? "/login?mode=signup" : "/login"

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-3xl mx-auto px-6 py-12">
        <Button
          variant="ghost"
          size="sm"
          className="mb-8 gap-2 text-muted-foreground"
          onClick={() => router.push(backUrl)}
        >
          <ArrowLeft className="h-4 w-4" />
          Back
        </Button>

        <div className="flex items-center gap-3 mb-2">
          <img src="/logo.png" alt="verbox.ai" className="h-8 w-8 rounded-lg object-contain" />
          <span className="text-lg font-semibold text-foreground">verbox.ai</span>
        </div>

        <h1 className="text-3xl font-bold text-foreground mt-6 mb-2">Privacy Policy</h1>
        <p className="text-sm text-muted-foreground mb-10">Last updated: April 21, 2026</p>

        <div className="prose prose-sm max-w-none space-y-8 text-foreground">

          <section>
            <h2 className="text-xl font-semibold mb-3">1. Introduction</h2>
            <p className="text-muted-foreground leading-relaxed">
              verbox.ai is operated by <span className="text-foreground font-medium">Replitrust Ltd</span>, a company registered in England and Wales ("we", "us", "our"). Replitrust Ltd is the data controller for personal data collected through this Service. This Privacy Policy explains what data we collect, how we use it, and your rights regarding that data when you use our video dubbing platform at verbox.ai.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-semibold mb-3">2. Information We Collect</h2>
            <p className="text-muted-foreground leading-relaxed mb-3">We collect the following categories of information:</p>
            <ul className="list-disc pl-6 space-y-2 text-muted-foreground">
              <li><span className="text-foreground font-medium">Account information:</span> Name, email address, and username when you register. If you sign in via Google, we receive your name and email from Google.</li>
              <li><span className="text-foreground font-medium">Payment information:</span> Billing details are processed by Stripe. We do not store your full card number — only what Stripe shares with us (last 4 digits, card type, billing status).</li>
              <li><span className="text-foreground font-medium">Uploaded content:</span> Videos you upload and the resulting processed files (audio tracks, subtitles, dubbed output) are stored temporarily on our servers and in AWS S3 to deliver the Service.</li>
              <li><span className="text-foreground font-medium">Usage data:</span> Information about how you use the Service, including processing jobs run, languages selected, and credit consumption.</li>
              <li><span className="text-foreground font-medium">Technical data:</span> IP address, browser type, and session tokens used for authentication and security.</li>
            </ul>
          </section>

          <section>
            <h2 className="text-xl font-semibold mb-3">3. How We Use Your Information</h2>
            <ul className="list-disc pl-6 space-y-2 text-muted-foreground">
              <li>To provide, operate, and improve the Service</li>
              <li>To process your videos and return dubbed output to you</li>
              <li>To manage your account, credits, and billing</li>
              <li>To send transactional emails (email verification, password resets, billing receipts)</li>
              <li>To detect and prevent fraud, abuse, or security incidents</li>
              <li>To comply with legal obligations</li>
            </ul>
          </section>

          <section>
            <h2 className="text-xl font-semibold mb-3">4. Third-Party Services</h2>
            <p className="text-muted-foreground leading-relaxed mb-3">
              To deliver the Service, we work with trusted third-party providers in the following categories:
            </p>
            <ul className="list-disc pl-6 space-y-2 text-muted-foreground">
              <li><span className="text-foreground font-medium">Cloud storage</span> — to securely store uploaded and processed files</li>
              <li><span className="text-foreground font-medium">AI processing</span> — for speech recognition, translation, and voice synthesis</li>
              <li><span className="text-foreground font-medium">Payment processing</span> — to handle billing and transactions securely</li>
              <li><span className="text-foreground font-medium">Authentication</span> — to support optional third-party sign-in methods</li>
            </ul>
            <p className="text-muted-foreground leading-relaxed mt-3">
              Each provider is carefully selected and operates under their own privacy policy and data processing agreements. We only share the minimum data necessary for each provider to perform their function, and we do not sell your data to any third party.
            </p>
            <p className="text-muted-foreground leading-relaxed mt-3">
              We do not use uploaded content to train our own AI models without explicit user consent.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-semibold mb-3">5. Data Retention</h2>
            <p className="text-muted-foreground leading-relaxed">
              Uploaded videos and processed output files are retained for a limited period necessary to deliver the Service and allow download of results. In most cases, content is automatically deleted after processing unless required for technical or legal reasons. We do not retain your content indefinitely. Account information is retained for as long as your account exists. You may request deletion of your data at any time by contacting us.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-semibold mb-3">6. Data Security</h2>
            <p className="text-muted-foreground leading-relaxed">
              We implement industry-standard security measures including encrypted connections (HTTPS), hashed password storage, and session-based authentication. No method of transmission over the internet is 100% secure, and we cannot guarantee absolute security.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-semibold mb-3">7. Your Rights</h2>
            <p className="text-muted-foreground leading-relaxed mb-3">Depending on your location, you may have the right to:</p>
            <ul className="list-disc pl-6 space-y-2 text-muted-foreground">
              <li>Access the personal data we hold about you</li>
              <li>Request correction of inaccurate data</li>
              <li>Request deletion of your account and associated data</li>
              <li>Object to or restrict certain processing of your data</li>
              <li>Data portability (receiving your data in a machine-readable format)</li>
            </ul>
            <p className="text-muted-foreground leading-relaxed mt-3">
              To exercise any of these rights, contact us at <a href="mailto:support@verbox.ai" className="text-primary hover:underline">support@verbox.ai</a>.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-semibold mb-3">8. Cookies</h2>
            <p className="text-muted-foreground leading-relaxed">
              We use a single session cookie to keep you logged in. We do not use third-party tracking or advertising cookies.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-semibold mb-3">9. Children's Privacy</h2>
            <p className="text-muted-foreground leading-relaxed">
              The Service is not directed at children under the age of 13. We do not knowingly collect personal information from children. If you believe a child has provided us with personal data, please contact us and we will delete it.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-semibold mb-3">10. UK & GDPR Rights</h2>
            <p className="text-muted-foreground leading-relaxed">
              As a UK-based service, we comply with the UK GDPR and the Data Protection Act 2018. Our lawful basis for processing your personal data is contractual necessity (to deliver the Service) and legitimate interests (security and fraud prevention). You have the right to lodge a complaint with the Information Commissioner's Office (ICO) at <a href="https://ico.org.uk" target="_blank" className="text-primary hover:underline">ico.org.uk</a> if you believe your data rights have been violated.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-semibold mb-3">12. Changes to This Policy</h2>
            <p className="text-muted-foreground leading-relaxed">
              We may update this Privacy Policy from time to time. We will notify you of material changes via email or a prominent notice on the Service. Continued use of the Service after changes take effect constitutes acceptance of the updated policy.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-semibold mb-3">13. Contact Us</h2>
            <p className="text-muted-foreground leading-relaxed">
              For any privacy-related questions or requests, contact us at <a href="mailto:support@verbox.ai" className="text-primary hover:underline">support@verbox.ai</a>.
            </p>
            <p className="text-muted-foreground leading-relaxed mt-2">
              Replitrust Ltd (Data Controller)<br />
              167-169 Great Portland Street<br />
              London, England, W1W 5PF<br />
              United Kingdom
            </p>
          </section>

        </div>
      </div>
    </div>
  )
}

export default function PrivacyPage() {
  return (
    <Suspense>
      <PrivacyContent />
    </Suspense>
  )
}
