"use client"

import { useRouter, useSearchParams } from "next/navigation"
import { Button } from "@/components/ui/button"
import { ArrowLeft } from "lucide-react"
import { Suspense } from "react"

function TermsContent() {
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

        <h1 className="text-3xl font-bold text-foreground mt-6 mb-2">Terms of Service</h1>
        <p className="text-sm text-muted-foreground mb-10">Last updated: April 21, 2026</p>

        <div className="prose prose-sm max-w-none space-y-8 text-foreground">

          <section>
            <h2 className="text-xl font-semibold mb-3">1. Acceptance of Terms</h2>
            <p className="text-muted-foreground leading-relaxed">
              verbox.ai is operated by <span className="text-foreground font-medium">Replitrust Ltd</span>, a company registered in England and Wales. By creating an account or using verbox.ai ("the Service"), you agree to be bound by these Terms of Service. If you do not agree to these terms, please do not use the Service.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-semibold mb-3">2. Description of Service</h2>
            <p className="text-muted-foreground leading-relaxed">
              verbox.ai is an AI-powered video dubbing and localization platform. It allows users to upload video content, which is then processed to produce translated and dubbed versions using automated speech recognition, translation, and text-to-speech technologies.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-semibold mb-3">3. Account Registration</h2>
            <p className="text-muted-foreground leading-relaxed">
              You must provide accurate and complete information when creating an account. You are responsible for maintaining the security of your account credentials and for all activity that occurs under your account. You must notify us immediately of any unauthorized use.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-semibold mb-3">4. Acceptable Use</h2>
            <p className="text-muted-foreground leading-relaxed mb-3">You agree not to use the Service to:</p>
            <ul className="list-disc pl-6 space-y-2 text-muted-foreground">
              <li>Upload content that infringes on any third party's intellectual property rights</li>
              <li>Process content containing illegal material, hate speech, or content that exploits or harms minors</li>
              <li>Create deepfakes or misleading media intended to deceive or defraud others</li>
              <li>Attempt to reverse-engineer, scrape, or interfere with the Service's infrastructure</li>
              <li>Resell or redistribute the Service without express written permission</li>
              <li>Violate any applicable laws or regulations</li>
              <li>Use or generate synthetic voices or likenesses without proper authorization or consent</li>
            </ul>
          </section>

          <section>
            <h2 className="text-xl font-semibold mb-3">5. User Content and Responsibilities</h2>
            <p className="text-muted-foreground leading-relaxed mb-3">
              You are solely responsible for all content you upload, submit, or process through the Service ("User Content").
            </p>
            <p className="text-muted-foreground leading-relaxed mb-3">You represent and warrant that:</p>
            <ul className="list-disc pl-6 space-y-2 text-muted-foreground mb-3">
              <li>You own or have obtained all necessary rights, licenses, consents, and permissions to use and process the User Content.</li>
              <li>You have the legal right and explicit consent to use, reproduce, or synthesize any voice, likeness, or identity included in the User Content.</li>
              <li>Your use of the Service does not violate any applicable laws, intellectual property rights, privacy rights, or publicity rights.</li>
            </ul>
            <p className="text-muted-foreground leading-relaxed mb-3">You agree not to use the Service to:</p>
            <ul className="list-disc pl-6 space-y-2 text-muted-foreground mb-3">
              <li>Process copyrighted content without authorization</li>
              <li>Generate synthetic media that is misleading, deceptive, or intended to impersonate others without consent</li>
              <li>Violate any laws or third-party rights</li>
            </ul>
            <p className="text-muted-foreground leading-relaxed">
              You agree to indemnify, defend, and hold harmless verbox.ai and Replitrust Ltd from any claims, damages, liabilities, and expenses arising from your User Content or your use of the Service.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-semibold mb-3">7. Intellectual Property</h2>
            <p className="text-muted-foreground leading-relaxed">
              You retain ownership of all content you upload. By uploading content, you grant verbox.ai a limited, non-exclusive, worldwide license to use, process, reproduce, modify, transform, and generate derivative outputs from your content solely for the purpose of providing and improving the Service. We do not claim ownership of your videos or output files. The verbox.ai platform, brand, and underlying technology remain the exclusive property of verbox.ai.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-semibold mb-3">8. Credits and Payments</h2>
            <p className="text-muted-foreground leading-relaxed">
              The Service operates on a credit-based system. Credits are consumed when processing jobs are run. Payments are processed securely through Stripe. Credits are non-refundable except where required by applicable law. We reserve the right to change pricing with reasonable notice.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-semibold mb-3">9. Disclaimer of Warranties</h2>
            <p className="text-muted-foreground leading-relaxed">
              The Service is provided "as is" and "as available" without warranties of any kind, express or implied. We do not warrant that the Service will be uninterrupted, error-free, or that output quality will meet any particular standard. AI-generated outputs may contain errors, and you are responsible for reviewing results before use.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-semibold mb-3">10. AI-Generated Content</h2>
            <p className="text-muted-foreground leading-relaxed">
              The Service uses artificial intelligence to generate translated audio, voices, and subtitles. Outputs may not be accurate, complete, or suitable for all purposes. verbox.ai does not guarantee the correctness, quality, or legality of generated content. You are solely responsible for reviewing and using outputs.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-semibold mb-3">11. Limitation of Liability</h2>
            <p className="text-muted-foreground leading-relaxed">
              To the maximum extent permitted by law, verbox.ai shall not be liable for any indirect, incidental, special, or consequential damages arising from your use of the Service, including but not limited to loss of data, loss of revenue, or loss of business opportunity. Our total liability to you for any claim shall not exceed the amount you paid us in the 30 days preceding the claim. verbox.ai is not responsible for how users distribute, publish, or use generated content outside the Service.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-semibold mb-3">12. Termination</h2>
            <p className="text-muted-foreground leading-relaxed">
              We reserve the right to suspend or terminate your account at any time for violations of these Terms. You may delete your account at any time. Upon termination, your right to use the Service ceases and any remaining credits are forfeited.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-semibold mb-3">13. Governing Law</h2>
            <p className="text-muted-foreground leading-relaxed">
              These Terms are governed by and construed in accordance with the laws of England and Wales. Any disputes arising under these Terms shall be subject to the exclusive jurisdiction of the courts of England and Wales.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-semibold mb-3">14. Changes to Terms</h2>
            <p className="text-muted-foreground leading-relaxed">
              We may update these Terms at any time. We will notify you of significant changes via email or an in-app notice. Continued use of the Service after changes take effect constitutes acceptance of the new Terms.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-semibold mb-3">15. Contact</h2>
            <p className="text-muted-foreground leading-relaxed">
              For questions about these Terms, please contact us at <a href="mailto:support@verbox.ai" className="text-primary hover:underline">support@verbox.ai</a>.
            </p>
            <p className="text-muted-foreground leading-relaxed mt-2">
              Replitrust Ltd<br />
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

export default function TermsPage() {
  return (
    <Suspense>
      <TermsContent />
    </Suspense>
  )
}
