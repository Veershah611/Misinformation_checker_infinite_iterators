"use client"

import { Button } from "@/components/ui/button"
import { Navbar } from "@/components/navbar"
import { Footer } from "@/components/footer"
import { StatsSection } from "@/components/stats-section"
import { FloatingElements } from "@/components/floating-elements"
import { CheckCircle, Gavel, ClipboardCheck, Shield, Zap, Users } from "lucide-react"
import Link from "next/link"
import Particles from "@/components/Particles"
import BlurText from "@/components/BlurText"
import ProfileCard from "@/components/ProfileCard" // Make sure the path is correct

const handleAnimationComplete = () => {
  console.log("Animation completed!");
};

export default function HomePage() {
  // NOTE: I've used placeholder images from pravatar.cc. 
  // You should replace the `avatarUrl` and `miniAvatarUrl` props with your actual image URLs.
  const teamMembers = [
    {
      name: "Jemil Patel",
      title: "UI/UX Designer",
      handle: "alex_ai",
      avatarUrl: "/images/jemil_avatar.png",
      miniAvatarUrl: "/images/jemil_avatar.png",
    },
    {
      name: "Veer Shah",
      title: "Backend Developer",
      handle: "sam_data",
      avatarUrl: "https://i.pravatar.cc/500?img=2",
      miniAvatarUrl: "https://i.pravatar.cc/100?img=2",
    },
    {
      name: "Michael Chen",
      title: "Cybersecurity Expert",
      handle: "mike_sec",
      avatarUrl: "https://i.pravatar.cc/500?img=3",
      miniAvatarUrl: "https://i.pravatar.cc/100?img=3",
    },
    {
      name: "Jessica Davis",
      title: "UX/UI Designer",
      handle: "jess_designs",
      avatarUrl: "https://i.pravatar.cc/500?img=4",
      miniAvatarUrl: "https://i.pravatar.cc/100?img=4",
    },
    {
      name: "David Wilson",
      title: "Full-Stack Developer",
      handle: "dave_codes",
      avatarUrl: "https://i.pravatar.cc/500?img=5",
      miniAvatarUrl: "https://i.pravatar.cc/100?img=5",
    },
    {
      name: "Emily Carter",
      title: "Product Manager",
      handle: "em_prod",
      avatarUrl: "https://i.pravatar.cc/500?img=6",
      miniAvatarUrl: "https://i.pravatar.cc/100?img=6",
    },
  ]

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col">
      <Navbar />

      <main className="flex-1">
        {/* Hero Section */}
        <section className="relative flex items-center justify-center min-h-[60vh] md:min-h-[70vh] p-4 bg-gradient-to-br from-primary/10 via-background to-secondary/10 overflow-hidden">
          <div style={{ width: '100%', height: '600px', position: 'absolute', inset: 0, zIndex: 0 }}>
            <Particles />
          </div>
          <FloatingElements />
          <div className="container mx-auto px-4 sm:px-6 lg:px-8 text-center relative z-10">
            <div className="max-w-3xl mx-auto">

              <h1 className="text-center">
                <BlurText
                  text="Empowering Truth in a World of Noise"
                  delay={150}
                  animateBy="words"
                  direction="top"
                  onAnimationComplete={handleAnimationComplete}
                  className="text-4xl sm:text-3xl lg:text-4xl font-extrabold tracking-wider text-foreground"
                />
              </h1>

              <p className="mt-4 text-lg sm:text-xl text-muted-foreground tracking-wide animate-in fade-in-0 slide-in-from-bottom-8 duration-1000 delay-200">
                Our platform helps you verify information and combat the spread of misinformation.
              </p>

              <div className="mt-8 flex flex-col sm:flex-row gap-4 justify-center animate-in fade-in-0 slide-in-from-bottom-8 duration-1000 delay-400">
                <Link href="/">
                  <Button className="font-bold tracking-wide h-12 px-6 text-base shadow-lg transform hover:scale-105 transition-all duration-300 animate-pulse-glow">
                    Check Claim
                  </Button>
                </Link>
                <Button
                  variant="outline"
                  className="font-bold tracking-wide h-12 px-6 text-base transform hover:scale-105 transition-all duration-300 bg-transparent"
                >
                  Learn More
                </Button>
              </div>
            </div>
          </div>
        </section>

        <StatsSection />

        <section className="py-16 sm:py-24 bg-background">
          <div className="container mx-auto px-4 sm:px-6 lg:px-8">
            <div className="max-w-4xl mx-auto text-center animate-in fade-in-0 slide-in-from-bottom-4 duration-700">
              <h2 className="text-3xl sm:text-4xl font-extrabold tracking-wider text-foreground">
                Why Choose ClaimGuard?
              </h2>
              <p className="mt-4 text-lg text-muted-foreground tracking-wide">
                Advanced AI technology meets human expertise for unparalleled accuracy.
              </p>
            </div>
            <div className="mt-12 grid gap-8 md:grid-cols-3">
              <div className="flex flex-col items-center text-center gap-4 rounded-lg bg-card p-6 border shadow-sm transform hover:-translate-y-2 hover:shadow-lg transition-all duration-300 animate-float animate-delay-100">
                <div className="flex-shrink-0">
                  <Shield className="w-12 h-12 text-primary" />
                </div>
                <div className="flex flex-col">
                  <h3 className="text-xl font-bold tracking-wide">Trusted Sources</h3>
                  <p className="mt-2 text-muted-foreground tracking-wide">
                    We cross-reference multiple reliable sources to ensure accuracy and provide comprehensive analysis.
                  </p>
                </div>
              </div>
              <div className="flex flex-col items-center text-center gap-4 rounded-lg bg-card p-6 border shadow-sm transform hover:-translate-y-2 hover:shadow-lg transition-all duration-300 animate-float animate-delay-200">
                <div className="flex-shrink-0">
                  <Zap className="w-12 h-12 text-primary" />
                </div>
                <div className="flex flex-col">
                  <h3 className="text-xl font-bold tracking-wide">Lightning Fast</h3>
                  <p className="mt-2 text-muted-foreground tracking-wide">
                    Get results in seconds with our advanced AI algorithms that process information at incredible speed.
                  </p>
                </div>
              </div>
              <div className="flex flex-col items-center text-center gap-4 rounded-lg bg-card p-6 border shadow-sm transform hover:-translate-y-2 hover:shadow-lg transition-all duration-300 animate-float animate-delay-300">
                <div className="flex-shrink-0">
                  <Users className="w-12 h-12 text-primary" />
                </div>
                <div className="flex flex-col">
                  <h3 className="text-xl font-bold tracking-wide">Community Driven</h3>
                  <p className="mt-2 text-muted-foreground tracking-wide">
                    Join thousands of users working together to combat misinformation and promote truth.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="py-16 sm:py-24 bg-secondary/5" id="about">
          <div className="container mx-auto px-4 sm:px-6 lg:px-8">
            <div className="max-w-4xl mx-auto text-center animate-in fade-in-0 slide-in-from-bottom-4 duration-700">
              <h2 className="text-3xl sm:text-4xl font-extrabold tracking-wider text-foreground">How It Works</h2>
              <p className="mt-4 text-lg text-muted-foreground tracking-wide">
                A simple, transparent process to fact-check information and stay informed.
              </p>
            </div>
            <div className="mt-12 grid gap-8 md:grid-cols-3">
              <div className="flex flex-col items-center text-center gap-4 rounded-lg bg-card p-6 border shadow-sm transform hover:-translate-y-2 hover:shadow-lg transition-all duration-300 animate-slide-in-right animate-delay-100">
                <div className="flex-shrink-0">
                  <CheckCircle className="w-12 h-12 text-primary" />
                </div>
                <div className="flex flex-col">
                  <h3 className="text-xl font-bold tracking-wide">1. Submit a Claim</h3>
                  <p className="mt-2 text-muted-foreground tracking-wide">
                    Found a questionable piece of information? Submit it through our easy-to-use form with any
                    supporting links or evidence.
                  </p>
                </div>
              </div>
              <div className="flex flex-col items-center text-center gap-4 rounded-lg bg-card p-6 border shadow-sm transform hover:-translate-y-2 hover:shadow-lg transition-all duration-300 animate-slide-in-right animate-delay-200">
                <div className="flex-shrink-0">
                  <Gavel className="w-12 h-12 text-primary" />
                </div>
                <div className="flex flex-col">
                  <h3 className="text-xl font-bold tracking-wide">2. We Analyze</h3>
                  <p className="mt-2 text-muted-foreground tracking-wide">
                    Our AI analyze the claim, comparing it against reliable sources and identifying patterns of
                    misinformation.
                  </p>
                </div>
              </div>
              <div className="flex flex-col items-center text-center gap-4 rounded-lg bg-card p-6 border shadow-sm transform hover:-translate-y-2 hover:shadow-lg transition-all duration-300 animate-slide-in-right animate-delay-300">
                <div className="flex-shrink-0">
                  <ClipboardCheck className="w-12 h-12 text-primary" />
                </div>
                <div className="flex flex-col">
                  <h3 className="text-xl font-bold tracking-wide">3. Get a Verdict</h3>
                  <p className="mt-2 text-muted-foreground tracking-wide">
                    Receive a detailed verdict on the claim's accuracy. We provide a clear rating and a breakdown of our
                    findings to keep you informed.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="bg-primary/5 py-16 sm:py-24">
          <div className="container mx-auto px-4 sm:px-6 lg:px-8 text-center">
            <div className="max-w-3xl mx-auto animate-in fade-in-0 slide-in-from-bottom-4 duration-700">
              <h2 className="text-3xl sm:text-4xl font-extrabold tracking-wider text-foreground">
                Ready to Seek the Truth?
              </h2>
              <p className="mt-4 text-lg text-muted-foreground tracking-wide">
                Join ClaimGuard today and be a part of the movement against misinformation.
              </p>
              <Link href="/">
                <Button className="mt-8 font-bold tracking-wide h-12 px-6 text-base shadow-lg transform hover:scale-105 transition-all duration-300 animate-pulse-glow">
                  Check Claim
                </Button>
              </Link>
            </div>
          </div>
        </section>

        {/* --- NEW TEAM SECTION --- */}
        <section className="py-16 sm:py-24 bg-background">
          <div className="container mx-auto px-4 sm:px-6 lg:px-8">
            <div className="max-w-4xl mx-auto text-center mb-16 animate-in fade-in-0 slide-in-from-bottom-4 duration-700">
              <h2 className="text-3xl sm:text-4xl font-extrabold tracking-wider text-foreground">
                Meet the Team
              </h2>
              <p className="mt-4 text-lg text-muted-foreground tracking-wide">
                The experts dedicated to fighting misinformation.
              </p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-y-16 gap-x-8 justify-items-center">
              {teamMembers.map((member) => (
                <ProfileCard
                  key={member.handle}
                  name={member.name}
                  title={member.title}
                  handle={member.handle}
                  avatarUrl={member.avatarUrl}
                  miniAvatarUrl={member.miniAvatarUrl}
                  behindGradient="linear-gradient(135deg, #0f172a 0%, #1e293b 100%)"
                  innerGradient="linear-gradient(135deg, #1e293b 0%, #0f172a 100%)"

                  onContactClick={() => alert(`Contacting ${member.name}`)}
                />
              ))}
            </div>
          </div>
        </section>

      </main>

      <Footer />
    </div>
  )
}