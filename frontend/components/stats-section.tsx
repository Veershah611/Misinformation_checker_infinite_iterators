"use client"

import { AnimatedCounter } from "./animated-counter"
import { useEffect, useState } from "react"

export function StatsSection() {
  const [isVisible, setIsVisible] = useState(false)

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true)
        }
      },
      { threshold: 0.1 },
    )

    const element = document.getElementById("stats-section")
    if (element) {
      observer.observe(element)
    }

    return () => observer.disconnect()
  }, [])

  return (
    <section id="stats-section" className="py-16 bg-primary/5">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 text-center">
          <div className="animate-bounce-in animate-delay-100">
            <div className="text-4xl font-bold text-primary mb-2">
              {isVisible && <AnimatedCounter end={1000} suffix="+" />}
            </div>
            <p className="text-muted-foreground">Claims Verified</p>
          </div>
          <div className="animate-bounce-in animate-delay-200">
            <div className="text-4xl font-bold text-primary mb-2">
              {isVisible && <AnimatedCounter end={95} suffix="%" />}
            </div>
            <p className="text-muted-foreground">Accuracy Rate</p>
          </div>
          <div className="animate-bounce-in animate-delay-300">
            <div className="text-4xl font-bold text-primary mb-2">
              {isVisible && <AnimatedCounter end={24} suffix="/7" />}
            </div>
            <p className="text-muted-foreground">Available</p>
          </div>
        </div>
      </div>
    </section>
  )
}
