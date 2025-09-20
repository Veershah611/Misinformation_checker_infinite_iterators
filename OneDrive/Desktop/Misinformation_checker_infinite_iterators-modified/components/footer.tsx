export function Footer() {
  return (
    <footer className="bg-background border-t border-border">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex flex-wrap items-center justify-center gap-6">
            <a
              className="text-sm text-muted-foreground hover:text-foreground transition-colors duration-300 tracking-wide"
              href="#privacy"
            >
              Privacy Policy
            </a>
            <a
              className="text-sm text-muted-foreground hover:text-foreground transition-colors duration-300 tracking-wide"
              href="#terms"
            >
              Terms of Service
            </a>
            <a
              className="text-sm text-muted-foreground hover:text-foreground transition-colors duration-300 tracking-wide"
              href="#contact"
            >
              Contact Us
            </a>
          </div>
          <p className="text-sm text-muted-foreground tracking-wide">© 2024 ClaimGuard. All rights reserved.</p>
        </div>
      </div>
    </footer>
  )
}
