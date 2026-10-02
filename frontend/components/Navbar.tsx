"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Music2, Menu, X } from "lucide-react";
import { useEffect, useState } from "react";

const links = [
  { label: "Classify", href: "/" },
  { label: "Spectrogram Lab", href: "/spectrogram-lab" },
  { label: "Explainability", href: "/explainability" },
  { label: "Model", href: "/model" },
  { label: "Performance", href: "/performance" },
  { label: "History", href: "/history" },
];

export default function Navbar() {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);
  const [logoClicks, setLogoClicks] = useState(0);
  const [showSunflowers, setShowSunflowers] = useState(false);

  useEffect(() => {
    if (!showSunflowers) return;
    const timer = window.setTimeout(() => setShowSunflowers(false), 1500);
    return () => window.clearTimeout(timer);
  }, [showSunflowers]);

  const handleLogoClick = () => {
    setLogoClicks((count) => {
      const next = count + 1;
      if (next >= 5) {
        setShowSunflowers(true);
        return 0;
      }
      return next;
    });
  };

  return (
    <header className="border-b border-black/10 bg-[#F4F1EE]">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-5 lg:px-10">
        <Link
          href="/"
          onClick={() => setMobileOpen(false)}
          className="flex items-center gap-3"
        >
          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-[#C96F70] text-white">
            <Music2 size={19} />
          </div>

          <div className="relative">
            <button type="button" onClick={handleLogoClick} className="text-xl font-bold tracking-tight" aria-label="GENGROGRAM">
              GENGROGRAM
            </button>
            {showSunflowers && (
              <div className="pointer-events-none absolute left-1/2 top-1/2 z-20 h-10 w-32 -translate-x-1/2 -translate-y-1/2" aria-hidden="true">
                <span className="sunflower-pop absolute left-0 top-2"><span className="sunflower-petals" /><span className="sunflower-center" /></span>
                <span className="sunflower-pop sunflower-pop-delay absolute left-1/2 top-0"><span className="sunflower-petals" /><span className="sunflower-center" /></span>
                <span className="sunflower-pop absolute right-0 top-2"><span className="sunflower-petals" /><span className="sunflower-center" /></span>
              </div>
            )}
          </div>
        </Link>

        {/* Desktop navigation */}
        <nav className="hidden items-center gap-6 lg:flex">
          {links.map((link) => {
            const active =
              link.href === "/"
                ? pathname === "/"
                : pathname.startsWith(link.href);

            return (
              <Link
                key={link.href}
                href={link.href}
                className={`font-mono text-[11px] uppercase tracking-wider transition ${active
                  ? "text-[#161616]"
                  : "text-[#77716E] hover:text-[#161616]"
                  }`}
              >
                {link.label}
              </Link>
            );
          })}
        </nav>

        {/* Mobile menu button */}
        <button
          onClick={() => setMobileOpen((open) => !open)}
          className="flex h-10 w-10 items-center justify-center rounded-xl border border-black/10 bg-white lg:hidden"
          aria-label="Toggle navigation"
        >
          {mobileOpen ? (
            <X size={19} />
          ) : (
            <Menu size={19} />
          )}
        </button>
      </div>

      {/* Mobile navigation */}
      {mobileOpen && (
        <div className="border-t border-black/10 px-6 pb-5 pt-3 lg:hidden">
          <nav className="flex flex-col">
            {links.map((link) => {
              const active =
                link.href === "/"
                  ? pathname === "/"
                  : pathname.startsWith(link.href);

              return (
                <Link
                  key={link.href}
                  href={link.href}
                  onClick={() => setMobileOpen(false)}
                  className={`border-b border-black/5 py-4 font-mono text-xs uppercase tracking-wider ${active
                    ? "text-[#161616]"
                    : "text-[#77716E]"
                    }`}
                >
                  {link.label}
                </Link>
              );
            })}
          </nav>
        </div>
      )}
    </header>
  );
}
