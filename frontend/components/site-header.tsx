"use client";

import React from "react";
import { PanelLeft } from "lucide-react";

interface SiteHeaderProps {
  title?: string;
  onToggleSidebar?: () => void;
}

export function SiteHeader({
  title = "Satellite Super-Resolution & Epistemic Uncertainty Dashboard",
  onToggleSidebar,
}: SiteHeaderProps) {
  return (
    <header className="h-16 border-b border-[#21293a] bg-[#0c0f16]/80 backdrop-blur-lg sticky top-0 z-40 px-6 flex items-center justify-between">
      <div className="flex items-center gap-3.5">
        <button
          onClick={onToggleSidebar}
          className="p-2 rounded-full hover:bg-[#1a2130] text-[#94a3b8] hover:text-[#4fe0cd] transition-all border border-[#273247] active:scale-95 shrink-0"
          title="Toggle Navigation"
        >
          <PanelLeft className="w-4 h-4" />
        </button>
        <div className="h-5 w-px bg-[#242d40]" />
        
        {/* Clearly Visible BhuVistaar Brand Logo */}
        <img
          src="/logo-dark.png"
          alt="BhuVistaar Logo"
          className="h-9 w-auto max-w-[180px] object-contain"
        />
        
        <div className="hidden md:block h-4 w-px bg-[#242d40]" />
        <span className="hidden md:inline-block text-xs sm:text-sm font-bold tracking-tight text-white">
          {title}
        </span>
      </div>
    </header>
  );
}
