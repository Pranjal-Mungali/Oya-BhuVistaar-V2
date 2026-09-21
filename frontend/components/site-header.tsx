"use client";

import React from "react";
import { PanelLeft, Satellite, Cpu, CheckCircle2 } from "lucide-react";

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
      <div className="flex items-center gap-3">
        <button
          onClick={onToggleSidebar}
          className="p-2 rounded-full hover:bg-[#1a2130] text-[#94a3b8] hover:text-[#4fe0cd] transition-all border border-[#273247] active:scale-95"
          title="Toggle Navigation"
        >
          <PanelLeft className="w-4 h-4" />
        </button>
        <div className="h-4 w-px bg-[#242d40]" />
        <span className="text-xs sm:text-sm font-bold tracking-tight text-white">{title}</span>
      </div>

      <div className="hidden md:flex items-center gap-2.5">
        <div className="px-3 py-1 rounded-full bg-[#162522] border border-[#234c41] text-[11px] font-semibold text-[#4fe0cd] flex items-center gap-1.5 shadow-sm">
          <CheckCircle2 className="w-3.5 h-3.5" />
          <span>Trained Weights Active</span>
        </div>
        <div className="px-3 py-1 rounded-full bg-[#172033] border border-[#2a3c61] text-[11px] font-semibold text-[#93c5fd] flex items-center gap-1.5 shadow-sm">
          <Cpu className="w-3.5 h-3.5" />
          <span>NVIDIA RTX 5050 / CUDA</span>
        </div>
      </div>
    </header>
  );
}
