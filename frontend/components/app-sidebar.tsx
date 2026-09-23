"use client";

import React from "react";
import { Plus } from "lucide-react";
import { NavMain } from "./nav-main";

interface AppSidebarProps {
  onQuickProcess?: () => void;
}

export function AppSidebar({ onQuickProcess }: AppSidebarProps) {
  return (
    <aside className="w-64 shrink-0 bg-[#090d15]/90 backdrop-blur-md border-r border-white/[0.06] flex flex-col justify-between p-4 h-screen sticky top-0 overflow-y-auto">
      <div className="space-y-5">
        {/* Brand logo */}
        <div className="px-2 py-1">
          <img
            src="/logo-dark.png"
            alt="BhuVistaar Logo"
            className="h-9 w-auto max-w-[190px] object-contain"
          />
        </div>

        {/* Quick action */}
        <div className="pt-1">
          <button
            onClick={onQuickProcess}
            className="w-full bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold text-xs py-2.5 px-3.5 rounded-xl flex items-center justify-center gap-2 transition-all shadow-sm active:scale-[0.98] cursor-pointer"
          >
            <Plus className="w-4 h-4 stroke-[2.5]" />
            <span>Process Satellite Scene</span>
          </button>
        </div>

        {/* Navigation */}
        <div className="pt-2 space-y-1.5">
          <div className="px-3 text-[10px] font-semibold uppercase tracking-wider text-slate-500">
            Navigation
          </div>
          <NavMain />
        </div>
      </div>
    </aside>
  );
}
