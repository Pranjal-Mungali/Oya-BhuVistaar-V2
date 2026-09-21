"use client";

import React from "react";
import { Plus } from "lucide-react";
import { NavMain } from "./nav-main";

interface AppSidebarProps {
  onQuickProcess?: () => void;
}

export function AppSidebar({ onQuickProcess }: AppSidebarProps) {
  return (
    <aside className="w-64 shrink-0 bg-[#09090b] border-r border-[#27272a] flex flex-col justify-between p-4 h-screen sticky top-0 overflow-y-auto">
      <div className="space-y-4">
        {/* Brand Header with Official Logo */}
        <div className="px-2 py-1">
          <img
            src="/logo-dark.png"
            alt="BhuVistaar Logo"
            className="h-10 w-auto max-w-[200px] object-contain"
          />
        </div>

        {/* Action Button */}
        <div className="pt-1">
          <button
            onClick={onQuickProcess}
            className="w-full bg-white hover:bg-zinc-200 text-black font-semibold text-xs py-2.5 px-3 rounded-lg flex items-center justify-center gap-2 transition-colors shadow-sm"
          >
            <Plus className="w-4 h-4 stroke-[2.5]" />
            <span>Process Satellite Scene</span>
          </button>
        </div>

        {/* Functional Navigation */}
        <div className="pt-2">
          <div className="px-3 pb-2 text-[10px] font-semibold uppercase tracking-wider text-[#52525b]">
            Navigation
          </div>
          <NavMain />
        </div>
      </div>
    </aside>
  );
}
