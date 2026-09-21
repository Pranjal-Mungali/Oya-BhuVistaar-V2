"use client";

import React from "react";
import {
  LayoutDashboard,
  Layers,
  BarChart3,
  Database,
  LucideIcon,
} from "lucide-react";

interface NavItem {
  title: string;
  targetId: string;
  icon: LucideIcon;
}

export function NavMain() {
  const items: NavItem[] = [
    { title: "Overview & Metrics", targetId: "overview-section", icon: LayoutDashboard },
    { title: "Super Resolution Studio", targetId: "satellite-studio-section", icon: Layers },
    { title: "Spectral Band Telemetry", targetId: "telemetry-section", icon: BarChart3 },
    { title: "Scene Catalog", targetId: "datasets-section", icon: Database },
  ];

  const handleScroll = (targetId: string) => {
    const el = document.getElementById(targetId);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  };

  return (
    <div className="space-y-1">
      {items.map((item) => {
        const Icon = item.icon;
        return (
          <button
            key={item.title}
            onClick={() => handleScroll(item.targetId)}
            className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-xs font-medium text-[#a1a1aa] hover:bg-[#18181b] hover:text-[#f4f4f5] transition-colors text-left"
          >
            <Icon className="w-4 h-4 shrink-0 text-[#71717a]" />
            <span>{item.title}</span>
          </button>
        );
      })}
    </div>
  );
}
