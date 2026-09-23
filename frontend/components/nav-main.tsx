"use client";

import React, { useState, useEffect } from "react";
import {
  LayoutDashboard,
  Layers,
  BarChart3,
  Database,
  LucideIcon,
  ChevronRight,
} from "lucide-react";

interface NavItem {
  title: string;
  targetId: string;
  icon: LucideIcon;
}

export function NavMain() {
  const [activeId, setActiveId] = useState<string>("satellite-studio-section");

  const items: NavItem[] = [
    { title: "Super Resolution Studio", targetId: "satellite-studio-section", icon: Layers },
    { title: "Quality Metrics", targetId: "overview-section", icon: LayoutDashboard },
    { title: "Spectral Band Telemetry", targetId: "telemetry-section", icon: BarChart3 },
    { title: "Scene Catalog", targetId: "datasets-section", icon: Database },
  ];

  // Observe active section on scroll
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setActiveId(entry.target.id);
          }
        });
      },
      { rootMargin: "-20% 0px -60% 0px", threshold: 0 }
    );

    items.forEach((item) => {
      const el = document.getElementById(item.targetId);
      if (el) observer.observe(el);
    });

    return () => observer.disconnect();
  }, []);

  const handleScroll = (targetId: string) => {
    setActiveId(targetId);
    const el = document.getElementById(targetId);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  };

  return (
    <div className="space-y-1.5">
      {items.map((item) => {
        const Icon = item.icon;
        const isActive = activeId === item.targetId;

        return (
          <button
            key={item.title}
            onClick={() => handleScroll(item.targetId)}
            className={`w-full group flex items-center justify-between px-3 py-2 rounded-xl text-xs font-medium transition-all text-left ${
              isActive
                ? "bg-teal-500/10 text-teal-300 border-l-2 border-teal-400 pl-2.5 font-semibold"
                : "text-slate-400 hover:text-slate-200 hover:bg-[#151c29]/60"
            }`}
          >
            <div className="flex items-center gap-2.5">
              <Icon
                className={`w-4 h-4 shrink-0 transition-colors ${
                  isActive ? "text-teal-400" : "text-slate-500 group-hover:text-slate-300"
                }`}
              />
              <span>{item.title}</span>
            </div>

            {isActive ? (
              <span className="w-1.5 h-1.5 rounded-full bg-teal-400" />
            ) : (
              <ChevronRight className="w-3 h-3 text-slate-600 opacity-0 group-hover:opacity-100 transition-opacity" />
            )}
          </button>
        );
      })}
    </div>
  );
}
