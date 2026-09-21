"use client";

import React from "react";
import { LifeBuoy, Send, LucideIcon } from "lucide-react";

interface SecondaryItem {
  title: string;
  url: string;
  icon: LucideIcon;
}

export function NavSecondary({ className }: { className?: string }) {
  const items: SecondaryItem[] = [
    { title: "Documentation", url: "https://github.com", icon: LifeBuoy },
    { title: "Feedback", url: "#", icon: Send },
  ];

  return (
    <div className={`space-y-1 ${className}`}>
      {items.map((item) => {
        const Icon = item.icon;
        return (
          <a
            key={item.title}
            href={item.url}
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-3 px-3 py-1.5 rounded-lg text-xs text-[#71717a] hover:text-[#f4f4f5] transition-colors"
          >
            <Icon className="w-3.5 h-3.5 shrink-0" />
            <span>{item.title}</span>
          </a>
        );
      })}
    </div>
  );
}
