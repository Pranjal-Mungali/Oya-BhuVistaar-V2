"use client";

import React from "react";
import { Database, FileCode, Layers2, Compass, LucideIcon } from "lucide-react";

interface DocumentItem {
  title: string;
  url: string;
  icon: LucideIcon;
}

export function NavDocuments() {
  const items: DocumentItem[] = [
    { title: "Sentinel-2 MSI (10m/20m)", url: "#", icon: Layers2 },
    { title: "Landsat-8/9 OLI (30m)", url: "#", icon: Database },
    { title: "GeoTIFF Metadata Registry", url: "#", icon: FileCode },
    { title: "GIS & QGIS Integration", url: "#", icon: Compass },
  ];

  return (
    <div className="pt-4">
      <div className="px-3 pb-2 text-[11px] font-semibold uppercase tracking-wider text-[#71717a]">
        Geospatial Assets
      </div>
      <div className="space-y-1">
        {items.map((item) => {
          const Icon = item.icon;
          return (
            <a
              key={item.title}
              href={item.url}
              className="flex items-center gap-3 px-3 py-2 rounded-lg text-xs text-[#a1a1aa] hover:bg-[#18181b] hover:text-[#f4f4f5] transition-colors"
            >
              <Icon className="w-3.5 h-3.5 shrink-0" />
              <span>{item.title}</span>
            </a>
          );
        })}
      </div>
    </div>
  );
}
