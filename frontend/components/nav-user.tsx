"use client";

import React from "react";
import { Sparkles, ChevronsUpDown } from "lucide-react";

export function NavUser() {
  return (
    <div className="flex items-center justify-between p-2 rounded-lg bg-[#141416] border border-[#27272a] text-[#f4f4f5]">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-emerald-500 to-cyan-500 flex items-center justify-center font-bold text-xs text-black">
          BV
        </div>
        <div className="text-left">
          <div className="text-xs font-semibold text-[#f4f4f5] leading-tight">BhuVistaar AI</div>
          <div className="text-[11px] text-[#71717a] leading-tight">admin@bhuvistaar.ai</div>
        </div>
      </div>
      <ChevronsUpDown className="w-4 h-4 text-[#71717a]" />
    </div>
  );
}
