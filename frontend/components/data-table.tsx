"use client";

import React from "react";
import { Database, CheckCircle2, Clock } from "lucide-react";

export interface SceneRecord {
  id: string;
  sampleKey?: string;
  name: string;
  sensor: string;
  bands: string;
  format: string;
  crs: string;
  psnr: string;
  ssim: string;
  status: "Unprocessed" | "Processed" | "Archived" | "Pending";
}

interface DataTableProps {
  records?: SceneRecord[];
}

export function DataTable({ records }: DataTableProps) {
  const data = records || [];

  return (
    <div className="bg-[#121215] border border-zinc-800 rounded-2xl overflow-hidden shadow-[0_4px_24px_rgba(0,0,0,0.4)]">
      <div className="px-6 py-4 border-b border-zinc-800 flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold text-zinc-100">Session History</h3>
          <p className="text-xs text-zinc-400 mt-0.5">
            Record of satellite imagery processed during this session
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-zinc-400 bg-[#0d0d0f] px-2.5 py-1 rounded-lg border border-zinc-800 font-mono">
            {data.length} {data.length === 1 ? "scene" : "scenes"}
          </span>
        </div>
      </div>

      <div className="overflow-x-auto">
        {data.length === 0 ? (
          <div className="py-12 text-center text-xs text-zinc-500 space-y-1">
            <p className="text-zinc-400 font-medium">No processed scenes yet</p>
            <p>Uploaded satellite images will appear here with their verified quality metrics.</p>
          </div>
        ) : (
          <table className="w-full text-left text-xs">
            <thead className="bg-[#0d0d0f] text-zinc-400 uppercase tracking-wider font-semibold border-b border-zinc-800 text-[11px]">
              <tr>
                <th className="px-6 py-3">Scene Name</th>
                <th className="px-6 py-3">Type</th>
                <th className="px-6 py-3">Bands</th>
                <th className="px-6 py-3">Format</th>
                <th className="px-6 py-3">CRS</th>
                <th className="px-6 py-3">Status</th>
                <th className="px-6 py-3">PSNR</th>
                <th className="px-6 py-3">SSIM</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800/80 text-zinc-200">
              {data.map((row) => (
                <tr key={row.id} className="hover:bg-zinc-800/30 transition-colors">
                  <td className="px-6 py-3.5 font-medium flex items-center gap-2">
                    <Database className="w-3.5 h-3.5 text-zinc-500" />
                    <span className="truncate max-w-[220px] text-zinc-100">{row.name}</span>
                  </td>
                  <td className="px-6 py-3.5 text-zinc-400">{row.sensor}</td>
                  <td className="px-6 py-3.5 text-zinc-300 font-mono text-[11px]">{row.bands}</td>
                  <td className="px-6 py-3.5 text-zinc-400">{row.format}</td>
                  <td className="px-6 py-3.5 text-zinc-400 font-mono text-[11px]">{row.crs}</td>
                  <td className="px-6 py-3.5">
                    {row.status === "Processed" ? (
                      <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[11px] font-medium bg-teal-500/10 text-teal-300 border border-teal-500/20">
                        <CheckCircle2 className="w-3 h-3 text-teal-400" />
                        Processed
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[11px] font-medium bg-zinc-800 text-zinc-400 border border-zinc-700">
                        <Clock className="w-3 h-3 text-zinc-400" />
                        Ready
                      </span>
                    )}
                  </td>
                  <td className="px-6 py-3.5 font-mono text-teal-400 font-medium">{row.psnr}</td>
                  <td className="px-6 py-3.5 font-mono text-teal-400 font-medium">{row.ssim}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
