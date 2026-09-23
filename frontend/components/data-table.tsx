"use client";

import React from "react";
import { Database, ArrowUpRight } from "lucide-react";

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
  onSelectScene?: (sampleKey: string) => void;
}

export function DataTable({ records, onSelectScene }: DataTableProps) {
  const defaultRecords: SceneRecord[] = [
    {
      id: "SCENE-001",
      sampleKey: "sentinel2_nir",
      name: "Sentinel-2 Multispectral RGB+NIR (4-Band)",
      sensor: "ESA MSI Level-2A",
      bands: "B4 (Red), B3 (Green), B2 (Blue), B8 (NIR)",
      format: "GeoTIFF (uint16)",
      crs: "EPSG:4326 (WGS 84)",
      psnr: "--",
      ssim: "--",
      status: "Unprocessed",
    },
    {
      id: "SCENE-002",
      sampleKey: "sentinel2_rgb",
      name: "Sentinel-2 True Color RGB (3-Band)",
      sensor: "ESA MSI Level-2A",
      bands: "B4 (Red), B3 (Green), B2 (Blue)",
      format: "GeoTIFF (uint16)",
      crs: "EPSG:4326 (WGS 84)",
      psnr: "--",
      ssim: "--",
      status: "Unprocessed",
    },
  ];

  const data = records || defaultRecords;

  return (
    <div className="bg-[#0e1524]/75 backdrop-blur-md border border-white/[0.08] rounded-2xl overflow-hidden shadow-[0_8px_30px_rgba(0,0,0,0.3)]">
      <div className="px-6 py-4 border-b border-white/[0.06] flex items-center justify-between">
        <div>
          <h3 className="text-base font-bold text-slate-100">Earth Observation Datasets</h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Geospatial scenes available for super-resolution inference and uncertainty quantification
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-400 bg-[#070b13]/85 px-2.5 py-1 rounded-lg border border-white/[0.07]">
            {data.length} scenes registered
          </span>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead className="bg-[#070b13]/90 text-slate-400 uppercase tracking-wider font-semibold border-b border-white/[0.06]">
            <tr>
              <th className="px-6 py-3">Scene Name</th>
              <th className="px-6 py-3">Sensor / Platform</th>
              <th className="px-6 py-3">Bands</th>
              <th className="px-6 py-3">Format & Bit Depth</th>
              <th className="px-6 py-3">CRS</th>
              <th className="px-6 py-3">Status</th>
              <th className="px-6 py-3">PSNR (vs Bicubic)</th>
              <th className="px-6 py-3">SSIM</th>
              <th className="px-6 py-3 text-right">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-white/[0.05] text-slate-200">
            {data.map((row) => (
              <tr key={row.id} className="hover:bg-white/[0.03] transition-colors">
                <td className="px-6 py-3.5 font-medium flex items-center gap-2">
                  <Database className="w-3.5 h-3.5 text-slate-400" />
                  <span>{row.name}</span>
                </td>
                <td className="px-6 py-3.5 text-slate-400">{row.sensor}</td>
                <td className="px-6 py-3.5 text-slate-400">{row.bands}</td>
                <td className="px-6 py-3.5">
                  <span className="px-2 py-0.5 rounded-md bg-[#0e131d] border border-[#1c2434] text-slate-400 font-mono text-[11px]">
                    {row.format}
                  </span>
                </td>
                <td className="px-6 py-3.5 text-slate-400">{row.crs}</td>
                <td className="px-6 py-3.5">
                  <span
                    className={`px-2 py-0.5 rounded-full text-[11px] font-semibold border ${
                      row.status === "Processed"
                        ? "border-emerald-500/25 bg-emerald-500/10 text-emerald-400"
                        : "border-[#1c2434] bg-[#0e131d] text-slate-500"
                    }`}
                  >
                    {row.status}
                  </span>
                </td>
                <td className="px-6 py-3.5 font-mono text-slate-400">{row.psnr}</td>
                <td className="px-6 py-3.5 font-mono text-slate-400">{row.ssim}</td>
                <td className="px-6 py-3.5 text-right">
                  <button
                    onClick={() => onSelectScene && onSelectScene(row.sampleKey || row.id)}
                    className="inline-flex items-center gap-1.5 text-xs font-semibold text-teal-300 bg-teal-500/10 hover:bg-teal-500 hover:text-slate-950 border border-teal-500/25 px-3 py-1.5 rounded-lg transition-all shadow-sm active:scale-95 group"
                  >
                    <span>Run SR</span>
                    <ArrowUpRight className="w-3.5 h-3.5 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
