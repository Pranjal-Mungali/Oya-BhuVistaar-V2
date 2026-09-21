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
      sampleKey: "sentinel2_rgb",
      name: "Sentinel-2 True Color RGB",
      sensor: "MSI Level-2A",
      bands: "B4 (Red), B3 (Green), B2 (Blue)",
      format: "GeoTIFF (uint16)",
      crs: "EPSG:4326 (WGS 84)",
      psnr: "--",
      ssim: "--",
      status: "Unprocessed",
    },
    {
      id: "SCENE-002",
      sampleKey: "sentinel2_nir",
      name: "Sentinel-2 Multispectral RGB+NIR",
      sensor: "MSI Level-2A",
      bands: "B4, B3, B2, B8 (NIR)",
      format: "GeoTIFF (uint16)",
      crs: "EPSG:4326 (WGS 84)",
      psnr: "--",
      ssim: "--",
      status: "Unprocessed",
    },
    {
      id: "SCENE-003",
      sampleKey: "urban_optical",
      name: "Urban Optical High-Contrast",
      sensor: "Optical Airborne",
      bands: "RGB (3-Band)",
      format: "PNG (uint8)",
      crs: "Non-projected",
      psnr: "--",
      ssim: "--",
      status: "Unprocessed",
    },
  ];

  const data = records || defaultRecords;

  return (
    <div className="bg-[#141416] border border-[#27272a] rounded-xl overflow-hidden shadow-sm">
      <div className="px-6 py-4 border-b border-[#27272a] flex items-center justify-between">
        <div>
          <h3 className="text-base font-semibold text-[#f4f4f5]">Earth Observation Datasets</h3>
          <p className="text-xs text-[#71717a] mt-0.5">
            Geospatial scenes available for 2x super-resolution inference and uncertainty quantification
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-[#a1a1aa] bg-[#1c1c1f] px-2.5 py-1 rounded-md border border-[#27272a]">
            {data.length} scenes registered
          </span>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead className="bg-[#09090b] text-[#71717a] uppercase tracking-wider font-semibold border-b border-[#27272a]">
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
          <tbody className="divide-y divide-[#1f1f23] text-[#f4f4f5]">
            {data.map((row) => (
              <tr key={row.id} className="hover:bg-[#18181b] transition-colors">
                <td className="px-6 py-3.5 font-medium flex items-center gap-2">
                  <Database className="w-3.5 h-3.5 text-[#a1a1aa]" />
                  <span>{row.name}</span>
                </td>
                <td className="px-6 py-3.5 text-[#a1a1aa]">{row.sensor}</td>
                <td className="px-6 py-3.5 text-[#a1a1aa]">{row.bands}</td>
                <td className="px-6 py-3.5">
                  <span className="px-2 py-0.5 rounded bg-[#1c1c1f] border border-[#27272a] text-[#a1a1aa]">
                    {row.format}
                  </span>
                </td>
                <td className="px-6 py-3.5 text-[#a1a1aa]">{row.crs}</td>
                <td className="px-6 py-3.5">
                  <span
                    className={`px-2 py-0.5 rounded text-[11px] font-medium border ${
                      row.status === "Processed"
                        ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-400"
                        : "border-[#27272a] bg-[#1c1c1f] text-[#71717a]"
                    }`}
                  >
                    {row.status}
                  </span>
                </td>
                <td className="px-6 py-3.5 font-mono text-[#a1a1aa]">{row.psnr}</td>
                <td className="px-6 py-3.5 font-mono text-[#a1a1aa]">{row.ssim}</td>
                <td className="px-6 py-3.5 text-right">
                  <button
                    onClick={() => onSelectScene && onSelectScene(row.sampleKey || row.id)}
                    className="inline-flex items-center gap-1 text-xs text-white bg-[#27272a] hover:bg-[#3f3f46] px-2.5 py-1 rounded transition-colors"
                  >
                    <span>Run SR</span>
                    <ArrowUpRight className="w-3 h-3" />
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
