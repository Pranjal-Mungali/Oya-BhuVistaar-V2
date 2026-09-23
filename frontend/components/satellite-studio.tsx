"use client";

import React, { useState, useRef, useEffect } from "react";
import {
  Upload,
  Play,
  Download,
  AlertCircle,
  Eye,
  Sliders,
  Sparkles,
  ArrowRight,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Move,
  FileCheck,
  CheckCircle2,
  X,
  Layers,
  Activity
} from "lucide-react";

interface SatelliteStudioProps {
  onMetricsUpdate?: (metrics: any, sceneName?: string) => void;
  onInferenceComplete?: (data: any, sceneName?: string) => void;
  selectedSceneKey?: string | null;
}

export function SatelliteStudio({ onMetricsUpdate, onInferenceComplete, selectedSceneKey }: SatelliteStudioProps) {
  const [selectedSample, setSelectedSample] = useState<string | null>(selectedSceneKey || "sentinel2_nir");
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [isDragOver, setIsDragOver] = useState<boolean>(false);
  const [numPasses, setNumPasses] = useState<number>(15);
  const [colormap, setColormap] = useState<string>("turbo");
  const [activeViewTab, setActiveViewTab] = useState<string>("curtain");
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<any | null>(null);

  // Viewport / Slider State
  const [curtainPos, setCurtainPos] = useState<number>(50); // 0 - 100 percentage
  const [zoom, setZoom] = useState<number>(1);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [interactionMode, setInteractionMode] = useState<"curtain" | "pan">("curtain");
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const dragStartRef = useRef<{ clientX: number; clientY: number; panX: number; panY: number } | null>(null);
  const [containerSize, setContainerSize] = useState<{ width: number; height: number }>({ width: 0, height: 0 });

  const curtainRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Measure and track viewport container size
  useEffect(() => {
    if (!curtainRef.current) return;
    const updateDimensions = () => {
      if (curtainRef.current) {
        setContainerSize({
          width: curtainRef.current.clientWidth,
          height: curtainRef.current.clientHeight,
        });
      }
    };
    updateDimensions();
    const observer = new ResizeObserver(updateDimensions);
    observer.observe(curtainRef.current);
    return () => observer.disconnect();
  }, [result, activeViewTab]);

  // Sync scene selection from outside (e.g. data table click)
  useEffect(() => {
    if (selectedSceneKey) {
      setSelectedSample(selectedSceneKey);
      setUploadedFile(null);
    }
  }, [selectedSceneKey]);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      setUploadedFile(file);
      setSelectedSample(null);
      setError(null);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  const handleRunInference = async () => {
    if (!uploadedFile && !selectedSample) {
      setError("Please select a benchmark scene or upload satellite imagery first.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      if (uploadedFile) {
        formData.append("file", uploadedFile);
      } else if (selectedSample) {
        formData.append("sample_id", selectedSample);
      }
      formData.append("num_passes", numPasses.toString());
      formData.append("colormap", colormap);

      const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
      let res;
      try {
        res = await fetch(`${apiBase}/api/predict`, {
          method: "POST",
          body: formData,
        });
      } catch (directErr) {
        res = await fetch("/api/predict", {
          method: "POST",
          body: formData,
        });
      }

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error: ${res.status}`);
      }

      const data = await res.json();
      setResult(data);
      const sceneLabel = uploadedFile ? uploadedFile.name : (selectedSample || "Sentinel-2 Scene");
      if (onInferenceComplete) {
        onInferenceComplete(data, sceneLabel);
      }
      if (onMetricsUpdate && data.metrics) {
        onMetricsUpdate(data.metrics, sceneLabel);
      }
    } catch (err: any) {
      console.error(err);
      setError(
        err.message ||
          "Failed to process image. Make sure the FastAPI backend is running with 'python server.py'"
      );
    } finally {
      setLoading(false);
    }
  };

  const handleCurtainMove = (e: React.MouseEvent<HTMLDivElement> | React.TouchEvent<HTMLDivElement>) => {
    if (!curtainRef.current) return;
    const rect = curtainRef.current.getBoundingClientRect();
    const clientX = "touches" in e ? e.touches[0].clientX : e.clientX;
    const offset = clientX - rect.left;
    const pct = Math.max(0, Math.min(100, (offset / rect.width) * 100));
    setCurtainPos(pct);
  };

  const handleZoomChange = (newZoom: number) => {
    const clampedZoom = Math.max(1, Math.min(4, Math.round(newZoom * 10) / 10));
    setZoom(clampedZoom);
    if (clampedZoom === 1) {
      setPan({ x: 0, y: 0 });
    } else {
      const w = containerSize.width || (curtainRef.current?.clientWidth ?? 768);
      const h = containerSize.height || (curtainRef.current?.clientHeight ?? 480);
      const maxPanX = (w * (clampedZoom - 1)) / 2;
      const maxPanY = (h * (clampedZoom - 1)) / 2;
      setPan((prev) => ({
        x: Math.max(-maxPanX, Math.min(maxPanX, prev.x)),
        y: Math.max(-maxPanY, Math.min(maxPanY, prev.y)),
      }));
    }
  };

  const resetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
    setCurtainPos(50);
    setInteractionMode("curtain");
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (interactionMode === "pan" || e.button === 1 || e.altKey) {
      setIsDragging(true);
      dragStartRef.current = {
        clientX: e.clientX,
        clientY: e.clientY,
        panX: pan.x,
        panY: pan.y,
      };
    } else {
      handleCurtainMove(e);
      setIsDragging(true);
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!isDragging) return;
    if (interactionMode === "pan" || dragStartRef.current) {
      if (!dragStartRef.current) return;
      const dx = e.clientX - dragStartRef.current.clientX;
      const dy = e.clientY - dragStartRef.current.clientY;
      const w = containerSize.width || (curtainRef.current?.clientWidth ?? 768);
      const h = containerSize.height || (curtainRef.current?.clientHeight ?? 480);
      const maxPanX = (w * (zoom - 1)) / 2;
      const maxPanY = (h * (zoom - 1)) / 2;
      setPan({
        x: Math.max(-maxPanX, Math.min(maxPanX, dragStartRef.current.panX + dx)),
        y: Math.max(-maxPanY, Math.min(maxPanY, dragStartRef.current.panY + dy)),
      });
    } else {
      handleCurtainMove(e);
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
    dragStartRef.current = null;
  };

  const handleTouchStart = (e: React.TouchEvent<HTMLDivElement>) => {
    if (e.touches.length === 1) {
      const touch = e.touches[0];
      if (interactionMode === "pan") {
        setIsDragging(true);
        dragStartRef.current = {
          clientX: touch.clientX,
          clientY: touch.clientY,
          panX: pan.x,
          panY: pan.y,
        };
      } else {
        handleCurtainMove(e);
        setIsDragging(true);
      }
    }
  };

  const handleTouchMove = (e: React.TouchEvent<HTMLDivElement>) => {
    if (!isDragging || e.touches.length !== 1) return;
    const touch = e.touches[0];
    if (interactionMode === "pan" && dragStartRef.current) {
      const dx = touch.clientX - dragStartRef.current.clientX;
      const dy = touch.clientY - dragStartRef.current.clientY;
      const w = containerSize.width || (curtainRef.current?.clientWidth ?? 768);
      const h = containerSize.height || (curtainRef.current?.clientHeight ?? 480);
      const maxPanX = (w * (zoom - 1)) / 2;
      const maxPanY = (h * (zoom - 1)) / 2;
      setPan({
        x: Math.max(-maxPanX, Math.min(maxPanX, dragStartRef.current.panX + dx)),
        y: Math.max(-maxPanY, Math.min(maxPanY, dragStartRef.current.panY + dy)),
      });
    } else {
      handleCurtainMove(e);
    }
  };

  const getSceneNameDisplay = () => {
    if (uploadedFile) return uploadedFile.name;
    if (selectedSample === "sentinel2_nir") return "Sentinel-2 RGB+NIR (4-Band)";
    if (selectedSample === "sentinel2_rgb") return "Sentinel-2 True Color (3-Band)";
    return "--";
  };

  const colormapList = [
    { id: "turbo", name: "Turbo", color: "from-blue-500 via-emerald-400 to-red-500" },
    { id: "magma", name: "Magma", color: "from-purple-900 via-red-500 to-amber-300" },
  ];

  return (
    <div className="relative bg-[#0e1524]/75 backdrop-blur-md border border-white/[0.08] rounded-2xl p-5 sm:p-6 md:p-8 space-y-6 md:space-y-7 shadow-[0_12px_40px_rgba(0,0,0,0.35)]">
      {/* Top Section: Imagery Upload & Execution Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 md:gap-6 items-stretch">
        {/* Left Column (7 cols): Prominent Image Upload & Scene Selector */}
        <div className="lg:col-span-7 flex flex-col justify-between space-y-4 bg-[#090e17]/70 backdrop-blur-md border border-white/[0.06] rounded-2xl p-5 md:p-6 shadow-sm">
          <div className="space-y-3">
            <div className="flex items-center justify-between pb-1 border-b border-white/[0.06]">
              <span className="text-xs font-bold text-teal-400 uppercase tracking-wider flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-teal-400" />
                Source Satellite Imagery
              </span>
              <span className="text-[11px] font-mono text-slate-500">Input: 10m GSD</span>
            </div>

            {/* Prominent Upload Dropzone */}
            {!uploadedFile ? (
              <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className={`border-2 border-dashed rounded-2xl p-6 sm:p-7 transition-all duration-200 cursor-pointer flex flex-col items-center justify-center text-center gap-3 group select-none ${
                  isDragOver
                    ? "border-teal-400 bg-teal-500/10 shadow-[0_0_24px_rgba(45,212,191,0.2)]"
                    : "border-white/[0.08] hover:border-teal-500/40 bg-[#070b13]/80 hover:bg-[#0b101c]"
                }`}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".tif,.tiff,.png,.jpg,.jpeg"
                  className="hidden"
                  onChange={(e) => {
                    if (e.target.files && e.target.files[0]) {
                      setUploadedFile(e.target.files[0]);
                      setSelectedSample(null);
                      setError(null);
                    }
                  }}
                />
                <div className="w-12 h-12 rounded-xl bg-teal-500/10 border border-teal-500/25 flex items-center justify-center text-teal-400 group-hover:scale-105 group-hover:bg-teal-500/15 transition-all">
                  <Upload className="w-5 h-5 stroke-[2]" />
                </div>
                <div className="space-y-1">
                  <p className="text-sm font-semibold text-slate-200 group-hover:text-white">
                    Drop satellite imagery here, or <span className="text-teal-400 underline underline-offset-2">browse</span>
                  </p>
                  <p className="text-xs text-slate-400 max-w-sm">
                    Upload raw GeoTIFF (.tif, .tiff) or high-res raster imagery (.png, .jpg)
                  </p>
                </div>
                <div className="flex flex-wrap items-center justify-center gap-2 pt-1">
                  <span className="text-[10px] px-2.5 py-0.5 rounded-md bg-[#111826] border border-white/[0.06] text-slate-400 font-mono">GeoTIFF (uint16)</span>
                  <span className="text-[10px] px-2.5 py-0.5 rounded-md bg-[#111826] border border-white/[0.06] text-slate-400 font-mono">10m Ground Resolution</span>
                  <span className="text-[10px] px-2.5 py-0.5 rounded-md bg-[#111826] border border-white/[0.06] text-slate-400 font-mono">RGB / Multispectral</span>
                </div>
              </div>
            ) : (
              <div className="border border-teal-500/35 bg-teal-500/[0.08] backdrop-blur-sm rounded-2xl p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-4 transition-all">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".tif,.tiff,.png,.jpg,.jpeg"
                  className="hidden"
                  onChange={(e) => {
                    if (e.target.files && e.target.files[0]) {
                      setUploadedFile(e.target.files[0]);
                      setSelectedSample(null);
                      setError(null);
                    }
                  }}
                />
                <div className="flex items-center gap-3.5 min-w-0">
                  <div className="w-12 h-12 rounded-xl bg-teal-500/20 border border-teal-500/30 flex items-center justify-center text-teal-300 shrink-0">
                    <FileCheck className="w-6 h-6" />
                  </div>
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <p className="text-sm font-semibold text-teal-100 truncate max-w-[220px] sm:max-w-[320px]">
                        {uploadedFile.name}
                      </p>
                      <span className="text-[10px] px-2 py-0.5 rounded-full bg-teal-500/25 text-teal-300 font-mono shrink-0">
                        Active Upload
                      </span>
                    </div>
                    <p className="text-xs text-slate-400 font-mono mt-0.5">
                      {formatFileSize(uploadedFile.size)} • Ready for 4x super-resolution
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2 shrink-0 self-end sm:self-auto">
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="px-3 py-1.5 rounded-lg bg-[#141c2a] hover:bg-[#1c273b] border border-white/[0.08] text-xs font-medium text-slate-300 hover:text-white transition-colors cursor-pointer"
                  >
                    Change
                  </button>
                  <button
                    onClick={() => {
                      setUploadedFile(null);
                      setSelectedSample("sentinel2_nir");
                      setError(null);
                    }}
                    className="p-1.5 rounded-lg bg-[#141c2a] hover:bg-red-500/15 border border-white/[0.08] hover:border-red-500/30 text-slate-400 hover:text-red-400 transition-colors cursor-pointer"
                    title="Remove uploaded file and restore benchmark scene"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Benchmark Scene Presets */}
          <div className="space-y-2 pt-1">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-400 font-medium">Or select a benchmark scene:</span>
              <span className="text-[11px] text-slate-500 font-mono">ESA Sentinel-2 MSI</span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
              <button
                onClick={() => {
                  setSelectedSample("sentinel2_nir");
                  setUploadedFile(null);
                  setError(null);
                }}
                className={`text-left p-3.5 rounded-xl border transition-all duration-200 flex flex-col justify-between gap-1.5 cursor-pointer ${
                  selectedSample === "sentinel2_nir" && !uploadedFile
                    ? "border-teal-500/50 bg-teal-500/10 text-teal-200 shadow-sm"
                    : "border-white/[0.06] bg-[#070b13]/70 hover:bg-[#0c121e] text-slate-300 hover:border-white/[0.14]"
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="text-xs font-bold text-white flex items-center gap-1.5">
                    {selectedSample === "sentinel2_nir" && !uploadedFile && (
                      <span className="w-2 h-2 rounded-full bg-teal-400" />
                    )}
                    Sentinel-2 RGB+NIR
                  </span>
                  <span className="text-[10px] px-2 py-0.5 rounded-md bg-[#111826] border border-white/[0.05] text-teal-300 font-mono font-medium">
                    4-Band
                  </span>
                </div>
                <p className="text-[11px] text-slate-400">
                  B4 (Red), B3 (Green), B2 (Blue), B8 (NIR)
                </p>
              </button>

              <button
                onClick={() => {
                  setSelectedSample("sentinel2_rgb");
                  setUploadedFile(null);
                  setError(null);
                }}
                className={`text-left p-3.5 rounded-xl border transition-all duration-200 flex flex-col justify-between gap-1.5 cursor-pointer ${
                  selectedSample === "sentinel2_rgb" && !uploadedFile
                    ? "border-teal-500/50 bg-teal-500/10 text-teal-200 shadow-sm"
                    : "border-white/[0.06] bg-[#070b13]/70 hover:bg-[#0c121e] text-slate-300 hover:border-white/[0.14]"
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="text-xs font-bold text-white flex items-center gap-1.5">
                    {selectedSample === "sentinel2_rgb" && !uploadedFile && (
                      <span className="w-2 h-2 rounded-full bg-teal-400" />
                    )}
                    Sentinel-2 True Color
                  </span>
                  <span className="text-[10px] px-2 py-0.5 rounded-md bg-[#111826] border border-white/[0.05] text-sky-300 font-mono font-medium">
                    3-Band
                  </span>
                </div>
                <p className="text-[11px] text-slate-400">
                  B4 (Red), B3 (Green), B2 (Blue)
                </p>
              </button>
            </div>
          </div>
        </div>

        {/* Right Column (5 cols): Prominent Action Button & Bayesian Inference Configuration */}
        <div className="lg:col-span-5 flex flex-col justify-between space-y-4 bg-[#090e17]/70 backdrop-blur-md border border-white/[0.06] rounded-2xl p-5 md:p-6 shadow-sm">
          <div className="space-y-4">
            <div className="flex items-center justify-between pb-1 border-b border-white/[0.06]">
              <span className="text-xs font-bold text-sky-400 uppercase tracking-wider flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-sky-400" />
                Execution & Parameters
              </span>
              <span className="text-[11px] font-mono text-slate-500">Output: 2.5m GSD</span>
            </div>

            {/* Prominent Primary Call-to-Action */}
            <button
              onClick={handleRunInference}
              disabled={loading}
              className="w-full bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold text-sm px-6 py-4 rounded-xl flex items-center justify-center gap-3 transition-all shadow-md shadow-teal-500/10 hover:shadow-teal-500/20 active:scale-[0.98] disabled:opacity-50 disabled:pointer-events-none cursor-pointer"
            >
              {loading ? (
                <>
                  <div className="w-5 h-5 border-2 border-slate-950 border-t-transparent rounded-full animate-spin" />
                  <span>Running {numPasses} Monte Carlo Passes...</span>
                </>
              ) : (
                <>
                  <div className="w-6 h-6 rounded-lg bg-slate-950/15 flex items-center justify-center">
                    <Play className="w-3.5 h-3.5 fill-current ml-0.5" />
                  </div>
                  <span>Generate Super-Resolution</span>
                  <ArrowRight className="w-4 h-4 opacity-75" />
                </>
              )}
            </button>

            {/* Active Selection Status */}
            <div className="flex items-center justify-between text-xs px-1">
              <span className="text-slate-400">Target Scene:</span>
              <span className="text-teal-300 font-medium font-mono truncate max-w-[200px]">
                {getSceneNameDisplay()}
              </span>
            </div>

            {/* Stochastic Monte Carlo Passes Slider */}
            <div className="pt-1 space-y-2">
              <div className="flex justify-between items-center text-xs font-medium">
                <span className="text-slate-400">Stochastic MC Passes:</span>
                <span className="px-2.5 py-0.5 rounded-full bg-[#182336] text-sky-300 font-mono font-bold text-xs border border-sky-500/20">
                  {numPasses} passes
                </span>
              </div>
              <input
                type="range"
                min={5}
                max={30}
                step={1}
                value={numPasses}
                onChange={(e) => setNumPasses(parseInt(e.target.value))}
                className="w-full h-2 bg-[#1d273a] rounded-lg appearance-none cursor-pointer accent-teal-400"
              />
              <div className="flex justify-between text-[11px] text-slate-500">
                <span>Fast (5)</span>
                <span className="text-teal-400 font-medium">Recommended (15)</span>
                <span>Deep (30)</span>
              </div>
            </div>

            {/* Uncertainty Colormap */}
            <div className="pt-1 space-y-2">
              <span className="text-xs text-slate-400 block font-medium">Uncertainty Colormap:</span>
              <div className="grid grid-cols-2 gap-2">
                {colormapList.map((cm) => (
                  <button
                    key={cm.id}
                    onClick={() => setColormap(cm.id)}
                    className={`px-3 py-2 rounded-xl text-xs font-semibold flex items-center justify-center gap-2 border transition-all duration-150 cursor-pointer ${
                      colormap === cm.id
                        ? "border-teal-500/40 bg-teal-500/10 text-teal-200"
                        : "border-white/[0.06] bg-[#070b13]/70 text-slate-400 hover:bg-[#0c121e] hover:text-slate-200"
                    }`}
                  >
                    <span className={`w-3 h-3 rounded-full bg-gradient-to-r ${cm.color} shrink-0`} />
                    <span>{cm.name}</span>
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Output Telemetry & Download CTA if ready */}
          <div className="pt-2 border-t border-white/[0.06] space-y-2.5">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-400">Enhancement:</span>
              <span className="px-2 py-0.5 rounded-md bg-teal-500/10 text-teal-300 font-semibold border border-teal-500/20 text-[11px]">
                4x SR (10m → 2.5m GSD)
              </span>
            </div>

            {result && (
              <a
                href={`${process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000"}${result.download_url}`}
                target="_blank"
                download
                className="w-full bg-sky-500/10 hover:bg-sky-500/20 border border-sky-500/30 text-sky-300 font-semibold text-xs py-2.5 px-4 rounded-xl flex items-center justify-center gap-2 transition-all shadow-sm active:scale-[0.98] cursor-pointer"
              >
                <Download className="w-3.5 h-3.5 text-sky-400" />
                <span>Download Enhanced GeoTIFF ({result.filename})</span>
              </a>
            )}
          </div>
        </div>
      </div>

      {error && (
        <div className="p-4 bg-red-950/30 border border-red-800/40 rounded-xl text-xs sm:text-sm text-red-300 flex items-center gap-3 animate-in fade-in duration-200">
          <AlertCircle className="w-4 h-4 shrink-0 text-red-400" />
          <span className="font-medium">{error}</span>
        </div>
      )}

      {/* Visual Results Section: Viewport Controls & Canvas with ALL TABS */}
      <div className="space-y-4 pt-2 border-t border-white/[0.06]">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 pb-1">
          <div className="inline-flex p-1 rounded-xl bg-[#070b13]/85 backdrop-blur-sm border border-white/[0.07] overflow-x-auto max-w-full shadow-inner">
            <button
              onClick={() => setActiveViewTab("curtain")}
              className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all whitespace-nowrap cursor-pointer ${
                activeViewTab === "curtain"
                  ? "bg-teal-500/15 text-teal-300 border border-teal-500/30 shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Curtain View
            </button>
            <button
              onClick={() => setActiveViewTab("split")}
              className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all whitespace-nowrap cursor-pointer ${
                activeViewTab === "split"
                  ? "bg-teal-500/15 text-teal-300 border border-teal-500/30 shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Side-by-Side
            </button>
            <button
              onClick={() => setActiveViewTab("uncertainty")}
              className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all whitespace-nowrap cursor-pointer ${
                activeViewTab === "uncertainty"
                  ? "bg-teal-500/15 text-teal-300 border border-teal-500/30 shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Uncertainty Map
            </button>
          </div>

          {result && (
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <span className="px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-400 font-semibold border border-emerald-500/20 font-mono">
                PSNR: {result.metrics.psnr_db} dB
              </span>
              <span className="px-3 py-1 rounded-full bg-sky-500/10 text-sky-400 font-semibold border border-sky-500/20 font-mono">
                SSIM: {result.metrics.ssim}
              </span>
              <span className="px-3 py-1 rounded-full bg-amber-500/10 text-amber-400 font-semibold border border-amber-500/20 font-mono">
                Gain: {result.metrics.sharpness_gain_ratio}x
              </span>
              <span className="px-3 py-1 rounded-full bg-teal-500/10 text-teal-300 font-semibold border border-teal-500/20 font-mono">
                Mean &sigma;: {result.metrics.mean_uncertainty}
              </span>
            </div>
          )}
        </div>

        {/* Viewport Display Box */}
        <div className="bg-[#060910]/80 backdrop-blur-md border border-white/[0.07] rounded-2xl p-5 md:p-6 min-h-[440px] md:min-h-[480px] flex items-center justify-center relative overflow-hidden shadow-inner">
          {!result && !loading && (
            <div className="text-center space-y-3.5 text-slate-500 py-16">
              <div className="w-12 h-12 rounded-xl bg-[#111723] border border-[#1e283b] flex items-center justify-center mx-auto text-teal-400 shadow-sm">
                <Eye className="w-6 h-6 stroke-1.5" />
              </div>
              <div className="space-y-1">
                <p className="text-sm font-semibold text-slate-200">No active inference output</p>
                <p className="text-xs sm:text-sm max-w-md mx-auto text-slate-400">
                  Select a satellite scene or upload your own imagery above, then click{" "}
                  <span className="text-teal-400 font-semibold">"Generate Super-Resolution"</span>.
                </p>
              </div>
            </div>
          )}

          {loading && (
            <div className="text-center space-y-4 py-20">
              <div className="w-10 h-10 border-3 border-teal-400 border-t-transparent rounded-full animate-spin mx-auto shadow-sm" />
              <div className="space-y-1">
                <div className="text-sm font-semibold text-slate-200">
                  Executing {numPasses}-Pass Monte Carlo Inference...
                </div>
                <p className="text-xs text-slate-400 max-w-sm mx-auto">
                  Stochastic weights sampling & epistemic uncertainty quantification on GPU
                </p>
              </div>
            </div>
          )}

          {result && !loading && (
            <div className="w-full flex flex-col items-center">
              {/* TAB 1: CURTAIN VIEW */}
              {activeViewTab === "curtain" && (
                <div className="w-full flex flex-col items-center space-y-3.5">
                  {/* Zoom & Inspection Toolbar */}
                  <div className="flex flex-wrap items-center justify-between gap-3 w-full max-w-3xl bg-[#090e17]/85 backdrop-blur-sm border border-white/[0.08] rounded-xl px-3.5 py-2 shadow-sm">
                    {/* Mode Toggle: Wipe vs Pan */}
                    <div className="flex items-center gap-1 bg-[#060910] p-1 rounded-lg border border-white/[0.06]">
                      <button
                        onClick={() => setInteractionMode("curtain")}
                        className={`flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-semibold transition-all cursor-pointer ${
                          interactionMode === "curtain"
                            ? "bg-teal-500/15 text-teal-300 border border-teal-500/30 shadow-sm"
                            : "text-slate-400 hover:text-slate-200"
                        }`}
                        title="Interactive Curtain Split Mode"
                      >
                        <Sliders className="w-3.5 h-3.5 rotate-90" />
                        <span>Curtain Wipe</span>
                      </button>

                      <button
                        onClick={() => {
                          setInteractionMode("pan");
                          if (zoom === 1) handleZoomChange(2);
                        }}
                        className={`flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-semibold transition-all cursor-pointer ${
                          interactionMode === "pan"
                            ? "bg-teal-500/15 text-teal-300 border border-teal-500/30 shadow-sm"
                            : "text-slate-400 hover:text-slate-200"
                        }`}
                        title="Pan and Drag Image Viewport"
                      >
                        <Move className="w-3.5 h-3.5" />
                        <span>Pan & Drag</span>
                      </button>
                    </div>

                    {/* Zoom Presets & Controls */}
                    <div className="flex items-center gap-2">
                      <div className="flex items-center gap-1 bg-[#060910] p-1 rounded-lg border border-white/[0.06]">
                        <button
                          onClick={() => handleZoomChange(zoom - 0.5)}
                          disabled={zoom <= 1}
                          className="p-1 text-slate-400 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed rounded transition-colors cursor-pointer"
                          title="Zoom Out"
                        >
                          <ZoomOut className="w-3.5 h-3.5" />
                        </button>

                        {[1, 1.5, 2, 3, 4].map((z) => (
                          <button
                            key={z}
                            onClick={() => handleZoomChange(z)}
                            className={`px-2 py-0.5 rounded text-xs font-mono font-semibold transition-all cursor-pointer ${
                              zoom === z
                                ? "bg-[#182130] text-teal-300 border border-teal-500/30 shadow-sm"
                                : "text-slate-400 hover:text-white"
                            }`}
                          >
                            {z}x
                          </button>
                        ))}

                        <button
                          onClick={() => handleZoomChange(zoom + 0.5)}
                          disabled={zoom >= 4}
                          className="p-1 text-slate-400 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed rounded transition-colors cursor-pointer"
                          title="Zoom In"
                        >
                          <ZoomIn className="w-3.5 h-3.5" />
                        </button>
                      </div>

                      <button
                        onClick={resetView}
                        className="p-1.5 text-slate-400 hover:text-teal-300 bg-[#060910] border border-white/[0.06] hover:border-teal-500/30 rounded-lg transition-all cursor-pointer"
                        title="Reset Zoom & Pan"
                      >
                        <RotateCcw className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>

                  {/* Main Curtain Viewport */}
                  <div
                    ref={curtainRef}
                    onMouseDown={handleMouseDown}
                    onMouseMove={handleMouseMove}
                    onMouseUp={handleMouseUp}
                    onMouseLeave={handleMouseUp}
                    onTouchStart={handleTouchStart}
                    onTouchMove={handleTouchMove}
                    onTouchEnd={handleMouseUp}
                    className="relative w-full max-w-3xl aspect-video md:aspect-[16/10] rounded-xl overflow-hidden select-none border border-white/[0.08] shadow-lg bg-black"
                    style={{
                      cursor:
                        interactionMode === "pan"
                          ? isDragging
                            ? "grabbing"
                            : "grab"
                          : "ew-resize",
                    }}
                  >
                    {/* Super-Resolved Base Image */}
                    <img
                      src={result.images.super_res}
                      alt="Super-Resolved"
                      style={{
                        transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                        transformOrigin: "center center",
                        transition: isDragging ? "none" : "transform 0.15s ease-out",
                      }}
                      className="absolute inset-0 w-full h-full object-contain pointer-events-none"
                    />

                    {/* Low-Resolution Clipped Overlay */}
                    <div
                      style={{ width: `${curtainPos}%` }}
                      className="absolute inset-y-0 left-0 overflow-hidden border-r-2 border-teal-400 shadow-[1px_0_12px_rgba(45,212,191,0.4)] pointer-events-none"
                    >
                      <img
                        src={result.images.low_res}
                        alt="Low-Resolution"
                        style={{
                          width: containerSize.width ? `${containerSize.width}px` : "100%",
                          height: containerSize.height ? `${containerSize.height}px` : "100%",
                          transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                          transformOrigin: "center center",
                          transition: isDragging ? "none" : "transform 0.15s ease-out",
                        }}
                        className="absolute inset-0 w-full h-full object-contain max-w-none"
                      />
                    </div>

                    {/* Divider Handle Badge */}
                    <div
                      style={{ left: `${curtainPos}%` }}
                      className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-8 h-8 rounded-full bg-teal-400 text-slate-950 font-bold flex items-center justify-center shadow-md pointer-events-none transition-transform duration-150 ease-out"
                    >
                      <Sliders className="w-3.5 h-3.5 rotate-90" />
                    </div>

                    {/* Floating Corner Badges */}
                    <div className="absolute top-3 left-3 px-2.5 py-1 rounded-full bg-slate-950/80 backdrop-blur-sm border border-white/10 text-[11px] font-medium text-slate-200 pointer-events-none shadow-sm flex items-center gap-1.5">
                      <span className="w-1.5 h-1.5 rounded-full bg-slate-400" />
                      <span>Input (10m)</span>
                    </div>
                    <div className="absolute top-3 right-3 px-2.5 py-1 rounded-full bg-teal-950/80 backdrop-blur-sm border border-teal-500/30 text-[11px] font-medium text-teal-300 pointer-events-none shadow-sm flex items-center gap-1.5">
                      <span className="w-1.5 h-1.5 rounded-full bg-teal-400" />
                      <span>Super-Res (2.5m)</span>
                    </div>

                    {zoom > 1 && (
                      <div className="absolute bottom-3 right-3 px-2.5 py-0.5 rounded-full bg-slate-950/80 backdrop-blur-sm border border-teal-500/30 text-[11px] font-mono font-medium text-teal-300 pointer-events-none shadow-sm">
                        {zoom}x Magnified {interactionMode === "pan" ? "• Pan Mode" : ""}
                      </div>
                    )}
                  </div>

                  {/* Interactive Range Slider below Image */}
                  <div className="w-full max-w-3xl flex items-center gap-4 bg-[#090e17]/85 backdrop-blur-sm border border-white/[0.08] rounded-xl px-4 py-2.5 shadow-sm">
                    <span className="text-xs font-medium text-slate-400 flex items-center gap-1.5 whitespace-nowrap">
                      Input ({Math.round(curtainPos)}%)
                    </span>
                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={curtainPos}
                      onChange={(e) => setCurtainPos(Number(e.target.value))}
                      className="w-full accent-teal-400 h-2 bg-[#1d273a] rounded-lg cursor-pointer"
                    />
                    <span className="text-xs font-medium text-teal-400 flex items-center gap-1.5 whitespace-nowrap">
                      Super-Res ({100 - Math.round(curtainPos)}%)
                    </span>
                  </div>
                </div>
              )}

              {/* TAB 2: SIDE-BY-SIDE */}
              {activeViewTab === "split" && (
                <div className="space-y-4 w-full max-w-4xl">
                  {/* Zoom controls for dual view */}
                  <div className="flex items-center justify-end gap-2">
                    <span className="text-xs text-slate-400 font-medium">Magnify Both:</span>
                    <div className="flex items-center gap-1 bg-[#090e17]/85 backdrop-blur-sm border border-white/[0.08] p-1 rounded-lg">
                      {[1, 1.5, 2, 3].map((z) => (
                        <button
                          key={z}
                          onClick={() => handleZoomChange(z)}
                          className={`px-2 py-0.5 rounded text-xs font-mono font-medium transition-all cursor-pointer ${
                            zoom === z
                              ? "bg-[#182130] text-teal-300 border border-teal-500/30 shadow-sm"
                              : "text-slate-400 hover:text-white"
                          }`}
                        >
                          {z}x
                        </button>
                      ))}
                      <button
                        onClick={resetView}
                        className="p-1 text-slate-400 hover:text-teal-300 rounded transition-colors cursor-pointer"
                        title="Reset Zoom"
                      >
                        <RotateCcw className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full items-center">
                    <div className="space-y-2.5 text-center">
                      <span className="text-xs font-semibold px-3 py-1 rounded-full bg-sky-500/10 text-sky-300 border border-sky-500/20 inline-block font-mono">
                        Original Low-Resolution ({result.metadata.input_shape.join("x")})
                      </span>
                      <div className="overflow-hidden rounded-xl border border-white/[0.08] shadow-md bg-black">
                        <img
                          src={result.images.low_res}
                          alt="Low-res"
                          style={{
                            transform: `scale(${zoom})`,
                            transition: "transform 0.15s ease-out",
                          }}
                          className="mx-auto max-h-[360px] object-contain"
                        />
                      </div>
                    </div>
                    <div className="space-y-2.5 text-center">
                      <span className="text-xs font-semibold px-3 py-1 rounded-full bg-teal-500/10 text-teal-300 border border-teal-500/20 inline-block font-mono">
                        Super-Resolved ({result.metadata.output_shape.join("x")})
                      </span>
                      <div className="overflow-hidden rounded-xl border border-white/[0.08] shadow-md bg-black">
                        <img
                          src={result.images.super_res}
                          alt="Super-res"
                          style={{
                            transform: `scale(${zoom})`,
                            transition: "transform 0.15s ease-out",
                          }}
                          className="mx-auto max-h-[360px] object-contain"
                        />
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* TAB 3: UNCERTAINTY MAP (MONTE CARLO EPISTEMIC HEATMAP) */}
              {activeViewTab === "uncertainty" && (
                <div className="flex flex-col items-center space-y-4 w-full">
                  <div className="relative rounded-xl overflow-hidden border border-white/[0.08] shadow-lg bg-black/80 max-w-2xl">
                    <img
                      src={result.images.uncertainty_heatmap}
                      alt="Epistemic Uncertainty Heatmap"
                      className="max-h-[460px] object-contain mx-auto"
                    />
                  </div>
                  <div className="p-4 rounded-xl bg-black/50 border border-white/[0.06] text-xs text-slate-300 max-w-2xl leading-relaxed space-y-2">
                    <div className="flex items-center justify-between border-b border-white/[0.06] pb-2 font-mono text-[11px]">
                      <span className="text-teal-400 font-bold flex items-center gap-1.5">
                        <Activity className="w-3.5 h-3.5" />
                        {numPasses}-Pass Monte Carlo Dropout Uncertainty
                      </span>
                      <span className="text-zinc-400">
                        Colormap: <strong className="text-white uppercase">{colormap}</strong>
                      </span>
                    </div>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 font-mono text-center pt-1">
                      <div className="p-2 rounded-lg bg-[#0e1524] border border-white/[0.04]">
                        <span className="text-[10px] text-zinc-500 block">Mean &sigma;</span>
                        <span className="text-teal-300 font-bold">{result.metrics.mean_uncertainty}</span>
                      </div>
                      <div className="p-2 rounded-lg bg-[#0e1524] border border-white/[0.04]">
                        <span className="text-[10px] text-zinc-500 block">Peak &sigma;</span>
                        <span className="text-amber-300 font-bold">{result.metrics.max_uncertainty}</span>
                      </div>
                      <div className="p-2 rounded-lg bg-[#0e1524] border border-white/[0.04]">
                        <span className="text-[10px] text-zinc-500 block">Std Dev &sigma;</span>
                        <span className="text-sky-300 font-bold">{result.metrics.std_uncertainty}</span>
                      </div>
                      <div className="p-2 rounded-lg bg-[#0e1524] border border-white/[0.04]">
                        <span className="text-[10px] text-zinc-500 block">P95 &sigma;</span>
                        <span className="text-purple-300 font-bold">{result.metrics.p95_uncertainty}</span>
                      </div>
                    </div>
                    <p className="text-slate-400 pt-1 text-[11px]">
                      <strong className="text-slate-200">Interpretation:</strong> Warmer colors highlight pixels with higher predictive standard deviation across the {numPasses} stochastic Monte Carlo dropout forward passes. High uncertainty naturally concentrates at complex structural boundaries, dense road intersections, and shadowed building edges.
                    </p>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
