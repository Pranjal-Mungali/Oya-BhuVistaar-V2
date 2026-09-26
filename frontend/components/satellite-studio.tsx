"use client";

import React, { useState, useRef, useEffect } from "react";
import {
  Upload,
  Play,
  Download,
  AlertCircle,
  Eye,
  Sliders,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Move,
  FileCheck,
  X,
  Activity,
  Layers,
  Sparkles
} from "lucide-react";

interface SatelliteStudioProps {
  onMetricsUpdate?: (metrics: any, sceneName?: string) => void;
  onInferenceComplete?: (data: any, sceneName?: string) => void;
  selectedSceneKey?: string | null;
}

export function SatelliteStudio({ onMetricsUpdate, onInferenceComplete, selectedSceneKey }: SatelliteStudioProps) {
  const [selectedSample, setSelectedSample] = useState<string | null>(selectedSceneKey || null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [isDragOver, setIsDragOver] = useState<boolean>(false);
  const [numPasses, setNumPasses] = useState<number>(5);
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
      setError("Please upload a satellite image first, or click 'Try sample image'.");
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
      } catch {
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
      const sceneLabel = uploadedFile ? uploadedFile.name : (selectedSample || "Satellite Scene");
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
          "Failed to process image. Make sure the backend server is running."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleZoomChange = (newZoom: number) => {
    const clamped = Math.max(1, Math.min(4, newZoom));
    setZoom(clamped);
    if (clamped === 1) {
      setPan({ x: 0, y: 0 });
      setInteractionMode("curtain");
    }
  };

  const resetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
    setCurtainPos(50);
    setInteractionMode("curtain");
  };

  const handleCurtainMove = (e: React.MouseEvent<HTMLDivElement> | React.TouchEvent<HTMLDivElement>) => {
    if (!curtainRef.current) return;
    const rect = curtainRef.current.getBoundingClientRect();
    const clientX = "touches" in e ? e.touches[0].clientX : e.clientX;
    const offsetX = clientX - rect.left;
    const percentage = Math.max(0, Math.min(100, (offsetX / rect.width) * 100));
    setCurtainPos(percentage);
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

  const loadSampleScene = () => {
    setUploadedFile(null);
    setSelectedSample("satellite_sample");
    setError(null);
  };

  return (
    <div className="bg-[#121215] border border-zinc-800 rounded-2xl p-6 sm:p-8 space-y-7 shadow-[0_4px_30px_rgba(0,0,0,0.45)]">
      {/* 1. PRIMARY HERO SECTION: Upload & Execution */}
      <div className="space-y-6">
        {/* Upload Card - Clear, refined primary focus */}
        <div className="relative">
          {!uploadedFile && !selectedSample ? (
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-2xl p-10 sm:p-14 transition-all duration-200 ease-out cursor-pointer flex flex-col items-center justify-center text-center gap-4 group select-none ${
                isDragOver
                  ? "border-teal-400 bg-teal-500/[0.08]"
                  : "border-zinc-800 hover:border-teal-500/50 bg-[#0d0d0f] hover:bg-[#111114]"
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
              <div className="w-14 h-14 rounded-2xl bg-zinc-900 border border-zinc-700/80 flex items-center justify-center text-teal-400 group-hover:scale-105 group-hover:border-teal-500/50 group-hover:text-teal-300 group-hover:shadow-[0_0_20px_rgba(20,184,166,0.18)] transition-all duration-200 shadow-sm">
                <Upload className="w-6 h-6 stroke-[1.75]" />
              </div>
              <div className="space-y-1.5 max-w-md">
                <p className="text-sm sm:text-base font-medium text-zinc-100 group-hover:text-white transition-colors">
                  Drop your satellite image here, or <span className="text-teal-400 underline underline-offset-4 decoration-teal-500/40 group-hover:decoration-teal-400 font-semibold">browse</span>
                </p>
                <p className="text-xs text-zinc-400">
                  Supports GeoTIFF (.tif, .tiff) and optical imagery (.png, .jpg)
                </p>
              </div>
            </div>
          ) : (
            <div className="border border-zinc-800 bg-[#0d0d0f] rounded-2xl p-5 sm:p-6 flex flex-col sm:flex-row sm:items-center justify-between gap-4 transition-all duration-200 shadow-sm">
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
              <div className="flex items-center gap-4 min-w-0">
                <div className="w-12 h-12 rounded-xl bg-teal-500/10 border border-teal-500/20 flex items-center justify-center text-teal-400 shrink-0">
                  <FileCheck className="w-6 h-6 stroke-[1.75]" />
                </div>
                <div className="min-w-0">
                  <p className="text-sm font-semibold text-zinc-100 truncate max-w-[260px] sm:max-w-md">
                    {uploadedFile ? uploadedFile.name : "Sample Satellite Scene"}
                  </p>
                  <p className="text-xs text-zinc-400 font-mono mt-0.5">
                    {uploadedFile ? formatFileSize(uploadedFile.size) : "512 × 512 px"}
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-2 shrink-0 self-end sm:self-auto">
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="px-3.5 py-1.5 rounded-lg bg-[#18181c] hover:bg-[#202026] active:scale-[0.98] border border-zinc-700/80 text-xs font-medium text-zinc-300 hover:text-white transition-all duration-150 cursor-pointer"
                >
                  Change file
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setUploadedFile(null);
                    setSelectedSample(null);
                    setError(null);
                  }}
                  className="p-1.5 rounded-lg bg-[#18181c] hover:bg-red-500/10 active:scale-[0.95] border border-zinc-700/80 hover:border-red-500/20 text-zinc-400 hover:text-red-400 transition-all duration-150 cursor-pointer"
                  title="Remove image"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>
          )}
        </div>

        {/* 2. Controls & Action Bar */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-5 p-5 rounded-2xl bg-[#0d0d0f] border border-zinc-800 shadow-sm">
          <div className="flex flex-wrap items-center gap-5 sm:gap-7 text-xs text-zinc-300">
            {/* Stochastic passes slider */}
            <div className="flex items-center gap-2.5">
              <span className="text-zinc-400 font-medium">Monte Carlo passes:</span>
              <input
                type="range"
                min={5}
                max={30}
                step={5}
                value={numPasses}
                onChange={(e) => setNumPasses(parseInt(e.target.value))}
                className="w-28 h-1.5 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-teal-400"
              />
              <span className="font-mono text-teal-400 font-semibold w-6 text-right">
                {numPasses}
              </span>
            </div>

            {/* Uncertainty palette toggle */}
            <div className="flex items-center gap-2.5">
              <span className="text-zinc-400 font-medium">Palette:</span>
              <div className="inline-flex p-0.5 rounded-lg bg-[#18181c] border border-zinc-800">
                <button
                  type="button"
                  onClick={() => setColormap("turbo")}
                  className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all duration-150 cursor-pointer ${
                    colormap === "turbo"
                      ? "bg-zinc-800 text-teal-300 font-semibold shadow-sm"
                      : "text-zinc-400 hover:text-zinc-200"
                  }`}
                >
                  Turbo
                </button>
                <button
                  type="button"
                  onClick={() => setColormap("magma")}
                  className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all duration-150 cursor-pointer ${
                    colormap === "magma"
                      ? "bg-zinc-800 text-teal-300 font-semibold shadow-sm"
                      : "text-zinc-400 hover:text-zinc-200"
                  }`}
                >
                  Magma
                </button>
              </div>
            </div>
          </div>

          {/* Primary Action Button - Prominent */}
          <div className="flex items-center gap-3">
            <button
              onClick={handleRunInference}
              disabled={loading || (!uploadedFile && !selectedSample)}
              className="w-full md:w-auto bg-teal-500 hover:bg-teal-400 active:scale-[0.98] active:bg-teal-600 text-zinc-950 font-bold text-xs sm:text-sm px-7 py-3 rounded-xl flex items-center justify-center gap-2.5 transition-all duration-200 ease-out shadow-[0_2px_14px_rgba(20,184,166,0.28)] hover:shadow-[0_4px_22px_rgba(20,184,166,0.4)] active:shadow-none disabled:opacity-40 disabled:cursor-not-allowed disabled:shadow-none cursor-pointer tracking-wide"
            >
              {loading ? (
                <>
                  <div className="w-4 h-4 border-2 border-zinc-950 border-t-transparent rounded-full animate-spin" />
                  <span>Processing 4× resolution...</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-current" />
                  <span>Generate Super-Resolution</span>
                </>
              )}
            </button>

            {result && (
              <a
                href={`${process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000"}${result.download_url}`}
                target="_blank"
                download
                className="px-4 py-3 rounded-xl bg-[#18181c] hover:bg-[#202026] active:scale-[0.98] border border-zinc-700/80 text-zinc-200 hover:text-white text-xs font-medium flex items-center gap-2 transition-all duration-150 cursor-pointer shadow-sm"
                title="Download 4x Enhanced GeoTIFF"
              >
                <Download className="w-3.5 h-3.5 text-teal-400" />
                <span className="hidden sm:inline font-medium">Download</span>
              </a>
            )}
          </div>
        </div>
      </div>

      {error && (
        <div className="p-3.5 bg-red-950/20 border border-red-900/30 rounded-xl text-xs text-red-300 flex items-center gap-2.5 animate-tab-fade">
          <AlertCircle className="w-4 h-4 shrink-0 text-red-400" />
          <span>{error}</span>
        </div>
      )}

      {/* 3. VISUAL RESULTS SECTION */}
      <div className="space-y-4 pt-2 border-t border-zinc-800">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          {/* View Mode Segment Control */}
          <div className="inline-flex p-1 rounded-xl bg-[#0d0d0f] border border-zinc-800 text-xs">
            <button
              onClick={() => setActiveViewTab("curtain")}
              className={`px-3.5 py-1.5 rounded-lg font-medium transition-all duration-150 ease-out cursor-pointer ${
                activeViewTab === "curtain"
                  ? "bg-zinc-800 text-teal-300 shadow-sm"
                  : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/40"
              }`}
            >
              Curtain View
            </button>
            <button
              onClick={() => setActiveViewTab("split")}
              className={`px-3.5 py-1.5 rounded-lg font-medium transition-all duration-150 ease-out cursor-pointer ${
                activeViewTab === "split"
                  ? "bg-zinc-800 text-teal-300 shadow-sm"
                  : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/40"
              }`}
            >
              Side-by-Side
            </button>
            <button
              onClick={() => setActiveViewTab("uncertainty")}
              className={`px-3.5 py-1.5 rounded-lg font-medium transition-all duration-150 ease-out cursor-pointer ${
                activeViewTab === "uncertainty"
                  ? "bg-zinc-800 text-teal-300 shadow-sm"
                  : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/40"
              }`}
            >
              Uncertainty Map
            </button>
          </div>

          {/* Quick Metrics Summary Badge */}
          {result && (
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <span className="px-2.5 py-1 rounded-md bg-[#18181c] text-zinc-300 border border-zinc-800 font-mono">
                PSNR: <strong className="text-teal-400 font-semibold">{result.metrics.psnr_db} dB</strong>
              </span>
              <span className="px-2.5 py-1 rounded-md bg-[#18181c] text-zinc-300 border border-zinc-800 font-mono">
                SSIM: <strong className="text-teal-400 font-semibold">{result.metrics.ssim}</strong>
              </span>
              <span className="px-2.5 py-1 rounded-md bg-[#18181c] text-zinc-300 border border-zinc-800 font-mono">
                Gain: <strong className="text-teal-400 font-semibold">{result.metrics.sharpness_gain_ratio}×</strong>
              </span>
            </div>
          )}
        </div>

        {/* Viewport Canvas Container */}
        <div className="bg-[#09090b] border border-zinc-800 rounded-2xl p-4 sm:p-6 min-h-[420px] flex items-center justify-center relative overflow-hidden">
          {!result && !loading && (
            <div className="text-center space-y-3 text-zinc-500 py-16">
              <div className="w-11 h-11 rounded-xl bg-[#18181c] border border-zinc-800 flex items-center justify-center mx-auto text-zinc-400">
                <Eye className="w-5 h-5 stroke-[1.8]" />
              </div>
              <div className="space-y-1">
                <p className="text-sm font-medium text-zinc-300">No output generated yet</p>
                <p className="text-xs text-zinc-500 max-w-sm mx-auto">
                  Upload satellite imagery above and click "Generate Super-Resolution" to view the 4× enhanced reconstruction.
                </p>
              </div>
            </div>
          )}

          {loading && (
            <div className="text-center space-y-4 py-20">
              <div className="w-8 h-8 border-2 border-teal-400 border-t-transparent rounded-full animate-spin mx-auto" />
              <div className="space-y-1">
                <div className="text-sm font-medium text-zinc-200">
                  Executing 4× Real-ESRGAN super-resolution...
                </div>
                <p className="text-xs text-zinc-400 max-w-xs mx-auto">
                  Running {numPasses} Monte Carlo passes for epistemic uncertainty quantification
                </p>
              </div>
            </div>
          )}

          {result && !loading && (
            <div className="w-full flex flex-col items-center">
              {/* TAB 1: CURTAIN VIEW */}
              {activeViewTab === "curtain" && (
                <div key="curtain-tab" className="w-full flex flex-col items-center space-y-3.5 animate-tab-fade">
                  {/* Viewport Toolbar */}
                  <div className="flex flex-wrap items-center justify-between gap-3 w-full max-w-3xl bg-[#0d0d0f] border border-zinc-800 rounded-xl px-3.5 py-1.5 text-xs">
                    <div className="flex items-center gap-1 bg-[#18181c] p-0.5 rounded-lg border border-zinc-800">
                      <button
                        onClick={() => setInteractionMode("curtain")}
                        className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-medium transition-colors cursor-pointer ${
                          interactionMode === "curtain"
                            ? "bg-zinc-800 text-teal-300"
                            : "text-zinc-400 hover:text-zinc-200"
                        }`}
                        title="Curtain Wipe Mode"
                      >
                        <Sliders className="w-3.5 h-3.5 rotate-90" />
                        <span>Curtain Wipe</span>
                      </button>

                      <button
                        onClick={() => {
                          setInteractionMode("pan");
                          if (zoom === 1) handleZoomChange(2);
                        }}
                        className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-medium transition-colors cursor-pointer ${
                          interactionMode === "pan"
                            ? "bg-zinc-800 text-teal-300"
                            : "text-zinc-400 hover:text-zinc-200"
                        }`}
                        title="Pan and Drag Mode"
                      >
                        <Move className="w-3.5 h-3.5" />
                        <span>Pan & Drag</span>
                      </button>
                    </div>

                    {/* Zoom Presets */}
                    <div className="flex items-center gap-2">
                      <div className="flex items-center gap-1 bg-[#18181c] p-0.5 rounded-lg border border-zinc-800">
                        <button
                          onClick={() => handleZoomChange(zoom - 0.5)}
                          disabled={zoom <= 1}
                          className="p-1 text-zinc-400 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed rounded cursor-pointer"
                          title="Zoom Out"
                        >
                          <ZoomOut className="w-3.5 h-3.5" />
                        </button>

                        {[1, 1.5, 2, 3, 4].map((z) => (
                          <button
                            key={z}
                            onClick={() => handleZoomChange(z)}
                            className={`px-2 py-0.5 rounded text-xs font-mono font-medium transition-colors cursor-pointer ${
                              zoom === z
                                ? "bg-zinc-800 text-teal-300"
                                : "text-zinc-400 hover:text-zinc-200"
                            }`}
                          >
                            {z}x
                          </button>
                        ))}

                        <button
                          onClick={() => handleZoomChange(zoom + 0.5)}
                          disabled={zoom >= 4}
                          className="p-1 text-zinc-400 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed rounded cursor-pointer"
                          title="Zoom In"
                        >
                          <ZoomIn className="w-3.5 h-3.5" />
                        </button>
                      </div>

                      <button
                        onClick={resetView}
                        className="p-1.5 text-zinc-400 hover:text-teal-300 bg-[#18181c] border border-zinc-800 rounded-lg transition-colors cursor-pointer"
                        title="Reset View"
                      >
                        <RotateCcw className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>

                  {/* Main Curtain Image Viewport */}
                  <div
                    ref={curtainRef}
                    onMouseDown={handleMouseDown}
                    onMouseMove={handleMouseMove}
                    onMouseUp={handleMouseUp}
                    onMouseLeave={handleMouseUp}
                    onTouchStart={handleTouchStart}
                    onTouchMove={handleTouchMove}
                    onTouchEnd={handleMouseUp}
                    className="relative w-full max-w-3xl aspect-video md:aspect-[16/10] rounded-xl overflow-hidden select-none border border-zinc-800 bg-black shadow-sm"
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
                      className="absolute inset-y-0 left-0 overflow-hidden border-r-2 border-teal-400 pointer-events-none"
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

                    {/* Divider Handle */}
                    <div
                      style={{ left: `${curtainPos}%` }}
                      className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-7 h-7 rounded-full bg-teal-400 text-zinc-950 font-bold flex items-center justify-center shadow-md pointer-events-none"
                    >
                      <Sliders className="w-3.5 h-3.5 rotate-90" />
                    </div>

                    {/* Corner Labels */}
                    <div className="absolute top-3 left-3 px-2.5 py-0.5 rounded-md bg-zinc-950/80 border border-zinc-800 text-[11px] font-medium text-zinc-300 pointer-events-none">
                      Input (10m)
                    </div>
                    <div className="absolute top-3 right-3 px-2.5 py-0.5 rounded-md bg-zinc-950/80 border border-teal-500/30 text-[11px] font-medium text-teal-300 pointer-events-none">
                      Super-Res (2.5m)
                    </div>

                    {zoom > 1 && (
                      <div className="absolute bottom-3 right-3 px-2.5 py-0.5 rounded-md bg-zinc-950/80 border border-zinc-800 text-[11px] font-mono text-teal-300 pointer-events-none">
                        {zoom}x {interactionMode === "pan" ? "• Pan Mode" : ""}
                      </div>
                    )}
                  </div>

                  {/* Range Slider for Fine Control */}
                  <div className="w-full max-w-3xl flex items-center gap-3 bg-[#0d0d0f] border border-zinc-800 rounded-xl px-4 py-2 text-xs">
                    <span className="text-zinc-400 whitespace-nowrap">Input ({Math.round(curtainPos)}%)</span>
                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={curtainPos}
                      onChange={(e) => setCurtainPos(Number(e.target.value))}
                      className="w-full accent-teal-400 h-1.5 bg-zinc-800 rounded-lg cursor-pointer"
                    />
                    <span className="text-teal-400 whitespace-nowrap">Super-Res ({100 - Math.round(curtainPos)}%)</span>
                  </div>
                </div>
              )}

              {/* TAB 2: SIDE-BY-SIDE */}
              {activeViewTab === "split" && (
                <div key="split-tab" className="space-y-4 w-full max-w-4xl animate-tab-fade">
                  <div className="flex items-center justify-end gap-2 text-xs">
                    <span className="text-zinc-400">Zoom:</span>
                    <div className="flex items-center gap-1 bg-[#0d0d0f] border border-zinc-800 p-0.5 rounded-lg">
                      {[1, 1.5, 2, 3].map((z) => (
                        <button
                          key={z}
                          onClick={() => handleZoomChange(z)}
                          className={`px-2 py-0.5 rounded text-xs font-mono font-medium transition-colors cursor-pointer ${
                            zoom === z ? "bg-zinc-800 text-teal-300" : "text-zinc-400 hover:text-zinc-200"
                          }`}
                        >
                          {z}x
                        </button>
                      ))}
                      <button
                        onClick={resetView}
                        className="p-1 text-zinc-400 hover:text-teal-300 rounded cursor-pointer"
                        title="Reset Zoom"
                      >
                        <RotateCcw className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full">
                    <div className="space-y-2 text-center">
                      <span className="text-xs font-medium text-zinc-400 block font-mono">
                        Original Input ({result.metadata.input_shape.join(" × ")})
                      </span>
                      <div className="overflow-hidden rounded-xl border border-zinc-800 bg-black">
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
                    <div className="space-y-2 text-center">
                      <span className="text-xs font-medium text-teal-400 block font-mono">
                        4× Super-Resolved ({result.metadata.output_shape.join(" × ")})
                      </span>
                      <div className="overflow-hidden rounded-xl border border-zinc-800 bg-black">
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

              {/* TAB 3: UNCERTAINTY MAP */}
              {activeViewTab === "uncertainty" && (
                <div key="uncertainty-tab" className="flex flex-col items-center space-y-4 w-full animate-tab-fade">
                  <div className="relative rounded-xl overflow-hidden border border-zinc-800 bg-black max-w-2xl">
                    <img
                      src={result.images.uncertainty_heatmap}
                      alt="Epistemic Uncertainty Heatmap"
                      className="max-h-[440px] object-contain mx-auto"
                    />
                  </div>
                  <div className="p-4 rounded-xl bg-[#0d0d0f] border border-zinc-800 text-xs text-zinc-300 max-w-2xl leading-relaxed space-y-3">
                    <div className="flex items-center justify-between border-b border-zinc-800 pb-2 text-[11px] font-mono">
                      <span className="text-teal-400 font-medium flex items-center gap-1.5">
                        <Activity className="w-3.5 h-3.5" />
                        {numPasses}-Pass Monte Carlo Dropout Uncertainty
                      </span>
                      <span className="text-zinc-400">
                        Palette: <strong className="text-zinc-200 uppercase">{colormap}</strong>
                      </span>
                    </div>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 font-mono text-center">
                      <div className="p-2 rounded-lg bg-[#18181c] border border-zinc-800">
                        <span className="text-[10px] text-zinc-500 block">Mean σ</span>
                        <span className="text-zinc-100 font-semibold">{result.metrics.mean_uncertainty}</span>
                      </div>
                      <div className="p-2 rounded-lg bg-[#18181c] border border-zinc-800">
                        <span className="text-[10px] text-zinc-500 block">Peak σ</span>
                        <span className="text-zinc-100 font-semibold">{result.metrics.max_uncertainty}</span>
                      </div>
                      <div className="p-2 rounded-lg bg-[#18181c] border border-zinc-800">
                        <span className="text-[10px] text-zinc-500 block">Std Dev σ</span>
                        <span className="text-zinc-100 font-semibold">{result.metrics.std_uncertainty}</span>
                      </div>
                      <div className="p-2 rounded-lg bg-[#18181c] border border-zinc-800">
                        <span className="text-[10px] text-zinc-500 block">P95 σ</span>
                        <span className="text-zinc-100 font-semibold">{result.metrics.p95_uncertainty}</span>
                      </div>
                    </div>
                    <p className="text-zinc-400 text-[11px]">
                      Warmer regions indicate higher model variance across dropout iterations, primarily concentrated at complex structural edges, parcel divisions, and building shadows.
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
