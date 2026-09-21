"use client";

import React, { useState, useRef } from "react";
import {
  Upload,
  Play,
  Download,
  AlertCircle,
  Layers,
  Eye,
  Sliders,
  Sparkles,
  ChevronRight,
  Maximize2,
  FileCheck
} from "lucide-react";

interface SatelliteStudioProps {
  onMetricsUpdate?: (metrics: any, sceneName?: string) => void;
  selectedSceneKey?: string | null;
}

export function SatelliteStudio({ onMetricsUpdate, selectedSceneKey }: SatelliteStudioProps) {
  const [selectedSample, setSelectedSample] = useState<string | null>(selectedSceneKey || null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [numPasses, setNumPasses] = useState<number>(15);
  const [colormap, setColormap] = useState<string>("turbo");
  const [activeViewTab, setActiveViewTab] = useState<string>("curtain");
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<any | null>(null);
  const [curtainPos, setCurtainPos] = useState<number>(50); // 0 - 100 percentage
  const curtainRef = useRef<HTMLDivElement>(null);

  // Sync external scene selection if provided
  React.useEffect(() => {
    if (selectedSceneKey) {
      setSelectedSample(selectedSceneKey);
      setUploadedFile(null);
    }
  }, [selectedSceneKey]);

  const handleRunInference = async () => {
    if (!uploadedFile && !selectedSample) {
      setError("Please select one of the satellite scenes below or upload an image first.");
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

      // Call FastAPI backend with configurable URL
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
      if (onMetricsUpdate && data.metrics) {
        onMetricsUpdate(data.metrics, selectedSample || uploadedFile?.name);
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

  const getSceneNameDisplay = () => {
    if (uploadedFile) return uploadedFile.name;
    if (selectedSample === "sentinel2_nir") return "Sentinel-2 RGB+NIR (4-Band)";
    if (selectedSample === "sentinel2_rgb") return "Sentinel-2 True Color (3-Band)";
    if (selectedSample === "urban_optical") return "Urban Optical Scene (PNG)";
    return "--";
  };

  const colormapList = [
    { id: "turbo", name: "Turbo", color: "from-blue-500 via-emerald-400 to-red-500" },
    { id: "magma", name: "Magma", color: "from-purple-900 via-red-500 to-amber-300" },
    { id: "inferno", name: "Inferno", color: "from-black via-orange-600 to-yellow-300" },
    { id: "viridis", name: "Viridis", color: "from-indigo-900 via-teal-500 to-yellow-400" },
  ];

  return (
    <div className="relative overflow-hidden bg-[#10141d] border border-[#232c3f] rounded-3xl p-6 md:p-8 space-y-7 shadow-[0_20px_50px_rgba(0,0,0,0.6)]">
      {/* Material 3 Expressive Background Ambient Glows */}
      <div className="absolute -top-24 -right-24 w-96 h-96 bg-[#4fe0cd]/10 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute -bottom-24 -left-24 w-80 h-80 bg-[#3b82f6]/10 rounded-full blur-3xl pointer-events-none" />

      {/* Header Bar */}
      <div className="relative flex flex-col md:flex-row md:items-center justify-between gap-5 pb-5 border-b border-[#21293a]">
        <div className="space-y-1">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#18202d] border border-[#2a3449] text-xs font-semibold text-[#4fe0cd]">
            <Sparkles className="w-3.5 h-3.5 animate-pulse" />
            <span>Material 3 Studio</span>
          </div>
          <h2 className="text-xl md:text-2xl font-bold tracking-tight text-white flex items-center gap-2.5">
            <span>2x Satellite Super-Resolution & Uncertainty Studio</span>
          </h2>
          <p className="text-xs text-[#94a3b8] max-w-xl">
            Sub-pixel pixel-shuffle convolution with 15-pass Monte Carlo Dropout Bayesian approximation.
          </p>
        </div>

        {/* Expressive Hero Action Pill Button */}
        <button
          onClick={handleRunInference}
          disabled={loading}
          className="relative group bg-gradient-to-r from-[#4fe0cd] via-[#2dd4bf] to-[#14b8a6] hover:opacity-95 text-[#003831] font-bold text-sm px-7 py-3.5 rounded-full flex items-center justify-center gap-2.5 transition-all shadow-[0_4px_25px_rgba(79,224,205,0.35)] hover:shadow-[0_6px_35px_rgba(79,224,205,0.5)] active:scale-95 disabled:opacity-40 shrink-0"
        >
          {loading ? (
            <>
              <div className="w-4 h-4 border-2 border-[#003831] border-t-transparent rounded-full animate-spin" />
              <span>Running {numPasses} MC Passes...</span>
            </>
          ) : (
            <>
              <Play className="w-4 h-4 fill-current" />
              <span>Execute 2x Super-Resolution</span>
            </>
          )}
        </button>
      </div>

      {error && (
        <div className="p-4 bg-red-950/40 border border-red-800/60 rounded-2xl text-xs text-red-200 flex items-center gap-3 animate-in fade-in">
          <AlertCircle className="w-4 h-4 shrink-0 text-red-400" />
          <span>{error}</span>
        </div>
      )}

      {/* 3-Step Material Tonal Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        {/* Step 1: Input Scene */}
        <div className="bg-[#161c28] border border-[#252f44] rounded-2xl p-5 space-y-3.5 flex flex-col justify-between hover:border-[#333e56] transition-all">
          <div className="space-y-2.5">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-[#4fe0cd] uppercase tracking-wider flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-[#4fe0cd]" />
                1. Select Satellite Scene
              </span>
            </div>

            <div className="space-y-2">
              <button
                onClick={() => {
                  setSelectedSample("sentinel2_nir");
                  setUploadedFile(null);
                  setError(null);
                }}
                className={`w-full text-left px-3.5 py-2.5 rounded-xl text-xs font-semibold border transition-all flex items-center justify-between ${
                  selectedSample === "sentinel2_nir" && !uploadedFile
                    ? "border-[#4fe0cd] bg-[#1e2738] text-white shadow-[0_0_15px_rgba(79,224,205,0.15)]"
                    : "border-[#273247] bg-[#121620] text-[#94a3b8] hover:bg-[#1a2130] hover:text-white"
                }`}
              >
                <span>Sentinel-2 RGB+NIR (4-Band)</span>
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-[#242e42] text-[#94a3b8]">16-bit GeoTIFF</span>
              </button>

              <button
                onClick={() => {
                  setSelectedSample("sentinel2_rgb");
                  setUploadedFile(null);
                  setError(null);
                }}
                className={`w-full text-left px-3.5 py-2.5 rounded-xl text-xs font-semibold border transition-all flex items-center justify-between ${
                  selectedSample === "sentinel2_rgb" && !uploadedFile
                    ? "border-[#4fe0cd] bg-[#1e2738] text-white shadow-[0_0_15px_rgba(79,224,205,0.15)]"
                    : "border-[#273247] bg-[#121620] text-[#94a3b8] hover:bg-[#1a2130] hover:text-white"
                }`}
              >
                <span>Sentinel-2 True Color (3-Band)</span>
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-[#242e42] text-[#94a3b8]">16-bit GeoTIFF</span>
              </button>

              <button
                onClick={() => {
                  setSelectedSample("urban_optical");
                  setUploadedFile(null);
                  setError(null);
                }}
                className={`w-full text-left px-3.5 py-2.5 rounded-xl text-xs font-semibold border transition-all flex items-center justify-between ${
                  selectedSample === "urban_optical" && !uploadedFile
                    ? "border-[#4fe0cd] bg-[#1e2738] text-white shadow-[0_0_15px_rgba(79,224,205,0.15)]"
                    : "border-[#273247] bg-[#121620] text-[#94a3b8] hover:bg-[#1a2130] hover:text-white"
                }`}
              >
                <span>Urban Optical Scene</span>
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-[#242e42] text-[#94a3b8]">PNG</span>
              </button>
            </div>
          </div>

          <label className="border border-dashed border-[#34425d] hover:border-[#4fe0cd] bg-[#121620]/60 hover:bg-[#192130] rounded-xl p-3 flex items-center justify-center gap-2.5 cursor-pointer text-xs font-medium text-[#94a3b8] hover:text-white transition-all">
            <Upload className="w-4 h-4 text-[#4fe0cd]" />
            <span className="truncate max-w-[200px]">
              {uploadedFile ? uploadedFile.name : "Upload raw .tif / .png / .tiff"}
            </span>
            <input
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
          </label>
        </div>

        {/* Step 2: Inference Config */}
        <div className="bg-[#161c28] border border-[#252f44] rounded-2xl p-5 space-y-4 flex flex-col justify-between hover:border-[#333e56] transition-all">
          <div className="space-y-3.5">
            <span className="text-xs font-bold text-[#60a5fa] uppercase tracking-wider flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-[#60a5fa]" />
              2. Monte Carlo Bayesian Config
            </span>

            <div>
              <div className="flex justify-between text-xs font-medium mb-2">
                <span className="text-[#94a3b8]">Stochastic Passes (T):</span>
                <span className="px-2.5 py-0.5 rounded-full bg-[#1f283a] text-[#60a5fa] font-mono font-bold">
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
                className="w-full h-2 bg-[#21293a] rounded-lg appearance-none cursor-pointer accent-[#4fe0cd]"
              />
              <div className="flex justify-between text-[10px] text-[#64748b] mt-1">
                <span>Fast (5)</span>
                <span>Calibrated (15)</span>
                <span>Deep (30)</span>
              </div>
            </div>

            <div>
              <span className="text-xs text-[#94a3b8] block mb-2 font-medium">Uncertainty Colormap:</span>
              <div className="grid grid-cols-2 gap-2">
                {colormapList.map((cm) => (
                  <button
                    key={cm.id}
                    onClick={() => setColormap(cm.id)}
                    className={`px-3 py-1.5 rounded-full text-xs font-semibold flex items-center gap-2 border transition-all ${
                      colormap === cm.id
                        ? "border-[#4fe0cd] bg-[#1f283a] text-white shadow-[0_0_10px_rgba(79,224,205,0.2)]"
                        : "border-[#263043] bg-[#121620] text-[#94a3b8] hover:bg-[#1a2130]"
                    }`}
                  >
                    <span className={`w-3 h-3 rounded-full bg-gradient-to-r ${cm.color}`} />
                    <span>{cm.name}</span>
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Step 3: Scene Telemetry */}
        <div className="bg-[#161c28] border border-[#252f44] rounded-2xl p-5 flex flex-col justify-between hover:border-[#333e56] transition-all">
          <div className="space-y-3">
            <span className="text-xs font-bold text-[#f59e0b] uppercase tracking-wider flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-[#f59e0b]" />
              3. Telemetry Specs
            </span>

            <div className="space-y-2 text-xs">
              <div className="flex justify-between items-center py-1.5 border-b border-[#21293a]">
                <span className="text-[#64748b]">Selected Scene:</span>
                <span className="text-white font-mono font-medium truncate max-w-[160px] text-right">
                  {getSceneNameDisplay()}
                </span>
              </div>
              <div className="flex justify-between items-center py-1.5 border-b border-[#21293a]">
                <span className="text-[#64748b]">Magnification:</span>
                <span className="px-2 py-0.5 rounded-full bg-[#182622] text-[#4fe0cd] font-semibold">
                  2x Sub-Pixel
                </span>
              </div>
              <div className="flex justify-between items-center py-1.5 border-b border-[#21293a]">
                <span className="text-[#64748b]">Target Dimensions:</span>
                <span className="text-white font-mono font-medium">
                  {result ? `${result.metadata.output_shape.join(" x ")} px` : "-- x -- px"}
                </span>
              </div>
              <div className="flex justify-between items-center py-1.5">
                <span className="text-[#64748b]">Engine State:</span>
                <span className="text-white flex items-center gap-1.5 font-medium">
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                  Pretrained Weights Ready
                </span>
              </div>
            </div>
          </div>

          {result && (
            <a
              href={`${process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000"}${result.download_url}`}
              target="_blank"
              download
              className="w-full mt-3 bg-gradient-to-r from-[#1e2a3b] to-[#151f2d] hover:from-[#243348] hover:to-[#1b2738] border border-[#344561] text-[#93c5fd] font-semibold text-xs py-2.5 px-4 rounded-full flex items-center justify-center gap-2 transition-all shadow-md active:scale-95"
            >
              <Download className="w-4 h-4 text-[#60a5fa]" />
              <span>Download 2x GeoTIFF</span>
            </a>
          )}
        </div>
      </div>

      {/* Material 3 Segmented Pill Tabs & Action Header */}
      <div className="space-y-4 pt-2">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 border-b border-[#21293a] pb-3">
          {/* Expressive Segmented Pill Bar */}
          <div className="inline-flex p-1 rounded-full bg-[#151a25] border border-[#273247] overflow-x-auto max-w-full">
            <button
              onClick={() => setActiveViewTab("curtain")}
              className={`px-4 py-1.5 rounded-full text-xs font-semibold transition-all whitespace-nowrap ${
                activeViewTab === "curtain"
                  ? "bg-[#4fe0cd] text-[#003831] shadow-[0_2px_12px_rgba(79,224,205,0.35)]"
                  : "text-[#94a3b8] hover:text-white"
              }`}
            >
              Interactive Curtain (1x vs 2x)
            </button>
            <button
              onClick={() => setActiveViewTab("sbs")}
              className={`px-4 py-1.5 rounded-full text-xs font-semibold transition-all whitespace-nowrap ${
                activeViewTab === "sbs"
                  ? "bg-[#4fe0cd] text-[#003831] shadow-[0_2px_12px_rgba(79,224,205,0.35)]"
                  : "text-[#94a3b8] hover:text-white"
              }`}
            >
              Side-by-Side
            </button>
            <button
              onClick={() => setActiveViewTab("split")}
              className={`px-4 py-1.5 rounded-full text-xs font-semibold transition-all whitespace-nowrap ${
                activeViewTab === "split"
                  ? "bg-[#4fe0cd] text-[#003831] shadow-[0_2px_12px_rgba(79,224,205,0.35)]"
                  : "text-[#94a3b8] hover:text-white"
              }`}
            >
              Dual 1x / 2x
            </button>
            <button
              onClick={() => setActiveViewTab("uncertainty")}
              className={`px-4 py-1.5 rounded-full text-xs font-semibold transition-all whitespace-nowrap ${
                activeViewTab === "uncertainty"
                  ? "bg-[#4fe0cd] text-[#003831] shadow-[0_2px_12px_rgba(79,224,205,0.35)]"
                  : "text-[#94a3b8] hover:text-white"
              }`}
            >
              Epistemic Uncertainty
            </button>
            <button
              onClick={() => setActiveViewTab("cir")}
              className={`px-4 py-1.5 rounded-full text-xs font-semibold transition-all whitespace-nowrap ${
                activeViewTab === "cir"
                  ? "bg-[#4fe0cd] text-[#003831] shadow-[0_2px_12px_rgba(79,224,205,0.35)]"
                  : "text-[#94a3b8] hover:text-white"
              }`}
            >
              False-Color NIR
            </button>
            <button
              onClick={() => setActiveViewTab("meta")}
              className={`px-4 py-1.5 rounded-full text-xs font-semibold transition-all whitespace-nowrap ${
                activeViewTab === "meta"
                  ? "bg-[#4fe0cd] text-[#003831] shadow-[0_2px_12px_rgba(79,224,205,0.35)]"
                  : "text-[#94a3b8] hover:text-white"
              }`}
            >
              Scene Telemetry
            </button>
          </div>

          {result && (
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <span className="px-3 py-1 rounded-full bg-[#182622] text-[#4fe0cd] font-semibold border border-[#274c42]">
                PSNR: {result.metrics.psnr_db} dB
              </span>
              <span className="px-3 py-1 rounded-full bg-[#1b2538] text-[#93c5fd] font-semibold border border-[#2d4265]">
                SSIM: {result.metrics.ssim}
              </span>
              <span className="px-3 py-1 rounded-full bg-[#272118] text-[#fbcfe8] font-semibold border border-[#523d24]">
                Gain: {result.metrics.sharpness_gain_ratio}x
              </span>
            </div>
          )}
        </div>

        {/* Viewport Card */}
        <div className="bg-[#121622] border border-[#242e42] rounded-2xl p-4 md:p-6 min-h-[380px] flex items-center justify-center relative overflow-hidden">
          {!result && !loading && (
            <div className="text-center space-y-3 text-[#64748b] py-12">
              <div className="w-14 h-14 rounded-full bg-[#1a202e] border border-[#2b364e] flex items-center justify-center mx-auto text-[#4fe0cd]">
                <Eye className="w-6 h-6 stroke-1.5" />
              </div>
              <p className="text-sm font-semibold text-white">No active inference output</p>
              <p className="text-xs max-w-sm mx-auto text-[#94a3b8]">
                Select a satellite scene above (or upload your raw .tif) and click{" "}
                <span className="text-[#4fe0cd] font-semibold">"Execute 2x Super-Resolution"</span>.
              </p>
            </div>
          )}

          {loading && (
            <div className="text-center space-y-4 py-16">
              <div className="w-12 h-12 border-3 border-[#4fe0cd] border-t-transparent rounded-full animate-spin mx-auto shadow-[0_0_20px_rgba(79,224,205,0.4)]" />
              <div className="text-sm font-bold text-white">
                Performing {numPasses} Monte Carlo stochastic passes...
              </div>
              <p className="text-xs text-[#94a3b8] max-w-sm mx-auto">
                Sampling residual CNN weights & aggregating epistemic variance map on RTX 5050 GPU
              </p>
            </div>
          )}

          {result && !loading && (
            <div className="w-full flex flex-col items-center">
              {/* 1. Interactive Curtain Split Slider */}
              {activeViewTab === "curtain" && (
                <div className="w-full flex flex-col items-center space-y-3">
                  <div
                    ref={curtainRef}
                    onMouseMove={(e) => {
                      if (e.buttons === 1) handleCurtainMove(e);
                    }}
                    onClick={handleCurtainMove}
                    onTouchMove={handleCurtainMove}
                    className="relative w-full max-w-2xl aspect-video md:aspect-[16/10] rounded-2xl overflow-hidden cursor-ew-resize select-none border border-[#2b364e] shadow-2xl bg-black"
                  >
                    {/* Background Layer: 2x Super-Resolved */}
                    <img
                      src={result.images.super_res}
                      alt="Super-Resolved 2x"
                      className="absolute inset-0 w-full h-full object-contain pointer-events-none"
                    />

                    {/* Foreground Clipped Layer: 1x Input */}
                    <div
                      style={{ width: `${curtainPos}%` }}
                      className="absolute inset-y-0 left-0 overflow-hidden border-r-2 border-[#4fe0cd] shadow-[2px_0_15px_rgba(79,224,205,0.6)]"
                    >
                      <img
                        src={result.images.low_res}
                        alt="Low-Resolution 1x"
                        className="absolute inset-0 w-full h-full object-contain max-w-none pointer-events-none"
                        style={{
                          width: curtainRef.current ? `${curtainRef.current.clientWidth}px` : "100%",
                        }}
                      />
                    </div>

                    {/* Draggable Center Pill Handle */}
                    <div
                      style={{ left: `${curtainPos}%` }}
                      className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-8 h-8 rounded-full bg-[#4fe0cd] text-[#003831] font-bold flex items-center justify-center shadow-[0_0_15px_rgba(79,224,205,0.7)] pointer-events-none transition-transform active:scale-110"
                    >
                      <Sliders className="w-4 h-4 rotate-90" />
                    </div>

                    {/* Pill Overlay Badges */}
                    <div className="absolute top-3 left-3 px-3 py-1 rounded-full bg-black/70 backdrop-blur border border-white/10 text-[11px] font-bold text-white pointer-events-none">
                      Input (1x Low-Res)
                    </div>
                    <div className="absolute top-3 right-3 px-3 py-1 rounded-full bg-[#003831]/80 backdrop-blur border border-[#4fe0cd]/40 text-[11px] font-bold text-[#4fe0cd] pointer-events-none">
                      BhuVistaar (2x Super-Res)
                    </div>
                  </div>

                  <div className="flex items-center gap-3 text-xs text-[#94a3b8]">
                    <span>Drag or click anywhere on the image to wipe between 1x and 2x resolution</span>
                    <span className="font-mono text-[#4fe0cd] font-bold">({Math.round(curtainPos)}% / {100 - Math.round(curtainPos)}%)</span>
                  </div>
                </div>
              )}

              {/* 2. Side by Side Banner */}
              {activeViewTab === "sbs" && (
                <div className="flex flex-col items-center space-y-2">
                  <img
                    src={result.images.side_by_side}
                    alt="Side-by-side comparison"
                    className="rounded-xl border border-[#2b364e] max-h-[440px] object-contain shadow-xl"
                  />
                  <span className="text-xs text-[#94a3b8]">
                    Left: Low-Resolution Input (1x) | Right: BhuVistaar 2x Sub-Pixel Super-Resolved
                  </span>
                </div>
              )}

              {/* 3. Dual Split Grid */}
              {activeViewTab === "split" && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full max-w-3xl items-center">
                  <div className="space-y-2 text-center">
                    <span className="text-xs font-semibold px-3 py-1 rounded-full bg-[#1a2130] text-[#94a3b8] border border-[#2d3a52] inline-block">
                      Original Low-Resolution ({result.metadata.input_shape.join("x")})
                    </span>
                    <img
                      src={result.images.low_res}
                      alt="Low-res"
                      className="rounded-xl border border-[#2b364e] mx-auto max-h-[340px] object-contain"
                    />
                  </div>
                  <div className="space-y-2 text-center">
                    <span className="text-xs font-semibold px-3 py-1 rounded-full bg-[#182622] text-[#4fe0cd] border border-[#274c42] inline-block">
                      Super-Resolved 2x ({result.metadata.output_shape.join("x")})
                    </span>
                    <img
                      src={result.images.super_res}
                      alt="Super-res"
                      className="rounded-xl border border-[#2b364e] mx-auto max-h-[340px] object-contain"
                    />
                  </div>
                </div>
              )}

              {/* 4. Epistemic Uncertainty */}
              {activeViewTab === "uncertainty" && (
                <div className="flex flex-col items-center space-y-3">
                  <img
                    src={result.images.uncertainty_heatmap}
                    alt="Epistemic uncertainty heatmap"
                    className="rounded-xl border border-[#2b364e] max-h-[420px] object-contain shadow-xl"
                  />
                  <div className="text-xs text-[#94a3b8] text-center max-w-xl">
                    <strong>Epistemic Uncertainty Interpretation</strong>: Warmer colors mark areas with higher predictive standard deviation where high-frequency edge ambiguity exists. Mean σ = {result.metrics.mean_uncertainty}, Peak σ = {result.metrics.max_uncertainty}.
                  </div>
                </div>
              )}

              {/* 5. False-Color NIR */}
              {activeViewTab === "cir" && (
                <div className="flex flex-col items-center space-y-3">
                  {result.images.false_color_cir ? (
                    <>
                      <img
                        src={result.images.false_color_cir}
                        alt="False-Color Infrared"
                        className="rounded-xl border border-[#2b364e] max-h-[420px] object-contain shadow-xl"
                      />
                      <p className="text-xs text-[#94a3b8] text-center max-w-lg">
                        <strong>Color Infrared (CIR: NIR + Red + Green)</strong>: Cellular plant structures strongly reflect Near-Infrared, rendering active agricultural vegetation in vivid red tones and water bodies in dark absorption tones.
                      </p>
                    </>
                  ) : (
                    <div className="py-12 text-center text-[#94a3b8] text-xs">
                      The selected image is 3-band RGB or single-band. To inspect False-Color NIR, select the 4-band Sentinel-2 RGB+NIR scene.
                    </div>
                  )}
                </div>
              )}

              {/* 6. Metadata */}
              {activeViewTab === "meta" && (
                <div className="p-6 bg-[#161c28] rounded-2xl border border-[#263043] max-w-xl w-full space-y-3 text-xs">
                  <div className="font-bold text-white border-b border-[#252f44] pb-2 text-sm flex items-center justify-between">
                    <span>Geospatial Scene & Sensor Telemetry</span>
                    <span className="text-[10px] px-2 py-0.5 rounded-full bg-[#1e2738] text-[#4fe0cd]">EPSG Georeferenced</span>
                  </div>
                  <div className="flex justify-between py-1.5 border-b border-[#1f2738]">
                    <span className="text-[#64748b]">File Name:</span>
                    <span className="font-mono text-white font-medium">{result.metadata.file_name}</span>
                  </div>
                  <div className="flex justify-between py-1.5 border-b border-[#1f2738]">
                    <span className="text-[#64748b]">CRS:</span>
                    <span className="font-mono text-white font-medium">{result.metadata.crs}</span>
                  </div>
                  <div className="flex justify-between py-1.5 border-b border-[#1f2738]">
                    <span className="text-[#64748b]">Input Dimensions:</span>
                    <span className="font-mono text-white font-medium">{result.metadata.input_shape.join(" x ")} px</span>
                  </div>
                  <div className="flex justify-between py-1.5 border-b border-[#1f2738]">
                    <span className="text-[#64748b]">Output Dimensions:</span>
                    <span className="font-mono text-[#4fe0cd] font-bold">{result.metadata.output_shape.join(" x ")} px (2x)</span>
                  </div>
                  <div className="flex justify-between py-1.5 border-b border-[#1f2738]">
                    <span className="text-[#64748b]">Processed Spectral Bands:</span>
                    <span className="font-mono text-white font-medium">{result.metadata.bands_count} bands</span>
                  </div>
                  <div className="flex justify-between py-1.5">
                    <span className="text-[#64748b]">Radiometric Depth:</span>
                    <span className="font-mono text-white font-medium">{result.metadata.bit_depth}</span>
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
