"use client";

import React, { useState, useEffect } from "react";
import { TrendingUp, TrendingDown, Clock, Activity, ShieldCheck, Zap, Sparkles } from "lucide-react";

export interface CardData {
  title: string;
  value: string;
  trend: string;
  trendType?: "up" | "down" | "neutral";
  trendText: string;
  subText: string;
}

interface SectionCardsProps {
  cards?: CardData[];
}

export function SectionCards({ cards }: SectionCardsProps) {
  const initialStandbyCards: CardData[] = [
    {
      title: "Reconstruction Fidelity (PSNR)",
      value: "--",
      trend: "Standby",
      trendType: "neutral",
      trendText: "Awaiting image inference",
      subText: "Peak Signal-to-Noise Ratio [dB]",
    },
    {
      title: "Structural Similarity (SSIM)",
      value: "--",
      trend: "Standby",
      trendType: "neutral",
      trendText: "Awaiting image inference",
      subText: "Multi-scale structural edge fidelity index",
    },
    {
      title: "Sharpness Gradient Gain",
      value: "--",
      trend: "Standby",
      trendType: "neutral",
      trendText: "Awaiting image inference",
      subText: "High-frequency Laplacian variance ratio",
    },
    {
      title: "Epistemic Uncertainty (σ)",
      value: "--",
      trend: "Standby",
      trendType: "neutral",
      trendText: "Awaiting image inference",
      subText: "15-pass Monte Carlo Bayesian standard deviation",
    },
  ];

  const displayCards = cards && cards.length > 0 ? cards : initialStandbyCards;

  const cardThemes = [
    {
      accent: "#10b981", // Muted emerald green
      topAccent: "bg-emerald-500/60",
      iconBg: "bg-emerald-500/10 border-emerald-500/20 text-emerald-400",
      borderHover: "hover:border-emerald-500/30",
      unitColor: "text-emerald-400",
      badgeActive: "border-emerald-500/25 bg-emerald-500/10 text-emerald-300",
      icon: <Zap className="w-4 h-4" />,
    },
    {
      accent: "#38bdf8", // Soft sky blue
      topAccent: "bg-sky-500/60",
      iconBg: "bg-sky-500/10 border-sky-500/20 text-sky-400",
      borderHover: "hover:border-sky-500/30",
      unitColor: "text-sky-400",
      badgeActive: "border-sky-500/25 bg-sky-500/10 text-sky-300",
      icon: <Sparkles className="w-4 h-4" />,
    },
    {
      accent: "#f59e0b", // Muted warm amber/orange
      topAccent: "bg-amber-500/60",
      iconBg: "bg-amber-500/10 border-amber-500/20 text-amber-400",
      borderHover: "hover:border-amber-500/30",
      unitColor: "text-amber-400",
      badgeActive: "border-amber-500/25 bg-amber-500/10 text-amber-300",
      icon: <Activity className="w-4 h-4" />,
    },
    {
      accent: "#a855f7", // Soft purple/violet
      topAccent: "bg-purple-500/60",
      iconBg: "bg-purple-500/10 border-purple-500/20 text-purple-400",
      borderHover: "hover:border-purple-500/30",
      unitColor: "text-purple-400",
      badgeActive: "border-purple-500/25 bg-purple-500/10 text-purple-300",
      icon: <ShieldCheck className="w-4 h-4" />,
    },
  ];

  // Formatter with smooth value transitions
  const formatValue = (val: string, unitColor: string) => {
    if (val === "--") {
      return (
        <span className="text-3xl md:text-4xl font-bold tracking-tight font-mono text-slate-600 select-none">
          --
        </span>
      );
    }

    // Check for dB unit
    if (val.endsWith("dB")) {
      const num = val.replace("dB", "").trim();
      return (
        <div key={val} className="flex items-baseline">
          <span className="text-3xl md:text-4xl font-bold tracking-tight font-mono text-white">
            {num}
          </span>
          <span className={`ml-2 text-lg md:text-xl font-semibold font-mono ${unitColor}`}>
            dB
          </span>
        </div>
      );
    }

    // Check for multiplier (e.g. 1.66x)
    if (val.endsWith("x")) {
      const num = val.slice(0, -1).trim();
      return (
        <div key={val} className="flex items-baseline">
          <span className="text-3xl md:text-4xl font-bold tracking-tight font-mono text-white">
            {num}
          </span>
          <span className={`ml-1.5 text-lg md:text-xl font-semibold font-mono ${unitColor}`}>
            x
          </span>
        </div>
      );
    }

    // Raw float number (e.g. 0.9860 or 0.00343)
    return (
      <span key={val} className="text-3xl md:text-4xl font-bold tracking-tight font-mono text-white inline-block">
        {val}
      </span>
    );
  };

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 md:gap-5">
      {displayCards.map((card, idx) => {
        const isStandby = card.value === "--";
        const theme = cardThemes[idx % cardThemes.length];

        return (
          <div
            key={idx}
            className={`relative overflow-hidden bg-[#0e1626]/70 backdrop-blur-md border border-white/[0.08] rounded-2xl p-5 md:p-6 flex flex-col justify-between transition-all duration-200 shadow-[0_8px_25px_rgba(0,0,0,0.3)] hover:shadow-[0_12px_32px_rgba(0,0,0,0.45)] hover:bg-[#121c30]/85 hover:border-white/[0.16] cursor-default ${theme.borderHover}`}
          >
            {/* Top Subtle Accent Strip */}
            <div className={`absolute top-0 left-6 right-6 h-[2px] rounded-full ${theme.topAccent}`} />

            {/* Card Header & Trend Pill Badge */}
            <div className="relative z-10 flex items-center justify-between gap-3 pt-1">
              <div className="flex items-center gap-2.5">
                <div
                  className={`w-7 h-7 rounded-lg border flex items-center justify-center ${theme.iconBg}`}
                >
                  {theme.icon}
                </div>
                <span className="text-xs font-semibold text-slate-300">
                  {card.title.split("(")[0].trim()}
                </span>
              </div>

              <div
                className={`flex items-center gap-1.5 text-[11px] font-semibold px-2.5 py-0.5 rounded-full border transition-colors ${
                  isStandby
                    ? "border-slate-800 bg-slate-900/60 text-slate-500"
                    : theme.badgeActive
                }`}
              >
                {!isStandby && card.trendType === "up" ? (
                  <TrendingUp className="w-3 h-3" />
                ) : !isStandby && card.trendType === "down" ? (
                  <TrendingDown className="w-3 h-3" />
                ) : (
                  <Clock className="w-3 h-3 opacity-60" />
                )}
                <span>{card.trend}</span>
              </div>
            </div>

            {/* Prominent Metric Display */}
            <div className="relative z-10 my-4 md:my-5">
              {formatValue(card.value, theme.unitColor)}
            </div>

            {/* Footer with Clear Visual Hierarchy */}
            <div className="relative z-10 pt-3 border-t border-white/[0.06] space-y-0.5">
              <div
                className={`text-xs font-medium ${
                  isStandby ? "text-slate-500" : "text-slate-200"
                }`}
              >
                <span>{card.trendText}</span>
              </div>
              <p className="text-[11px] text-slate-500 leading-snug">
                {card.subText}
              </p>
            </div>
          </div>
        );
      })}
    </div>
  );
}
