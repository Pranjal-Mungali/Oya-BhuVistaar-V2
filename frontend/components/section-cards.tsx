"use client";

import React from "react";
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
      subText: "High-frequency Laplacian variance gain",
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

  const cardIcons = [
    <Zap className="w-4 h-4 text-[#4fe0cd]" key="zap" />,
    <Sparkles className="w-4 h-4 text-[#60a5fa]" key="sparkles" />,
    <Activity className="w-4 h-4 text-[#f59e0b]" key="activity" />,
    <ShieldCheck className="w-4 h-4 text-[#ec4899]" key="shield" />,
  ];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 md:gap-5">
      {displayCards.map((card, idx) => {
        const isStandby = card.value === "--";

        return (
          <div
            key={idx}
            className="group relative overflow-hidden bg-[#121622] border border-[#232c3f] rounded-3xl p-5 md:p-6 flex flex-col justify-between hover:border-[#3a4763] transition-all duration-300 shadow-[0_10px_30px_rgba(0,0,0,0.4)]"
          >
            {/* Subtle Tonal Corner Glow */}
            <div className="absolute -top-12 -right-12 w-28 h-28 bg-[#4fe0cd]/5 group-hover:bg-[#4fe0cd]/10 rounded-full blur-2xl transition-all duration-500 pointer-events-none" />

            {/* Card Header & Trend Pill Badge */}
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <div className="w-7 h-7 rounded-full bg-[#1b2333] border border-[#29354e] flex items-center justify-center">
                  {cardIcons[idx % cardIcons.length]}
                </div>
                <span className="text-xs font-bold uppercase tracking-wider text-[#94a3b8]">
                  {card.title.split("(")[0]}
                </span>
              </div>

              <div
                className={`flex items-center gap-1.5 text-[11px] font-bold px-3 py-1 rounded-full border transition-all ${
                  card.trendType === "up"
                    ? "border-[#4fe0cd]/40 bg-[#162725] text-[#4fe0cd] shadow-[0_0_10px_rgba(79,224,205,0.2)]"
                    : card.trendType === "down"
                    ? "border-[#60a5fa]/40 bg-[#182337] text-[#93c5fd]"
                    : "border-[#252f44] bg-[#161b26] text-[#64748b]"
                }`}
              >
                {card.trendType === "up" ? (
                  <TrendingUp className="w-3 h-3 text-[#4fe0cd]" />
                ) : card.trendType === "down" ? (
                  <TrendingDown className="w-3 h-3 text-[#60a5fa]" />
                ) : (
                  <Clock className="w-3 h-3 text-[#64748b]" />
                )}
                <span>{card.trend}</span>
              </div>
            </div>

            {/* Big Value Metric */}
            <div className="my-4">
              <div
                className={`text-3xl md:text-4xl font-extrabold tracking-tight font-mono ${
                  isStandby ? "text-[#3f4a61]" : "text-white"
                }`}
              >
                {card.value}
              </div>
            </div>

            {/* Footer Details */}
            <div className="pt-3 border-t border-[#1d2536] space-y-1">
              <div
                className={`flex items-center gap-1.5 text-xs font-semibold ${
                  isStandby ? "text-[#64748b]" : "text-[#e2e8f0]"
                }`}
              >
                <span>{card.trendText}</span>
              </div>
              <p className="text-[11px] text-[#64748b] leading-tight">{card.subText}</p>
            </div>
          </div>
        );
      })}
    </div>
  );
}
