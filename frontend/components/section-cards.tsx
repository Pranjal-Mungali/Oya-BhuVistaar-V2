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
      subText: "High-frequency Laplacian variance ratio",
    },
    {
      title: "Epistemic Uncertainty (σ)",
      value: "--",
      trend: "Standby",
      trendType: "neutral",
      trendText: "Awaiting image inference",
      subText: "Monte Carlo Bayesian standard deviation",
    },
  ];

  const displayCards = cards && cards.length > 0 ? cards : initialStandbyCards;

  const cardIcons = [
    <Zap key="zap" className="w-4 h-4" />,
    <Sparkles key="sparkles" className="w-4 h-4" />,
    <Activity key="activity" className="w-4 h-4" />,
    <ShieldCheck key="shield" className="w-4 h-4" />,
  ];

  const formatValue = (val: string) => {
    if (val === "--") {
      return (
        <span className="text-3xl md:text-4xl font-bold tracking-tight font-mono text-zinc-600 select-none">
          --
        </span>
      );
    }

    if (val.endsWith("dB")) {
      const num = val.replace("dB", "").trim();
      return (
        <div key={val} className="flex items-baseline">
          <span className="text-3xl md:text-4xl font-bold tracking-tight font-mono text-zinc-100">
            {num}
          </span>
          <span className="ml-2 text-base md:text-lg font-medium font-mono text-zinc-400">
            dB
          </span>
        </div>
      );
    }

    if (val.endsWith("x")) {
      const num = val.slice(0, -1).trim();
      return (
        <div key={val} className="flex items-baseline">
          <span className="text-3xl md:text-4xl font-bold tracking-tight font-mono text-zinc-100">
            {num}
          </span>
          <span className="ml-1.5 text-base md:text-lg font-medium font-mono text-teal-400">
            ×
          </span>
        </div>
      );
    }

    return (
      <span key={val} className="text-3xl md:text-4xl font-bold tracking-tight font-mono text-zinc-100 inline-block">
        {val}
      </span>
    );
  };

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 md:gap-5">
      {displayCards.map((card, idx) => {
        const isStandby = card.value === "--";
        const icon = cardIcons[idx % cardIcons.length];

        return (
          <div
            key={idx}
            className="bg-[#121215] border border-zinc-800 hover:border-zinc-700/80 rounded-2xl p-5 md:p-6 flex flex-col justify-between transition-all duration-200 ease-out shadow-[0_4px_20px_rgba(0,0,0,0.35)] hover:shadow-[0_8px_28px_rgba(0,0,0,0.5)] cursor-default"
          >
            {/* Header: Label & Status Indicator */}
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-2.5">
                <div className="w-7 h-7 rounded-lg bg-zinc-900 border border-zinc-800 flex items-center justify-center text-zinc-400">
                  {icon}
                </div>
                <span className="text-xs font-medium text-zinc-300">
                  {card.title.split("(")[0].trim()}
                </span>
              </div>

              <div
                className={`flex items-center gap-1.5 text-[11px] font-mono px-2 py-0.5 rounded-md border transition-colors ${
                  isStandby
                    ? "border-zinc-800 bg-zinc-900/60 text-zinc-500"
                    : "border-teal-500/20 bg-teal-500/10 text-teal-400"
                }`}
              >
                {!isStandby && card.trendType === "up" ? (
                  <TrendingUp className="w-3 h-3 text-teal-400" />
                ) : !isStandby && card.trendType === "down" ? (
                  <TrendingDown className="w-3 h-3 text-teal-400" />
                ) : (
                  <Clock className="w-3 h-3 opacity-60 text-zinc-500" />
                )}
                <span>{card.trend}</span>
              </div>
            </div>

            {/* Metric Display */}
            <div className="my-4 md:my-5">
              {formatValue(card.value)}
            </div>

            {/* Footer with clean hierarchy */}
            <div className="pt-3 border-t border-zinc-800/80 space-y-0.5">
              <div className={`text-xs font-medium ${isStandby ? "text-zinc-500" : "text-zinc-300"}`}>
                <span>{card.trendText}</span>
              </div>
              <p className="text-[11px] text-zinc-500 leading-snug">
                {card.subText}
              </p>
            </div>
          </div>
        );
      })}
    </div>
  );
}
