"use client";

import React, { useEffect, useRef, useState } from "react";
import { Layers, MapPin, Crosshair, Sparkles, Navigation } from "lucide-react";

interface SatelliteMapProps {
  aoi: [number, number, number, number]; // [minLon, minLat, maxLon, maxLat]
  onAOIChange: (bbox: [number, number, number, number], name?: string) => void;
  sentinelFootprint?: any;
  cartosatFootprint?: any;
}

const PRESET_AOIS = [
  { name: "New Delhi / NCR", bbox: [77.10, 28.55, 77.30, 28.75] as [number, number, number, number] },
  { name: "Chennai Coastal", bbox: [80.15, 12.95, 80.35, 13.15] as [number, number, number, number] },
  { name: "Mumbai Urban", bbox: [72.80, 18.90, 73.00, 19.10] as [number, number, number, number] },
  { name: "Hyderabad Tech", bbox: [78.30, 17.35, 78.50, 17.50] as [number, number, number, number] },
];

export function SatelliteMap({ aoi, onAOIChange, sentinelFootprint, cartosatFootprint }: SatelliteMapProps) {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<any>(null);
  const aoiLayerRef = useRef<any>(null);
  const s2LayerRef = useRef<any>(null);
  const cartosatLayerRef = useRef<any>(null);
  const [activeLayer, setActiveLayer] = useState<"dark" | "satellite">("dark");
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
  }, []);

  useEffect(() => {
    if (!isClient || !mapContainerRef.current) return;

    let map = mapInstanceRef.current;
    if (!map) {
      const L = require("leaflet");

      const centerLat = (aoi[1] + aoi[3]) / 2;
      const centerLon = (aoi[0] + aoi[2]) / 2;

      map = L.map(mapContainerRef.current, {
        center: [centerLat, centerLon],
        zoom: 11,
        zoomControl: false,
        attributionControl: false,
      });

      L.control.zoom({ position: "bottomright" }).addTo(map);

      // Dark Matter CartoDB tiles
      const darkTiles = L.tileLayer(
        "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        { maxZoom: 19, subdomains: "abcd" }
      );

      darkTiles.addTo(map);
      mapInstanceRef.current = map;
      (map as any)._darkTiles = darkTiles;
    }

    return () => {
      // Keep map cached
    };
  }, [isClient]);

  // Update AOI and Footprints
  useEffect(() => {
    if (!mapInstanceRef.current || typeof window === "undefined") return;
    const L = require("leaflet");
    const map = mapInstanceRef.current;

    // Draw AOI Box
    if (aoiLayerRef.current) {
      map.removeLayer(aoiLayerRef.current);
    }

    const bounds: [[number, number], [number, number]] = [
      [aoi[1], aoi[0]],
      [aoi[3], aoi[2]],
    ];

    const aoiRect = L.rectangle(bounds, {
      color: "#14b8a6",
      weight: 2,
      dashArray: "4, 4",
      fillColor: "#14b8a6",
      fillOpacity: 0.12,
    }).addTo(map);

    aoiLayerRef.current = aoiRect;

    // Center map on AOI smoothly
    map.fitBounds(bounds, { padding: [25, 25], maxZoom: 13 });

    // Render Sentinel-2 Footprint
    if (s2LayerRef.current) map.removeLayer(s2LayerRef.current);
    if (sentinelFootprint && sentinelFootprint.geometry) {
      s2LayerRef.current = L.geoJSON(sentinelFootprint.geometry, {
        style: { color: "#38bdf8", weight: 2, fillOpacity: 0.08 }
      }).addTo(map);
    }

    // Render Cartosat Footprint
    if (cartosatLayerRef.current) map.removeLayer(cartosatLayerRef.current);
    if (cartosatFootprint && cartosatFootprint.geometry) {
      cartosatLayerRef.current = L.geoJSON(cartosatFootprint.geometry, {
        style: { color: "#f59e0b", weight: 2, dashArray: "2, 4", fillOpacity: 0.1 }
      }).addTo(map);
    }
  }, [aoi, sentinelFootprint, cartosatFootprint]);

  const toggleBasemap = () => {
    if (!mapInstanceRef.current) return;
    const L = require("leaflet");
    const map = mapInstanceRef.current;

    if (activeLayer === "dark") {
      if ((map as any)._darkTiles) map.removeLayer((map as any)._darkTiles);
      const satTiles = L.tileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        { maxZoom: 18 }
      ).addTo(map);
      (map as any)._satTiles = satTiles;
      setActiveLayer("satellite");
    } else {
      if ((map as any)._satTiles) map.removeLayer((map as any)._satTiles);
      const darkTiles = L.tileLayer(
        "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        { maxZoom: 19, subdomains: "abcd" }
      ).addTo(map);
      (map as any)._darkTiles = darkTiles;
      setActiveLayer("dark");
    }
  };

  return (
    <div className="relative w-full h-[280px] rounded-xl overflow-hidden border border-white/10 bg-[#0d121c]">
      <div ref={mapContainerRef} className="w-full h-full" />

      {/* Preset AOI Selector Chips */}
      <div className="absolute top-3 left-3 z-[1000] flex flex-wrap gap-1.5 max-w-[80%]">
        {PRESET_AOIS.map((preset) => {
          const isSelected =
            Math.abs(preset.bbox[0] - aoi[0]) < 0.01 &&
            Math.abs(preset.bbox[1] - aoi[1]) < 0.01;
          return (
            <button
              key={preset.name}
              onClick={() => onAOIChange(preset.bbox, preset.name)}
              className={`text-[11px] px-2.5 py-1 rounded-md transition-all flex items-center gap-1 font-medium ${
                isSelected
                  ? "bg-teal-500/20 text-teal-300 border border-teal-500/40 shadow-sm"
                  : "bg-black/60 backdrop-blur-md text-zinc-400 hover:text-zinc-200 border border-white/10"
              }`}
            >
              <MapPin className="w-2.5 h-2.5" />
              {preset.name}
            </button>
          );
        })}
      </div>

      {/* Map Controls */}
      <div className="absolute top-3 right-3 z-[1000] flex gap-1.5">
        <button
          onClick={toggleBasemap}
          className="text-[11px] px-2.5 py-1 rounded-md bg-black/70 backdrop-blur-md text-zinc-300 hover:text-white border border-white/10 flex items-center gap-1 font-medium transition-all"
        >
          <Layers className="w-3 h-3 text-teal-400" />
          {activeLayer === "dark" ? "Basemap: Dark" : "Basemap: Satellite"}
        </button>
      </div>

      {/* AOI Bounding Coordinate Badge */}
      <div className="absolute bottom-3 left-3 z-[1000] bg-black/75 backdrop-blur-md border border-white/10 px-2.5 py-1 rounded-md text-[10px] font-mono text-zinc-400 flex items-center gap-2">
        <Crosshair className="w-3 h-3 text-teal-400" />
        <span>
          AOI: [{aoi[0].toFixed(2)}°E, {aoi[1].toFixed(2)}°N] to [{aoi[2].toFixed(2)}°E, {aoi[3].toFixed(2)}°N]
        </span>
        <span className="text-zinc-600">|</span>
        <span className="text-teal-400">EPSG:4326</span>
      </div>
    </div>
  );
}
