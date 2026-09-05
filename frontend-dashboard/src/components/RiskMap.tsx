import { useEffect, useMemo, useState } from "react";
import { GeoJSON, MapContainer, TileLayer } from "react-leaflet";
import type { Feature, FeatureCollection, Polygon } from "geojson";
import { useTranslation } from "react-i18next";

import { fetchRiskZones } from "../api/endpoints";
import type { RiskLevel, RiskZone } from "../api/types";

const RISK_COLORS: Record<RiskLevel, string> = {
  Low: "#4caf50",
  Moderate: "#ffc107",
  High: "#ff9800",
  Severe: "#e53935",
};

interface RiskMapProps {
  onSelectZone: (zone: RiskZone) => void;
  refreshKey: number;
}

interface ZoneFeatureProps {
  zone: RiskZone;
}

export default function RiskMap({ onSelectZone, refreshKey }: RiskMapProps) {
  const { t } = useTranslation();
  const [zones, setZones] = useState<RiskZone[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetchRiskZones()
      .then((data) => {
        if (!cancelled) {
          setZones(data);
          setLoading(false);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setError(true);
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [refreshKey]);

  const geo = useMemo<FeatureCollection<Polygon, ZoneFeatureProps>>(
    () => ({
      type: "FeatureCollection",
      features: zones
        .filter((z) => z.geom)
        .map((zone) => ({
          type: "Feature",
          properties: { zone },
          geometry: zone.geom as Polygon,
        })),
    }),
    [zones]
  );

  const style = (feature?: Feature) => {
    const zone = (feature?.properties as ZoneFeatureProps | undefined)?.zone;
    const level: RiskLevel = zone?.current_risk_level ?? "Low";
    const color = RISK_COLORS[level] ?? "#9e9e9e";
    return {
      color: "#333",
      weight: 1.5,
      fillColor: color,
      fillOpacity: 0.6,
    };
  };

  const onEachFeature = (feature: Feature, layer: L.Layer) => {
    const zone = (feature.properties as ZoneFeatureProps).zone;
    layer.bindTooltip(`${zone.zone_name} — ${zone.current_risk_level}`);
    layer.on("click", () => onSelectZone(zone));
  };

  return (
    <div className="map-canvas">
      {loading && <div className="map-message">{t("map.loading")}</div>}
      {error && <div className="map-message map-error">{t("map.error")}</div>}
      {!loading && !error && (
        <MapContainer
          center={[25.57, 91.88]}
          zoom={8}
          className="leaflet-canvas"
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          <GeoJSON data={geo} style={style} onEachFeature={onEachFeature} />
        </MapContainer>
      )}
    </div>
  );
}