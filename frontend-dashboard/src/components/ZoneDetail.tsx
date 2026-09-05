import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import { getErrorMessage } from "../api/errors";
import { fetchZoneExplanation } from "../api/endpoints";
import type { RiskZone, RiskZoneExplanation } from "../api/types";

const RISK_BADGE_CLASSES: Record<string, string> = {
  Low: "badge-low",
  Moderate: "badge-moderate",
  High: "badge-high",
  Severe: "badge-severe",
};

interface ZoneDetailProps {
  zone: RiskZone;
  onClose: () => void;
  refreshKey: number;
}

export default function ZoneDetail({ zone, onClose, refreshKey }: ZoneDetailProps) {
  const { t } = useTranslation();
  const [explanation, setExplanation] = useState<RiskZoneExplanation | null>(
    null
  );
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetchZoneExplanation(zone.id)
      .then((data) => {
        if (!cancelled) {
          setExplanation(data);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(getErrorMessage(err, t("zone.error")));
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [zone.id, refreshKey, t]);

  const badgeLevel =
    explanation?.risk_level ?? zone.current_risk_level;
  const badgeClass = RISK_BADGE_CLASSES[badgeLevel] ?? "badge-low";

  return (
    <aside className="zone-detail">
      <div className="zone-detail-header">
        <div>
          <h3>{zone.zone_name}</h3>
          <p className="zone-meta">
            {zone.district}
            {zone.state ? `, ${zone.state}` : ""}
          </p>
        </div>
        <button className="btn-ghost" onClick={onClose}>
          ✕
        </button>
      </div>

      <p className={`risk-badge ${badgeClass}`}>
        {t("zone.riskLevel")}: {badgeLevel}
      </p>

      {loading && <p className="muted">{t("zone.loading")}</p>}

      {error && <p className="error-text">{error}</p>}

      {!loading && !error && explanation && (
        <>
          <h4>{t("zone.explanationTitle")}</h4>
          {/* Core differentiator — the explanation is the headline of this panel. */}
          <div className="explanation-block">
            {explanation.explanation || t("zone.explanationEmpty")}
          </div>

          {explanation.thresholds_checked.length > 0 && (
            <>
              <h4>{t("zone.thresholdsTitle")}</h4>
              <ul className="threshold-list">
                {explanation.thresholds_checked.map((th, i) => (
                  <li key={i}>
                    <pre>{JSON.stringify(th, null, 2)}</pre>
                  </li>
                ))}
              </ul>
            </>
          )}

          <h4>{t("zone.readingsTitle")}</h4>
          <table className="readings-table">
            <thead>
              <tr>
                <th>{t("common.submittedAt")}</th>
                <th>Rainfall (mm)</th>
                <th>Soil moisture (%)</th>
              </tr>
            </thead>
            <tbody>
              {explanation.actual_readings.map((r, i) => (
                <tr key={i}>
                  <td>{r.date ?? "—"}</td>
                  <td>{r.rainfall_mm ?? "—"}</td>
                  <td>{r.soil_moisture_pct ?? "—"}</td>
                </tr>
              ))}
              {explanation.actual_readings.length === 0 && (
                <tr>
                  <td colSpan={3} className="muted">
                    —
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </>
      )}
    </aside>
  );
}