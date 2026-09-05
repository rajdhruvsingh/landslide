import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import { fetchAlerts } from "../api/endpoints";
import type { Alert } from "../api/types";

const ALERT_BADGE: Record<string, string> = {
  Low: "badge-low",
  Moderate: "badge-moderate",
  High: "badge-high",
  Severe: "badge-severe",
};

interface AlertsConsoleProps {
  refreshKey: number;
}

export default function AlertsConsole({ refreshKey }: AlertsConsoleProps) {
  const { t } = useTranslation();
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetchAlerts()
      .then((data) => {
        if (!cancelled) {
          setAlerts(data);
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

  return (
    <section className="panel">
      <div className="panel-header">
        <h2>{t("alerts.title")}</h2>
      </div>

      {loading && <p className="muted">…</p>}
      {error && <p className="error-text">{t("map.error")}</p>}

      {!loading && !error && alerts.length === 0 && (
        <p className="muted">{t("alerts.noAlerts")}</p>
      )}

      <ul className="alert-list">
        {alerts.map((a) => (
          <li key={a.id} className="alert-item">
            <div className="alert-title-row">
              <span className={`risk-badge ${ALERT_BADGE[a.risk_level] ?? "badge-low"}`}>
                {a.risk_level}
              </span>
              <strong>{a.zone_name}</strong>
              <span className="muted small">
                {t("common.channel")}: {a.channel} · {t("common.dispatchedAt")}:{" "}
                {new Date(a.dispatched_at).toLocaleString()}
              </span>
            </div>
            <p className="alert-message">{a.message}</p>
            {a.explanation && <p className="alert-explanation">{a.explanation}</p>}
          </li>
        ))}
      </ul>
    </section>
  );
}