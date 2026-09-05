import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import { fetchReports } from "../api/endpoints";
import type { FieldReport, SyncStatus } from "../api/types";

const STATUS_ORDER: Array<SyncStatus | ""> = ["", "pending", "synced", "conflict"];

interface ReportsPanelProps {
  refreshKey: number;
}

export default function ReportsPanel({ refreshKey }: ReportsPanelProps) {
  const { t } = useTranslation();
  const [reports, setReports] = useState<FieldReport[]>([]);
  const [status, setStatus] = useState<SyncStatus | "">("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetchReports(status || undefined)
      .then((data) => {
        if (!cancelled) {
          setReports(data);
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
  }, [status, refreshKey]);

  return (
    <section className="panel">
      <div className="panel-header">
        <h2>{t("reports.title")}</h2>
        <div className="filter-row">
          {STATUS_ORDER.map((s) => (
            <button
              key={s || "all"}
              className={`filter-chip ${status === s ? "active" : ""}`}
              onClick={() => setStatus(s)}
            >
              {s === "" ? t("reports.filterAll") : t(`reports.filter${capitalize(s)}`)}
            </button>
          ))}
        </div>
      </div>

      {loading && <p className="muted">…</p>}
      {error && <p className="error-text">{t("map.error")}</p>}

      {!loading && !error && reports.length === 0 && (
        <p className="muted">{t("reports.noResults")}</p>
      )}

      <ul className="report-list">
        {reports.map((r) => (
          <li key={r.id} className="report-item">
            {r.photo_url ? (
              <img className="report-thumb" src={r.photo_url} alt={r.report_type} />
            ) : (
              <div className="report-thumb report-thumb-empty" />
            )}
            <div className="report-body">
              <div className="report-title-row">
                <strong>{r.report_type}</strong>
                <span className={`status-badge status-${r.sync_status}`}>
                  {t(`reports.status_${r.sync_status}`)}
                </span>
              </div>
              <p>{r.description}</p>
              <p className="muted small">
                {t("common.reporter")}: {r.user_phone} · {t("common.submittedAt")}:{" "}
                {new Date(r.submitted_at).toLocaleString()}
              </p>
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}