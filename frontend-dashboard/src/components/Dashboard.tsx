import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import { fetchDashboardSummary } from "../api/endpoints";
import type { DashboardSummary, RiskZone } from "../api/types";
import { useAuth } from "../context/AuthContext";
import AlertsConsole from "./AlertsConsole";
import ReportsPanel from "./ReportsPanel";
import RiskMap from "./RiskMap";
import SummaryCards from "./SummaryCards";
import ZoneDetail from "./ZoneDetail";

type Tab = "map" | "reports" | "alerts";

export default function Dashboard() {
  const { t, i18n } = useTranslation();
  const { logout } = useAuth();
  const [tab, setTab] = useState<Tab>("map");
  const [selectedZone, setSelectedZone] = useState<RiskZone | null>(null);
  const [summary, setSummary] = useState<DashboardSummary | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);

  useEffect(() => {
    let cancelled = false;
    fetchDashboardSummary()
      .then((data) => {
        if (!cancelled) setSummary(data);
      })
      .catch(() => {
        if (!cancelled) setSummary(null);
      });
    return () => {
      cancelled = true;
    };
  }, [refreshKey]);

  const changeLanguage = (lang: string) => {
    i18n.changeLanguage(lang);
  };

  return (
    <div className="dashboard">
      <header className="app-header">
        <div className="app-title">
          <h1>{t("app.title")}</h1>
          <span className="app-subtitle">{t("app.subtitle")}</span>
        </div>
        <div className="header-actions">
          <select
            className="lang-select"
            value={i18n.language}
            onChange={(e) => changeLanguage(e.target.value)}
            aria-label="Language"
          >
            <option value="en">English</option>
            <option value="hi">हिंदी</option>
          </select>
          <button className="btn-ghost" onClick={logout}>
            {t("app.logout")}
          </button>
          <button
            className="btn-ghost"
            onClick={() => setRefreshKey((k) => k + 1)}
          >
            {t("app.refresh")}
          </button>
        </div>
      </header>

      {summary && <SummaryCards summary={summary} />}

      <nav className="tabs">
        <button
          className={`tab ${tab === "map" ? "active" : ""}`}
          onClick={() => setTab("map")}
        >
          {t("nav.map")}
        </button>
        <button
          className={`tab ${tab === "reports" ? "active" : ""}`}
          onClick={() => setTab("reports")}
        >
          {t("nav.reports")}
        </button>
        <button
          className={`tab ${tab === "alerts" ? "active" : ""}`}
          onClick={() => setTab("alerts")}
        >
          {t("nav.alerts")}
        </button>
      </nav>

      {tab === "map" && (
        <div className="map-layout">
          <RiskMap onSelectZone={setSelectedZone} refreshKey={refreshKey} />
          {selectedZone ? (
            <ZoneDetail
              zone={selectedZone}
              onClose={() => setSelectedZone(null)}
              refreshKey={refreshKey}
            />
          ) : (
            <div className="zone-detail zone-detail-empty">
              <p className="muted">{t("map.selectHint")}</p>
            </div>
          )}
        </div>
      )}

      {tab === "reports" && <ReportsPanel refreshKey={refreshKey} />}
      {tab === "alerts" && <AlertsConsole refreshKey={refreshKey} />}
    </div>
  );
}