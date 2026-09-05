import { useTranslation } from "react-i18next";

import type { DashboardSummary } from "../api/types";

interface SummaryCardsProps {
  summary: DashboardSummary;
}

export default function SummaryCards({ summary }: SummaryCardsProps) {
  const { t } = useTranslation();
  const rc = summary.risk_counts;

  return (
    <div className="summary-cards">
      <div className="summary-card">
        <span className="summary-value">{summary.total_zones}</span>
        <span className="summary-label">{t("summary.totalZones")}</span>
      </div>

      <div className="summary-card">
        <span className="summary-value">{t("summary.riskDistribution")}</span>
        <div className="risk-distribution">
          <span className="dot dot-low">{rc.Low}</span>
          <span className="dot dot-moderate">{rc.Moderate}</span>
          <span className="dot dot-high">{rc.High}</span>
          <span className="dot dot-severe">{rc.Severe}</span>
        </div>
      </div>

      <div className="summary-card">
        <span className="summary-value">{summary.active_alerts}</span>
        <span className="summary-label">{t("summary.activeAlerts")}</span>
      </div>

      <div className="summary-card">
        <span className="summary-value">{summary.reports_pending}</span>
        <span className="summary-label">{t("summary.pendingReports")}</span>
      </div>
    </div>
  );
}