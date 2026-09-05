export type RiskLevel = "Low" | "Moderate" | "High" | "Severe";
export type SyncStatus = "synced" | "pending" | "conflict";

export interface Paginated<T> {
  count: number;
  next: string | null;
  previous: string | null;
  results: T[];
}

export interface RiskZone {
  id: number;
  zone_name: string;
  district: string;
  state: string;
  geom: GeoJSON.Polygon | null;
  current_risk_level: RiskLevel;
  last_computed_at: string | null;
}

export interface RiskZoneExplanation {
  zone_id: number;
  zone_name: string;
  risk_level: RiskLevel;
  explanation: string;
  thresholds_checked: unknown[];
  actual_readings: {
    date: string | null;
    rainfall_mm: number | null;
    soil_moisture_pct: number | null;
  }[];
}

export interface FieldReport {
  id: number;
  user: number;
  user_phone: string;
  geom: unknown | null;
  photo_url: string;
  video_url: string;
  description: string;
  report_type: string;
  submitted_at: string;
  sync_status: SyncStatus;
}

export interface Alert {
  id: number;
  zone: number;
  zone_name: string;
  risk_level: RiskLevel;
  message: string;
  language: string;
  channel: string;
  dispatched_at: string;
  explanation: string;
}

export interface RiskCounts {
  Low: number;
  Moderate: number;
  High: number;
  Severe: number;
}

export interface DashboardSummary {
  total_zones: number;
  risk_counts: RiskCounts;
  active_alerts: number;
  reports_pending: number;
}