import api from "./client";
import type {
  Alert,
  DashboardSummary,
  FieldReport,
  Paginated,
  RiskZone,
  RiskZoneExplanation,
} from "./types";

export interface LoginResponse {
  access_token: string;
  token_type: string;
}

export async function login(
  phoneNumber: string,
  otp: string
): Promise<LoginResponse> {
  const res = await api.post<LoginResponse>("/auth/login", {
    phone_number: phoneNumber,
    otp,
  });
  return res.data;
}

export async function fetchRiskZones(): Promise<RiskZone[]> {
  const res = await api.get<Paginated<RiskZone>>("/risk-zones/");
  return res.data.results;
}

export async function fetchZoneExplanation(
  zoneId: number | string
): Promise<RiskZoneExplanation> {
  const res = await api.get<RiskZoneExplanation>(
    `/risk-zones/${zoneId}/explanation/`
  );
  return res.data;
}

export async function fetchReports(
  status?: string
): Promise<FieldReport[]> {
  const res = await api.get<Paginated<FieldReport>>("/reports/", {
    params: status ? { status } : {},
  });
  return res.data.results;
}

export async function fetchAlerts(): Promise<Alert[]> {
  const res = await api.get<Paginated<Alert>>("/alerts/");
  return res.data.results;
}

export async function fetchDashboardSummary(): Promise<DashboardSummary> {
  const res = await api.get<DashboardSummary>("/dashboard/summary");
  return res.data;
}