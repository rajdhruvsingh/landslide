import axios from "axios";

import { getAuthToken } from "./tokenStore";

// Backend base URL. Falls back to "/api/v1" (served through the Vite dev proxy).
export const API_BASE_URL: string =
  import.meta.env.VITE_API_BASE_URL ?? "/api/v1";

// Single shared axios instance for every backend call.
//
// SECURITY NOTE (deliberate choice):
// The bearer token is held in memory ONLY (via AuthContext + tokenStore) and
// injected per request here. We intentionally do NOT persist it to
// localStorage/sessionStorage: those survive e.g. XSS-driven exfiltration
// and make "the user is logged in" persist longer than the user intends.
// The cost is that a page refresh returns the user to the login screen —
// acceptable for the authority dashboard; re-auth via OTP is cheap.
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 15000,
  headers: { "Content-Type": "application/json" },
});

api.interceptors.request.use((config) => {
  const token = getAuthToken();
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export default api;