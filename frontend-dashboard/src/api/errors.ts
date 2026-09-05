import type { AxiosError } from "axios";

export function getErrorMessage(err: unknown, fallback: string): string {
  if (axiosErrHasData(err)) {
    const detail = err.response?.data;
    if (detail && typeof detail === "object") {
      const errorField = (detail as { error?: unknown }).error;
      if (typeof errorField === "string") return errorField;
    }
  }
  return fallback;
}

function axiosErrHasData(err: unknown): err is AxiosError {
  return typeof err === "object" && err !== null && "response" in err;
}