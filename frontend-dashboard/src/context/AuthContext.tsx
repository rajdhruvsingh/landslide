import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import * as api from "../api/endpoints";
import { setAuthToken } from "../api/tokenStore";

interface AuthContextValue {
  token: string | null;
  login: (phoneNumber: string, otp: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setTokenState] = useState<string | null>(null);

  const login = useCallback(async (phoneNumber: string, otp: string) => {
    const res = await api.login(phoneNumber, otp);
    // In-memory token storage — see src/api/client.ts for the rationale.
    setAuthToken(res.access_token);
    setTokenState(res.access_token);
  }, []);

  const logout = useCallback(() => {
    setAuthToken(null);
    setTokenState(null);
  }, []);

  const value = useMemo(
    () => ({ token, login, logout }),
    [token, login, logout]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within an AuthProvider");
  return ctx;
}