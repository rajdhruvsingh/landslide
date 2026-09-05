import { useState, type FormEvent } from "react";
import { useTranslation } from "react-i18next";

import { getErrorMessage } from "../api/errors";
import { useAuth } from "../context/AuthContext";

export default function Login() {
  const { t } = useTranslation();
  const { login } = useAuth();
  const [phoneNumber, setPhoneNumber] = useState("");
  const [otp, setOtp] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      await login(phoneNumber.trim(), otp.trim());
    } catch (err) {
      setError(getErrorMessage(err, t("login.errorDefault")));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="login-page">
      <form className="login-card" onSubmit={onSubmit}>
        <h1>{t("app.title")}</h1>
        <h2 className="login-subtitle">{t("app.subtitle")}</h2>

        <label htmlFor="phone">{t("login.phone")}</label>
        <input
          id="phone"
          type="tel"
          autoComplete="tel"
          required
          value={phoneNumber}
          onChange={(e) => setPhoneNumber(e.target.value)}
          placeholder="9876543210"
        />

        <label htmlFor="otp">{t("login.otp")}</label>
        <input
          id="otp"
          type="text"
          autoComplete="one-time-code"
          required
          value={otp}
          onChange={(e) => setOtp(e.target.value)}
          placeholder="000000"
        />

        {error && <p className="error-text">{error}</p>}

        <button type="submit" disabled={submitting} className="btn-primary">
          {submitting ? "…" : t("login.submit")}
        </button>

        <p className="hint-text">{t("login.hint")}</p>
      </form>
    </div>
  );
}