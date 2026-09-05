import ReactDOM from "react-dom/client";

import "./i18n";
import "./index.css";
import App from "./App";
import { AuthProvider } from "./context/AuthContext";

// NOTE: React StrictMode is intentionally omitted. react-leaflet's
// MapContainer double-mounts under StrictMode in React 18 dev and can raise
// "Map container is already initialized".
ReactDOM.createRoot(document.getElementById("root")!).render(
  <AuthProvider>
    <App />
  </AuthProvider>
);