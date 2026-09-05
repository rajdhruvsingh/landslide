import { useAuth } from "./context/AuthContext";
import Dashboard from "./components/Dashboard";
import Login from "./components/Login";

function App() {
  const { token } = useAuth();
  return token ? <Dashboard /> : <Login />;
}

export default App;