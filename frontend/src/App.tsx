import { BrowserRouter, Navigate, Route, Routes, Link } from 'react-router-dom';
import Layout from './components/Layout';
import { ThemeProvider } from './context/ThemeContext';
import DashboardPage from './pages/DashboardPage';
import AnalysePage from './pages/AnalysePage';
import HistoryPage from './pages/HistoryPage';
import LivePage from './pages/LivePage';
import MotorPage from './pages/MotorPage';
import { Card } from './components/ui';

function NotFoundPage() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] text-center">
      <Card className="max-w-md w-full p-8 space-y-4">
        <span className="text-4xl">⚠️</span>
        <h2 className="text-xl font-bold text-white">404 - Page Not Found</h2>
        <p className="text-sm text-gray-400">
          The page you requested does not exist in the diagnostics system.
        </p>
        <Link
          to="/dashboard"
          className="inline-flex items-center justify-center px-4 py-2 text-sm font-medium text-white bg-brand-500 rounded-lg hover:bg-brand-600 transition-colors"
        >
          Return to Dashboard
        </Link>
      </Card>
    </div>
  );
}

export default function App() {
  return (
    <ThemeProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="/dashboard" element={<DashboardPage />} />
            <Route path="/analyse" element={<AnalysePage />} />
            <Route path="/motor" element={<MotorPage />} />
            <Route path="/history" element={<HistoryPage />} />
            <Route path="/live" element={<LivePage />} />
            <Route path="*" element={<NotFoundPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ThemeProvider>
  );
}

