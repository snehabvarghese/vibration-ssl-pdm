import type { ReactNode } from 'react';

interface StatusBadgeProps {
  anomalous: boolean;
  className?: string;
}

export function StatusBadge({ anomalous, className = '' }: StatusBadgeProps) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider ${
        anomalous
          ? 'bg-anomalous/15 text-anomalous border border-anomalous/20'
          : 'bg-healthy/15 text-healthy border border-healthy/20'
      } ${className}`}
    >
      <span
        className={`w-2 h-2 rounded-full ${anomalous ? 'bg-anomalous animate-pulse' : 'bg-healthy'}`}
      />
      {anomalous ? 'Anomalous' : 'Normal'}
    </span>
  );
}

interface CardProps {
  title?: string;
  children: ReactNode;
  className?: string;
}

export function Card({ title, children, className = '' }: CardProps) {
  return (
    <div className={`glass-card p-5 ${className}`}>
      {title && (
        <h3 className="text-sm font-semibold text-gray-300 mb-3 tracking-wide uppercase">
          {title}
        </h3>
      )}
      {children}
    </div>
  );
}

interface MetricCardProps {
  label: string;
  value: string | number;
  sub?: string;
  icon?: ReactNode;
  accent?: 'brand' | 'healthy' | 'anomalous' | 'warning';
}

export function MetricCard({ label, value, sub, icon, accent = 'brand' }: MetricCardProps) {
  const accentColor = {
    brand: 'text-brand-400',
    healthy: 'text-healthy',
    anomalous: 'text-anomalous',
    warning: 'text-warning',
  }[accent];

  return (
    <div className="glass-card p-4 flex items-start gap-3">
      {icon && (
        <div className={`text-2xl ${accentColor} mt-0.5`}>{icon}</div>
      )}
      <div className="min-w-0">
        <p className="text-xs text-gray-500 font-medium uppercase tracking-wider">{label}</p>
        <p className={`text-xl font-bold mt-0.5 ${accentColor}`}>{value}</p>
        {sub && <p className="text-xs text-gray-500 mt-0.5">{sub}</p>}
      </div>
    </div>
  );
}

export function Spinner({ size = 'md' }: { size?: 'sm' | 'md' | 'lg' }) {
  const s = { sm: 'w-4 h-4', md: 'w-6 h-6', lg: 'w-10 h-10' }[size];
  return (
    <div
      className={`${s} border-2 border-brand-500/30 border-t-brand-400 rounded-full animate-spin`}
    />
  );
}

interface ErrorBannerProps {
  message: string;
  onDismiss?: () => void;
}

export function ErrorBanner({ message, onDismiss }: ErrorBannerProps) {
  return (
    <div className="bg-anomalous/10 border border-anomalous/20 rounded-lg p-4 flex items-start gap-3">
      <span className="text-anomalous text-lg shrink-0">⚠️</span>
      <div className="flex-1 min-w-0">
        <p className="text-sm text-anomalous font-medium">Analysis Error</p>
        <p className="text-sm text-gray-300 mt-1 break-words">{message}</p>
      </div>
      {onDismiss && (
        <button
          onClick={onDismiss}
          className="text-gray-500 hover:text-gray-300 text-sm shrink-0"
        >
          ✕
        </button>
      )}
    </div>
  );
}
