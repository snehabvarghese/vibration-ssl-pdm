import { Card, MetricCard } from '../components/ui';

export default function LivePage() {
  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Page Header */}
      <div>
        <div className="flex items-center gap-2">
          <h2 className="text-2xl font-bold text-white tracking-tight">
            Live monitoring (Phase 2)
          </h2>
          <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-brand-500/15 text-brand-300 border border-brand-500/20">
            Coming Soon
          </span>
        </div>
        <p className="text-sm text-gray-500 mt-1">
          Real-time telemetry and streaming diagnostics for industrial machinery.
        </p>
      </div>

      {/* Architecture / Explanation Notice */}
      <div className="bg-brand-500/5 border border-brand-500/15 rounded-lg px-4 py-3 flex items-start gap-2.5">
        <span className="text-brand-400 text-sm shrink-0 mt-0.5">📡</span>
        <div className="text-xs text-gray-400 leading-relaxed">
          <p className="font-medium text-gray-300 mb-1">
            Real-time Sensor Streaming Architecture
          </p>
          <p>
            This page will stream readings from sensors deployed on live machines using edge
            telemetry (MQTT / WebSockets). In Phase 2, incoming sensor bursts are ingested into the
            self-supervised embedding model for real-time anomaly detection, early fault detection
            (EDR), and health state tracking without manual file uploads.
          </p>
        </div>
      </div>

      {/* Static Mock Dashboard Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <div className="flex items-center justify-between">
            <span className="text-xs text-gray-500 font-medium uppercase">Live Status</span>
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-gray-500/10 text-gray-400 border border-gray-500/20">
              <span className="w-2 h-2 rounded-full bg-gray-500" />
              Standby / Idle
            </span>
          </div>
          <p className="text-xs text-gray-500 mt-2">
            Target Machine: <strong className="text-gray-300">Machine-01 (Bearing Drive-End)</strong>
          </p>
        </Card>

        <MetricCard
          label="EDR Score"
          value="-- / --"
          sub="Threshold: Pending connection"
          accent="brand"
          icon="📈"
        />

        <MetricCard
          label="Sampling Rate"
          value="-- kHz"
          sub="Contract: 12 kHz / 48 kHz"
          accent="brand"
          icon="⚡"
        />

        <MetricCard
          label="Last Updated"
          value="--:--:--"
          sub="Awaiting telemetry stream"
          accent="brand"
          icon="⏱️"
        />
      </div>

      {/* Mock Telemetry Stream Placeholder Card */}
      <Card title="Live Vibration Stream (Buffer Placeholder)">
        <div className="h-48 border border-dashed border-white/10 rounded-lg flex flex-col items-center justify-center p-6 text-center">
          <span className="text-3xl mb-2 text-gray-600">📊</span>
          <p className="text-sm font-medium text-gray-300">
            Sensor stream receiver offline
          </p>
          <p className="text-xs text-gray-500 max-w-md mt-1">
            Telemetry ingestion gateway will activate in Phase 2. Live waveform snippets and EDR
            trend will appear here in real time.
          </p>
        </div>
      </Card>
    </div>
  );
}
