import { useEffect, useRef, useState, useCallback } from 'react';
import { Card, MetricCard, Spinner, ErrorBanner } from '../components/ui';
import { MultimodalTrendChart, type MultimodalTrendPoint } from '../components/Charts';
import { getReplayScenarios, WS_BASE_URL } from '../api/client';
import type {
  FusionResult,
  ModelResult,
  ReplayMessage,
  ScenarioMetadata,
} from '../types/api';

const FALLBACK_SCENARIOS: ScenarioMetadata[] = [
  {
    id: 'healthy_baseline',
    name: 'Healthy Baseline',
    description: 'Continuous nominal operation across all three sensor modalities.',
    total_steps: 60,
    duration_seconds: 60,
    is_replay: true,
  },
  {
    id: 'vibration_fault_onset',
    name: 'Vibration Fault Onset',
    description: 'Nominal baseline (0-19s), mechanical impact onset (20-39s), outer race defect (40-60s).',
    total_steps: 60,
    duration_seconds: 60,
    is_replay: true,
  },
  {
    id: 'audio_fault_onset',
    name: 'Acoustic Anomaly Onset',
    description: 'Nominal acoustic operation (0-19s) followed by progressive high-frequency hiss/rumble.',
    total_steps: 60,
    duration_seconds: 60,
    is_replay: true,
  },
  {
    id: 'motor_fault_onset',
    name: 'Motor Electrical & Thermal Fault',
    description: 'Balanced stator current (0-19s) with subsequent phase unbalance and thermal rise (20-60s).',
    total_steps: 60,
    duration_seconds: 60,
    is_replay: true,
  },
  {
    id: 'multi_fault_escalation',
    name: 'Multi-Fault Cascading Breakdown',
    description: 'Healthy (0-14s) -> Motor thermal warning (15-29s) -> Acoustic anomaly (30-44s) -> Bearing failure (45-60s).',
    total_steps: 60,
    duration_seconds: 60,
    is_replay: true,
  },
];

interface WarningLogEntry {
  id: string;
  timestamp: string;
  type: 'status_change' | 'fusion_reason';
  title: string;
  description: string;
  severity: 'normal' | 'warning' | 'alarm';
}

type PlaybackStatus = 'idle' | 'connecting' | 'playing' | 'paused' | 'stopped' | 'completed' | 'error';

interface ModelCardProps {
  title: string;
  modality: 'vibration' | 'audio' | 'motor';
  model?: ModelResult;
  icon: string;
}

function ModelCard({ title, modality, model, icon }: ModelCardProps) {
  const isUnavailable = !model || model.status === 'UNAVAILABLE' || !!model.error;
  const isAnomalous = model?.status === 'ANOMALOUS';

  // Calculate score vs threshold bar metrics
  const score = model?.anomaly_score ?? 0;
  const threshold = model?.threshold ?? 1.0;
  const ratio = model?.score_ratio ?? 0;

  // Visual bar: range up to max(2.0, ratio * 1.15)
  const maxRange = Math.max(2.0, ratio * 1.15, 1.2);
  const thresholdPercent = Math.min(100, Math.max(5, (1.0 / maxRange) * 100));
  const fillPercent = isUnavailable
    ? 0
    : Math.min(100, Math.max(0, (ratio / maxRange) * 100));

  const statusBadge = () => {
    if (isUnavailable) {
      return (
        <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-gray-500/15 text-gray-400 border border-gray-500/30">
          <span className="w-1.5 h-1.5 rounded-full bg-gray-500" />
          UNAVAILABLE
        </span>
      );
    }
    if (isAnomalous) {
      return (
        <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-rose-500/15 text-rose-400 border border-rose-500/30">
          <span className="w-1.5 h-1.5 rounded-full bg-rose-500 animate-pulse" />
          ANOMALOUS
        </span>
      );
    }
    return (
      <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/15 text-emerald-400 border border-emerald-500/30">
        <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
        NORMAL
      </span>
    );
  };

  const formattedTime = model?.timestamp
    ? new Date(model.timestamp).toLocaleTimeString([], { hour12: false })
    : '--:--:--';

  return (
    <Card className="flex flex-col justify-between space-y-4">
      {/* Header */}
      <div>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-xl">{icon}</span>
            <h4 className="text-sm font-bold text-gray-200 tracking-tight">{title}</h4>
          </div>
          {statusBadge()}
        </div>

        <p className="text-[11px] text-gray-400 mt-1 capitalize">
          Modality: <span className="font-mono text-gray-300">{modality}</span>
          {model?.model_name && ` • ${model.model_name}`}
        </p>
      </div>

      {/* Main Content */}
      {isUnavailable ? (
        <div className="bg-rose-500/10 border border-rose-500/20 rounded-lg p-3 text-xs text-rose-300 space-y-1">
          <div className="flex items-center gap-1.5 font-semibold text-rose-400">
            <span>⚠️</span> Model Unavailable
          </div>
          <p className="text-gray-300">
            {model?.error || 'Sensor stream or model inference is currently offline.'}
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {/* Numbers: Score vs Threshold */}
          <div className="flex items-baseline justify-between text-xs">
            <div>
              <span className="text-gray-400">Score: </span>
              <span className="font-mono font-bold text-sm text-gray-100">
                {score.toFixed(3)}
              </span>
            </div>
            <div>
              <span className="text-gray-400">Threshold: </span>
              <span className="font-mono font-semibold text-gray-300">
                {threshold.toFixed(3)}
              </span>
            </div>
            <div>
              <span className="text-gray-400">Ratio: </span>
              <span
                className={`font-mono font-bold ${
                  ratio >= 1.0 ? 'text-rose-400' : 'text-emerald-400'
                }`}
              >
                {ratio.toFixed(2)}x
              </span>
            </div>
          </div>

          {/* Visual Score vs Threshold Bar */}
          <div className="space-y-1">
            <div className="relative w-full h-3 bg-surface-2 dark:bg-white/5 rounded-full overflow-hidden border border-white/10">
              {/* Score Fill */}
              <div
                className={`h-full rounded-full transition-all duration-300 ${
                  ratio >= 1.0
                    ? 'bg-gradient-to-r from-amber-500 to-rose-500'
                    : 'bg-gradient-to-r from-emerald-500 to-teal-400'
                }`}
                style={{ width: `${fillPercent}%` }}
              />

              {/* Threshold Indicator Line */}
              <div
                className="absolute top-0 bottom-0 w-0.5 bg-rose-400 z-10 shadow-sm"
                style={{ left: `${thresholdPercent}%` }}
                title={`Threshold: 1.0 (${threshold.toFixed(3)})`}
              />
            </div>

            <div className="flex justify-between text-[10px] text-gray-500 font-mono">
              <span>0.0</span>
              <span
                className="text-rose-400 font-semibold"
                style={{ marginLeft: `${Math.max(0, thresholdPercent - 15)}%` }}
              >
                | 1.0 (Threshold)
              </span>
              <span>{maxRange.toFixed(1)}x</span>
            </div>
          </div>

          {/* Prediction Label */}
          <div className="flex items-center justify-between text-xs bg-white/5 px-2.5 py-1.5 rounded border border-white/5">
            <span className="text-gray-400 font-medium">Predicted Label:</span>
            <span className="font-semibold text-gray-200">
              {model.prediction_label || 'Normal operation'}
            </span>
          </div>
        </div>
      )}

      {/* Footer: Last Update */}
      <div className="border-t border-white/5 pt-2 flex items-center justify-between text-[11px] text-gray-400">
        <span>Last update:</span>
        <span className="font-mono text-gray-300">{formattedTime}</span>
      </div>
    </Card>
  );
}

export default function DashboardPage() {
  const [scenarios, setScenarios] = useState<ScenarioMetadata[]>(FALLBACK_SCENARIOS);
  const [selectedScenarioId, setSelectedScenarioId] = useState<string>('multi_fault_escalation');
  const [loadingScenarios, setLoadingScenarios] = useState<boolean>(true);

  // Playback & WebSocket State
  const [playbackState, setPlaybackState] = useState<PlaybackStatus>('idle');
  const [speed, setSpeed] = useState<number>(1.0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Telemetry Step Data
  const [latestStep, setLatestStep] = useState<ReplayMessage | null>(null);
  const [trendHistory, setTrendHistory] = useState<MultimodalTrendPoint[]>([]);
  const [warningLogs, setWarningLogs] = useState<WarningLogEntry[]>([]);

  const wsRef = useRef<WebSocket | null>(null);
  const prevStatusRef = useRef<string | null>(null);

  // Load available scenarios
  useEffect(() => {
    let mounted = true;
    getReplayScenarios()
      .then((scs) => {
        if (mounted && scs && scs.length > 0) {
          setScenarios(scs);
          setSelectedScenarioId(scs[0].id);
        }
      })
      .catch(() => {
        // Fall back to predefined scenarios silently
      })
      .finally(() => {
        if (mounted) setLoadingScenarios(false);
      });

    return () => {
      mounted = false;
    };
  }, []);

  // Cleanup WebSocket on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        try {
          wsRef.current.close();
        } catch {
          // ignore
        }
      }
    };
  }, []);

  // Connect & stream over WebSocket
  const startReplay = useCallback(() => {
    if (!selectedScenarioId) return;

    if (wsRef.current) {
      try {
        wsRef.current.close();
      } catch {
        // ignore
      }
      wsRef.current = null;
    }

    setErrorMessage(null);
    setPlaybackState('connecting');
    setTrendHistory([]);
    setWarningLogs([]);
    prevStatusRef.current = null;

    const wsUrl = `${WS_BASE_URL}/ws/replay/${selectedScenarioId}`;
    let socket: WebSocket;

    try {
      socket = new WebSocket(wsUrl);
      wsRef.current = socket;
    } catch (err: any) {
      setErrorMessage(`WebSocket connection failed: ${err.message || 'Network error'}`);
      setPlaybackState('error');
      return;
    }

    socket.onopen = () => {
      setPlaybackState('playing');
      if (speed !== 1.0) {
        socket.send(JSON.stringify({ action: 'set_speed', speed }));
      }
    };

    socket.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);

        // Check for completion message
        if (msg.type === 'completed') {
          setPlaybackState('completed');
          return;
        }

        // Check for error
        if (msg.error) {
          setErrorMessage(msg.error);
          setPlaybackState('error');
          return;
        }

        // Process step data
        const stepData = msg as ReplayMessage;
        setLatestStep(stepData);

        // Append to trend history
        const timestampStr = stepData.timestamp
          ? new Date(stepData.timestamp).toLocaleTimeString([], { hour12: false })
          : `${stepData.step}s`;

        const trendPoint: MultimodalTrendPoint = {
          step: stepData.step,
          timeLabel: `${stepData.step}s`,
          timestamp: timestampStr,
          vibrationRatio: stepData.models?.vibration?.score_ratio ?? null,
          audioRatio: stepData.models?.audio?.score_ratio ?? null,
          motorRatio: stepData.models?.motor?.score_ratio ?? null,
          fusedScore: stepData.fusion?.fused_score ?? null,
        } as MultimodalTrendPoint;

        setTrendHistory((prev) => {
          const updated = [...prev, trendPoint];
          // Keep up to 100 points
          return updated.slice(-100);
        });

        // Warnings / Status transitions check
        const currentStatus = stepData.fusion?.fused_status;
        const prevStatus = prevStatusRef.current;
        const nowFormatted = new Date().toLocaleTimeString([], { hour12: false });

        if (prevStatus && currentStatus && prevStatus !== currentStatus) {
          const severity =
            currentStatus === 'ALARM'
              ? 'alarm'
              : currentStatus === 'WARNING'
              ? 'warning'
              : 'normal';

          setWarningLogs((prev) => [
            {
              id: `${stepData.step}-status-${Date.now()}`,
              timestamp: nowFormatted,
              type: 'status_change',
              title: `Status Transition: ${prevStatus} ➔ ${currentStatus}`,
              description: `Multimodal fusion state shifted from ${prevStatus} to ${currentStatus} at step ${stepData.step}.`,
              severity,
            },
            ...prev.slice(0, 49),
          ]);
        }
        prevStatusRef.current = currentStatus;

        // Check for active fusion reasons
        if (stepData.fusion?.reasons && stepData.fusion.reasons.length > 0) {
          const reasonText = stepData.fusion.reasons.join(' • ');
          setWarningLogs((prev) => {
            // Avoid duplicate reason logs for identical consecutive steps
            if (prev.length > 0 && prev[0].description === reasonText) {
              return prev;
            }
            return [
              {
                id: `${stepData.step}-reason-${Date.now()}`,
                timestamp: nowFormatted,
                type: 'fusion_reason',
                title: `Fusion Anomaly Trigger (Step ${stepData.step})`,
                description: reasonText,
                severity: stepData.fusion.fused_status === 'ALARM' ? 'alarm' : 'warning',
              },
              ...prev.slice(0, 49),
            ];
          });
        }
      } catch (err: any) {
        console.error('Failed to parse replay message:', err);
      }
    };

    socket.onerror = () => {
      setErrorMessage('WebSocket connection encountered an error.');
      setPlaybackState('error');
    };

    socket.onclose = () => {
      setPlaybackState((prev) => (prev === 'playing' || prev === 'paused' ? 'stopped' : prev));
    };
  }, [selectedScenarioId, speed]);

  const togglePause = () => {
    if (!wsRef.current) return;
    if (playbackState === 'playing') {
      try {
        wsRef.current.send(JSON.stringify({ action: 'pause' }));
        setPlaybackState('paused');
      } catch {
        // ignore
      }
    } else if (playbackState === 'paused') {
      try {
        wsRef.current.send(JSON.stringify({ action: 'resume' }));
        setPlaybackState('playing');
      } catch {
        // ignore
      }
    }
  };

  const stopReplay = () => {
    if (wsRef.current) {
      try {
        wsRef.current.send(JSON.stringify({ action: 'stop' }));
        wsRef.current.close();
      } catch {
        // ignore
      }
      wsRef.current = null;
    }
    setPlaybackState('stopped');
  };

  const handleSpeedChange = (newSpeed: number) => {
    setSpeed(newSpeed);
    if (wsRef.current && (playbackState === 'playing' || playbackState === 'paused')) {
      try {
        wsRef.current.send(JSON.stringify({ action: 'set_speed', speed: newSpeed }));
      } catch {
        // ignore
      }
    }
  };

  // Persistent banner is driven strictly by is_replay
  const isReplay = latestStep?.is_replay ?? (playbackState !== 'idle');

  // Header Fusion Verdict calculations
  const fusion: FusionResult | undefined = latestStep?.fusion;
  const fusedStatus = fusion?.fused_status ?? 'NORMAL';
  const fusedScore = fusion?.fused_score ?? 0;
  const fusionMethod = fusion?.method ?? 'max';

  // Available models count
  const models = latestStep?.models;
  const availableCount = [
    models?.vibration,
    models?.audio,
    models?.motor,
  ].filter((m) => m && m.status !== 'UNAVAILABLE').length;

  const currentScenario = scenarios.find((s) => s.id === selectedScenarioId);

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-12">
      {/* 1. Persistent Banner: driven by is_replay, never hideable */}
      {isReplay && (
        <div
          role="alert"
          className="w-full bg-gradient-to-r from-amber-500/20 via-amber-500/30 to-amber-500/20 border-2 border-amber-500/50 text-amber-200 px-4 py-3 rounded-xl shadow-lg flex flex-col sm:flex-row sm:items-center justify-between gap-3 sticky top-0 z-30 backdrop-blur-md"
        >
          <div className="flex items-center gap-3">
            <span className="text-xl animate-pulse">⚠️</span>
            <div>
              <span className="font-bold text-sm sm:text-base tracking-wide text-amber-100">
                REPLAYED DATA: recorded samples, not a live sensor
              </span>
              <p className="text-xs text-amber-300/80">
                Benchmark telemetry from CWRU (Vibration), MIMII DG (Acoustic), and Paderborn (Motor).
              </p>
            </div>
          </div>
          <span className="px-3 py-1 rounded bg-amber-500/30 border border-amber-400/40 text-xs font-mono font-bold uppercase tracking-wider text-amber-100 self-start sm:self-center shrink-0">
            Replay Active
          </span>
        </div>
      )}

      {/* 2. Top Header & Fused Verdict Status */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-white/5 pb-5">
        <div>
          <div className="flex items-center gap-3">
            <h2 className="text-2xl font-bold text-gray-100 tracking-tight">
              Multimodal Diagnostics Dashboard
            </h2>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-brand-500/15 text-brand-300 border border-brand-500/20">
              Live & Replay
            </span>
          </div>
          <p className="text-sm text-gray-400 mt-1">
            Real-time multi-sensor fusion: vibration, acoustic sound, and stator motor current.
          </p>
        </div>

        {/* Fused Status Pill Badge */}
        <div className="flex items-center gap-3">
          <div className="text-right">
            <p className="text-[10px] text-gray-400 uppercase tracking-wider font-semibold">
              Fused Health State
            </p>
            <div className="mt-0.5">
              {fusedStatus === 'ALARM' ? (
                <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-rose-500/20 text-rose-400 border border-rose-500/40 animate-pulse">
                  <span className="w-2 h-2 rounded-full bg-rose-500" />
                  ALARM
                </span>
              ) : fusedStatus === 'WARNING' ? (
                <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-amber-500/20 text-amber-400 border border-amber-500/40">
                  <span className="w-2 h-2 rounded-full bg-amber-500" />
                  WARNING
                </span>
              ) : (
                <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-emerald-500/20 text-emerald-400 border border-emerald-500/40">
                  <span className="w-2 h-2 rounded-full bg-emerald-500" />
                  NORMAL
                </span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Header Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          label="Fused Status"
          value={fusedStatus}
          sub="Multimodal Verdict"
          accent={
            fusedStatus === 'ALARM'
              ? 'anomalous'
              : fusedStatus === 'WARNING'
              ? 'warning'
              : 'healthy'
          }
          icon={fusedStatus === 'ALARM' ? '🚨' : fusedStatus === 'WARNING' ? '⚠️' : '✅'}
        />

        <MetricCard
          label="Fused Anomaly Score"
          value={latestStep ? fusedScore.toFixed(3) : '--'}
          sub={`Decision Method: ${fusionMethod.toUpperCase()}`}
          accent={fusedScore >= 1.0 ? 'anomalous' : 'brand'}
          icon="📈"
        />

        <MetricCard
          label="Fusion Method"
          value={fusionMethod.toUpperCase()}
          sub="Late Fusion Engine"
          accent="brand"
          icon="⚙️"
        />

        <MetricCard
          label="Models Available"
          value={latestStep ? `${availableCount} / 3 Active` : '3 / 3 Ready'}
          sub="Vibration, Audio, Motor"
          accent="healthy"
          icon="📡"
        />
      </div>

      {/* Replay Controller Panel */}
      <Card title="Telemetry Stream & Scenario Replay Controller">
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Scenario Selection */}
            <div className="md:col-span-2 space-y-1.5">
              <label htmlFor="scenario-select" className="text-xs font-medium text-gray-300">
                Select 60-Second Replay Scenario:
              </label>
              {loadingScenarios ? (
                <div className="flex items-center gap-2 text-xs text-gray-400 py-2">
                  <Spinner size="sm" /> Loading scenarios catalogue...
                </div>
              ) : (
                <select
                  id="scenario-select"
                  value={selectedScenarioId}
                  onChange={(e) => setSelectedScenarioId(e.target.value)}
                  disabled={playbackState === 'playing' || playbackState === 'paused'}
                  className="w-full bg-surface-2 dark:bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-gray-200 focus:outline-none focus:ring-2 focus:ring-brand-500 disabled:opacity-50"
                >
                  {scenarios.map((sc) => (
                    <option key={sc.id} value={sc.id} className="bg-surface-1 text-gray-200">
                      {sc.name} ({sc.duration_seconds}s) — {sc.description}
                    </option>
                  ))}
                </select>
              )}
              {currentScenario && (
                <p className="text-xs text-gray-400 leading-relaxed">
                  {currentScenario.description}
                </p>
              )}
            </div>

            {/* Speed Control */}
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-gray-300">Playback Speed:</label>
              <div className="flex items-center gap-1.5">
                {[0.5, 1.0, 2.0, 5.0].map((s) => (
                  <button
                    key={s}
                    onClick={() => handleSpeedChange(s)}
                    className={`flex-1 py-1.5 px-2 text-xs font-semibold rounded-md border transition-all ${
                      speed === s
                        ? 'bg-brand-500 text-white border-brand-400 shadow-sm'
                        : 'bg-white/5 text-gray-300 border-white/10 hover:bg-white/10'
                    }`}
                  >
                    {s}x
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Action Buttons & Progress Bar */}
          <div className="pt-2 border-t border-white/5 flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-2.5">
              {playbackState !== 'playing' && playbackState !== 'paused' ? (
                <button
                  onClick={startReplay}
                  disabled={playbackState === 'connecting'}
                  className="inline-flex items-center gap-2 px-5 py-2 rounded-lg bg-brand-500 hover:bg-brand-600 text-white text-sm font-semibold shadow-md transition-colors disabled:opacity-50"
                >
                  {playbackState === 'connecting' ? (
                    <>
                      <Spinner size="sm" /> Connecting...
                    </>
                  ) : (
                    <>
                      <span>▶</span> Start Replay
                    </>
                  )}
                </button>
              ) : (
                <>
                  <button
                    onClick={togglePause}
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-amber-500/20 hover:bg-amber-500/30 text-amber-300 border border-amber-500/40 text-sm font-semibold transition-colors"
                  >
                    {playbackState === 'paused' ? '▶ Resume' : '⏸ Pause'}
                  </button>

                  <button
                    onClick={stopReplay}
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-rose-500/20 hover:bg-rose-500/30 text-rose-300 border border-rose-500/40 text-sm font-semibold transition-colors"
                  >
                    ⏹ Stop
                  </button>
                </>
              )}

              {/* WebSocket Reconnect if error or stopped */}
              {(playbackState === 'error' || playbackState === 'stopped') && (
                <button
                  onClick={startReplay}
                  className="inline-flex items-center gap-1.5 px-3 py-2 rounded-lg bg-white/5 hover:bg-white/10 text-gray-300 border border-white/10 text-xs font-medium transition-colors"
                >
                  🔄 Reconnect
                </button>
              )}
            </div>

            {/* Step Counter & Progress */}
            <div className="flex items-center gap-3">
              <span className="text-xs text-gray-400 font-mono">
                {latestStep
                  ? `Step ${latestStep.step} / ${currentScenario?.total_steps ?? 60} (${Math.round(
                      (latestStep.step / (currentScenario?.total_steps ?? 60)) * 100
                    )}%)`
                  : 'Awaiting telemetry...'}
              </span>
              <div className="w-32 h-2 bg-white/10 rounded-full overflow-hidden">
                <div
                  className="h-full bg-brand-500 rounded-full transition-all duration-300"
                  style={{
                    width: latestStep
                      ? `${Math.min(
                          100,
                          (latestStep.step / (currentScenario?.total_steps ?? 60)) * 100
                        )}%`
                      : '0%',
                  }}
                />
              </div>
            </div>
          </div>
        </div>
      </Card>

      {/* Error Banner */}
      {errorMessage && (
        <ErrorBanner
          message={errorMessage}
          onDismiss={() => setErrorMessage(null)}
        />
      )}

      {/* 3. Three Model Cards: Vibration, Audio, Motor */}
      <div className="space-y-2">
        <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
          Individual Sensor Modality Models
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          <ModelCard
            title="Vibration Anomaly Detector"
            modality="vibration"
            icon="📊"
            model={models?.vibration}
          />
          <ModelCard
            title="Acoustic Sound Monitor"
            modality="audio"
            icon="🔊"
            model={models?.audio}
          />
          <ModelCard
            title="Motor Current & Temp Model"
            modality="motor"
            icon="⚡"
            model={models?.motor}
          />
        </div>
      </div>

      {/* 4. Multimodal Trend Chart (Recharts) */}
      <Card title="Multimodal Anomaly Score Trend (Score Ratio vs Threshold 1.0)">
        {trendHistory.length === 0 ? (
          <div className="h-64 border border-dashed border-white/10 rounded-lg flex flex-col items-center justify-center p-6 text-center">
            <span className="text-3xl mb-2 text-gray-500">📈</span>
            <p className="text-sm font-medium text-gray-300">
              No Telemetry Stream Active
            </p>
            <p className="text-xs text-gray-500 max-w-md mt-1">
              Select a scenario above and click &quot;Start Replay&quot; to begin streaming multimodal telemetry.
              The chart displays score ratios for each modality alongside the fused score with a red reference line at 1.0.
            </p>
          </div>
        ) : (
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs text-gray-400">
              <span>Dynamic score ratios (&gt;1.0 indicates anomaly threshold exceedance)</span>
              <span className="font-mono text-[11px] text-gray-500">
                Data points: {trendHistory.length}
              </span>
            </div>
            <MultimodalTrendChart data={trendHistory} />
          </div>
        )}
      </Card>

      {/* 5. Warnings Panel: status changes and fusion reasons */}
      <Card title="Warnings & Fusion Diagnostics Log">
        <div className="space-y-4">
          {/* Active Reasons Callout */}
          {fusion?.reasons && fusion.reasons.length > 0 && (
            <div className="bg-amber-500/10 border border-amber-500/25 rounded-lg p-3">
              <div className="flex items-center gap-2 text-xs font-semibold text-amber-300 mb-1">
                <span>⚠️</span> Active Fusion Decision Reasons:
              </div>
              <ul className="list-disc list-inside text-xs text-gray-200 space-y-0.5">
                {fusion.reasons.map((reason, idx) => (
                  <li key={idx} className="leading-relaxed">
                    {reason}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Historical Warning & Status Transition Log */}
          {warningLogs.length === 0 ? (
            <div className="text-center py-6 text-gray-500 text-xs">
              No status changes or warning events recorded. System operating within normal thresholds.
            </div>
          ) : (
            <div className="max-h-60 overflow-y-auto space-y-2 pr-1">
              {warningLogs.map((log) => (
                <div
                  key={log.id}
                  className={`p-2.5 rounded-lg border text-xs flex items-start justify-between gap-3 ${
                    log.severity === 'alarm'
                      ? 'bg-rose-500/10 border-rose-500/20 text-rose-300'
                      : log.severity === 'warning'
                      ? 'bg-amber-500/10 border-amber-500/20 text-amber-300'
                      : 'bg-emerald-500/10 border-emerald-500/20 text-emerald-300'
                  }`}
                >
                  <div className="space-y-0.5">
                    <p className="font-semibold text-gray-100">{log.title}</p>
                    <p className="text-gray-300 text-[11px] leading-relaxed">
                      {log.description}
                    </p>
                  </div>
                  <span className="text-[10px] font-mono text-gray-400 shrink-0">
                    {log.timestamp}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      </Card>
    </div>
  );
}
