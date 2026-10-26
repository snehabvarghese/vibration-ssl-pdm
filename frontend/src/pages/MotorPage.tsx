import { useCallback, useEffect, useState } from 'react';
import type { ChangeEvent, DragEvent } from 'react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  analyseMotorFile,
  analyseMotorPreset,
  getMotorRecordings,
} from '../api/client';
import { Card, ErrorBanner, MetricCard, Spinner, StatusBadge } from '../components/ui';
import type { MotorAnalysisResult, MotorRecording } from '../types/api';

const GRID_COLOR = 'rgba(255, 255, 255, 0.07)';
const AXIS_COLOR = 'rgba(255, 255, 255, 0.35)';

export default function MotorPage() {
  const [sourceMode, setSourceMode] = useState<'preset' | 'upload'>('preset');
  const [modelMode, setModelMode] = useState<'combined' | 'current_only' | 'temperature_only'>('combined');
  const [temperatureOverride, setTemperatureOverride] = useState<string>('');
  const [samplingRate, setSamplingRate] = useState<number>(64_000);

  // Upload state
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [dragOver, setDragOver] = useState(false);

  // Presets catalogue state
  const [recordings, setRecordings] = useState<MotorRecording[]>([]);
  const [loadingRecordings, setLoadingRecordings] = useState(true);
  const [selectedFilename, setSelectedFilename] = useState<string>('');
  const [bearingFilter, setBearingFilter] = useState<'ALL' | 'K001' | 'KA01' | 'KI01'>('ALL');

  // Analysis result state
  const [result, setResult] = useState<MotorAnalysisResult | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load available motor recordings
  useEffect(() => {
    let ignore = false;
    getMotorRecordings()
      .then((data) => {
        if (!ignore) {
          setRecordings(data);
          if (data.length > 0) {
            setSelectedFilename((prev) => prev || data[0].filename);
          }
        }
      })
      .catch((err) => {
        if (!ignore) {
          console.error('Failed to fetch motor recordings:', err);
        }
      })
      .finally(() => {
        if (!ignore) {
          setLoadingRecordings(false);
        }
      });
    return () => {
      ignore = true;
    };
  }, []);


  const handleDrop = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragOver(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) {
      setUploadedFile(dropped);
    }
  }, []);

  const handleFileChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) {
      setUploadedFile(f);
    }
  }, []);

  // Quick preset test launcher (Healthy K001, Outer KA01, Inner KI01)
  const handleQuickTest = async (targetBearing: 'K001' | 'KA01' | 'KI01') => {
    const match = recordings.find((r) => r.bearing_id === targetBearing);
    if (!match) {
      setError(`No benchmark file found for bearing ${targetBearing} on disk.`);
      return;
    }
    setSourceMode('preset');
    setSelectedFilename(match.filename);
    await runAnalysis('preset', match.filename);
  };

  const runAnalysis = async (mode: 'preset' | 'upload', presetName?: string) => {
    setAnalyzing(true);
    setError(null);
    setResult(null);

    const parsedTemp =
      temperatureOverride.trim() !== '' ? parseFloat(temperatureOverride) : undefined;

    try {
      let res: MotorAnalysisResult;
      if (mode === 'upload') {
        if (!uploadedFile) {
          throw new Error('Please select a file to upload (.mat, .npy, or .csv).');
        }
        res = await analyseMotorFile(uploadedFile, parsedTemp, samplingRate, modelMode);
      } else {
        const fileToAnalyse = presetName ?? selectedFilename;
        if (!fileToAnalyse) {
          throw new Error('Please select a benchmark recording from the list.');
        }
        res = await analyseMotorPreset(fileToAnalyse, parsedTemp, modelMode);
      }
      setResult(res);
    } catch (err: unknown) {
      if (err && typeof err === 'object' && 'response' in err) {
        const axiosErr = err as { response?: { data?: { detail?: string } } };
        setError(axiosErr.response?.data?.detail ?? 'Motor analysis request failed.');
      } else if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('An unknown error occurred during motor analysis.');
      }
    } finally {
      setAnalyzing(false);
    }
  };

  const handleFilterChange = (filter: 'ALL' | 'K001' | 'KA01' | 'KI01') => {
    setBearingFilter(filter);
    const subset = recordings.filter((r) => filter === 'ALL' || r.bearing_id === filter);
    if (subset.length > 0 && !subset.some((r) => r.filename === selectedFilename)) {
      setSelectedFilename(subset[0].filename);
    }
  };

  // Filtered recordings for the selector
  const filteredRecordings = recordings.filter((r) => {
    if (bearingFilter === 'ALL') return true;
    return r.bearing_id === bearingFilter;
  });

  const selectedPresetMeta = recordings.find((r) => r.filename === selectedFilename);


  // Prepare waveform chart data
  const waveformData =
    result?.waveform_p1?.map((v1, idx) => {
      const dtMs = +((idx * (result.input_diagnostics.duration_seconds || 4.0) * 1000) / result.waveform_p1!.length).toFixed(1);
      return {
        timeMs: dtMs,
        phase1: +v1.toFixed(3),
        phase2: result.waveform_p2 && result.waveform_p2[idx] !== undefined ? +result.waveform_p2[idx].toFixed(3) : undefined,
      };
    }) ?? [];

  // Prepare spectrum chart data
  const spectrumData =
    result?.spectrum?.freqs?.map((f, idx) => ({
      freqHz: +f.toFixed(1),
      magnitude: +(result.spectrum!.magnitudes[idx] || 0).toFixed(4),
    })) ?? [];

  // Condition color helper
  const conditionColor = (cond: string) => {
    if (cond === 'Normal') return 'text-healthy border-healthy/30 bg-healthy/10';
    if (cond === 'Outer_Race') return 'text-amber-400 border-amber-400/30 bg-amber-400/10';
    return 'text-anomalous border-anomalous/30 bg-anomalous/10';
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-12">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-2xl">⚡</span>
            <h2 className="text-2xl font-bold text-white tracking-tight">
              Motor Current & Temperature PdM
            </h2>
          </div>
          <p className="text-sm text-gray-400 mt-1">
            Motor Current Signature Analysis (MCSA) & Thermal Baseline diagnostics on Paderborn University bearing benchmark.
          </p>
        </div>

        {/* Quick Test Shortcuts */}
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xs text-gray-500 font-medium">Quick Verify:</span>
          <button
            onClick={() => handleQuickTest('K001')}
            disabled={analyzing}
            className="px-2.5 py-1.5 rounded-lg text-xs font-semibold bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 hover:bg-emerald-500/25 transition-all flex items-center gap-1.5"
            title="Run analysis on K001 Healthy benchmark recording"
          >
            <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
            K001 Healthy
          </button>
          <button
            onClick={() => handleQuickTest('KA01')}
            disabled={analyzing}
            className="px-2.5 py-1.5 rounded-lg text-xs font-semibold bg-amber-500/15 text-amber-300 border border-amber-500/30 hover:bg-amber-500/25 transition-all flex items-center gap-1.5"
            title="Run analysis on KA01 Outer Race Fault recording"
          >
            <span className="w-2 h-2 rounded-full bg-amber-400"></span>
            KA01 Outer Race
          </button>
          <button
            onClick={() => handleQuickTest('KI01')}
            disabled={analyzing}
            className="px-2.5 py-1.5 rounded-lg text-xs font-semibold bg-rose-500/15 text-rose-300 border border-rose-500/30 hover:bg-rose-500/25 transition-all flex items-center gap-1.5"
            title="Run analysis on KI01 Inner Race Fault recording"
          >
            <span className="w-2 h-2 rounded-full bg-rose-400"></span>
            KI01 Inner Race
          </button>
        </div>
      </div>

      {/* Dataset & Benchmark Context Notice */}
      <div className="bg-brand-500/5 border border-brand-500/20 rounded-xl px-4 py-3 flex items-start gap-3">
        <span className="text-brand-400 text-lg shrink-0 mt-0.5">ℹ️</span>
        <div className="text-xs text-gray-300 space-y-1">
          <p>
            <strong className="text-brand-300">Phase 1 Benchmark Architecture:</strong> Evaluates dual-phase stator current (<code className="text-brand-200">phase_current_1, 2</code> sampled at 64 kHz) and bearing module temperature (<code className="text-brand-200">temp_2_bearing_module</code> regulated at 45–50 °C).
          </p>
          <p className="text-gray-400">
            MCSA extracted features (50 Hz fundamental, THD, sideband energy ratio, and Park's Vector Transform) provide high-resolution fault identification, while thermal tracking monitors thermal drift and steady-state operating limits.
          </p>
        </div>
      </div>

      {/* Main Configuration Card */}
      <div className="glass-card p-5 space-y-5">
        {/* Source Mode Tabs */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-white/5 pb-4">
          <div className="flex gap-1 bg-surface-0 rounded-lg p-1 w-full sm:w-auto">
            <button
              onClick={() => setSourceMode('preset')}
              className={`px-4 py-2 rounded-md text-xs font-medium transition-all ${
                sourceMode === 'preset'
                  ? 'bg-brand-500/25 text-brand-300 shadow-sm'
                  : 'text-gray-400 hover:text-gray-200'
              }`}
            >
              Select Benchmark Recording ({recordings.length})
            </button>
            <button
              onClick={() => setSourceMode('upload')}
              className={`px-4 py-2 rounded-md text-xs font-medium transition-all ${
                sourceMode === 'upload'
                  ? 'bg-brand-500/25 text-brand-300 shadow-sm'
                  : 'text-gray-400 hover:text-gray-200'
              }`}
            >
              Upload Current & Thermal Data
            </button>
          </div>

          {/* Model Mode Selection */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-400 font-medium">Model Mode:</span>
            <div className="flex gap-1 bg-surface-0 rounded-lg p-1">
              {(
                [
                  { id: 'combined', label: 'Combined (98.97%)' },
                  { id: 'current_only', label: 'Current Only (98.80%)' },
                  { id: 'temperature_only', label: 'Temp Only (52.61%)' },
                ] as const
              ).map((m) => (
                <button
                  key={m.id}
                  onClick={() => setModelMode(m.id)}
                  className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                    modelMode === m.id
                      ? 'bg-brand-500/25 text-brand-300 shadow-sm'
                      : 'text-gray-400 hover:text-gray-200'
                  }`}
                >
                  {m.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Source Mode Content */}
        {sourceMode === 'preset' ? (
          <div className="space-y-4">
            {/* Filter pills */}
            <div className="flex items-center gap-2">
              <span className="text-xs text-gray-400 font-medium">Filter Bearing:</span>
              <div className="flex gap-1">
                {(['ALL', 'K001', 'KA01', 'KI01'] as const).map((filter) => (
                  <button
                    key={filter}
                    onClick={() => handleFilterChange(filter)}
                    className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                      bearingFilter === filter
                        ? 'bg-brand-500/30 text-white border border-brand-500/40'
                        : 'text-gray-400 hover:text-gray-200 bg-surface-0'
                    }`}
                  >
                    {filter === 'ALL'
                      ? 'All Recordings'
                      : filter === 'K001'
                      ? 'Healthy (K001)'
                      : filter === 'KA01'
                      ? 'Outer Race (KA01)'
                      : 'Inner Race (KI01)'}
                  </button>
                ))}
              </div>
            </div>

            {/* Recording Select Dropdown */}
            {loadingRecordings ? (
              <div className="flex items-center gap-2 text-sm text-gray-400 py-4 justify-center">
                <Spinner size="sm" />
                <span>Loading Paderborn benchmark catalogue...</span>
              </div>
            ) : recordings.length === 0 ? (
              <div className="bg-surface-0 rounded-lg p-4 text-center text-sm text-gray-400">
                No local Paderborn recordings found in <code className="text-gray-200">data/paderborn</code>. You can upload custom signals via the Upload tab.
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="md:col-span-2 space-y-1.5">
                  <label className="text-xs text-gray-400 font-medium">Recording File</label>
                  <select
                    value={selectedFilename}
                    onChange={(e) => setSelectedFilename(e.target.value)}
                    className="w-full bg-surface-0 border border-white/10 rounded-lg px-3 py-2 text-sm text-gray-200 focus:outline-none focus:border-brand-400"
                  >
                    {filteredRecordings.map((r) => (
                      <option key={r.filename} value={r.filename}>
                        {r.filename} — {r.bearing_id} ({r.condition_label}, {r.speed_rpm} RPM, {r.load_torque_nm} Nm)
                      </option>
                    ))}
                  </select>
                </div>

                {/* Preset Metadata preview */}
                {selectedPresetMeta && (
                  <div className="bg-surface-0/60 rounded-lg p-3 text-xs space-y-1 border border-white/5">
                    <p className="text-gray-400">
                      Bearing ID:{' '}
                      <strong className="text-white">{selectedPresetMeta.bearing_id}</strong>
                    </p>
                    <p className="text-gray-400">
                      True Label:{' '}
                      <span className="text-brand-300 font-semibold">
                        {selectedPresetMeta.condition_label} ({selectedPresetMeta.fault_type})
                      </span>
                    </p>
                    <p className="text-gray-400">
                      Operating Point:{' '}
                      <span className="text-gray-200">
                        {selectedPresetMeta.speed_rpm} RPM / {selectedPresetMeta.load_torque_nm} Nm
                      </span>
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-4">
            {/* Drag & Drop Upload Zone */}
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setDragOver(true);
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              className={`border-2 border-dashed rounded-xl p-8 text-center transition-all cursor-pointer ${
                dragOver
                  ? 'border-brand-400 bg-brand-500/10'
                  : uploadedFile
                  ? 'border-emerald-500/40 bg-emerald-500/5'
                  : 'border-white/10 hover:border-white/20 bg-surface-0/40'
              }`}
            >
              <input
                type="file"
                id="motor-file-upload"
                className="hidden"
                accept=".mat,.npy,.csv"
                onChange={handleFileChange}
              />
              <label htmlFor="motor-file-upload" className="cursor-pointer space-y-2 block">
                <span className="text-3xl block">📥</span>
                {uploadedFile ? (
                  <div>
                    <p className="text-sm font-semibold text-emerald-300">
                      {uploadedFile.name}
                    </p>
                    <p className="text-xs text-gray-400 mt-1">
                      {(uploadedFile.size / (1024 * 1024)).toFixed(2)} MB • Ready for analysis
                    </p>
                  </div>
                ) : (
                  <div>
                    <p className="text-sm font-medium text-gray-200">
                      Drop motor current file here, or{' '}
                      <span className="text-brand-400 underline">browse</span>
                    </p>
                    <p className="text-xs text-gray-500 mt-1">
                      Supports Paderborn <code className="text-gray-400">.mat</code> structs, NumPy <code className="text-gray-400">.npy</code> arrays, or CSV tables
                    </p>
                  </div>
                )}
              </label>
            </div>

            {/* Upload settings: Sampling rate */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="text-xs text-gray-400 font-medium block mb-1">
                  Sampling Frequency (Hz)
                </label>
                <input
                  type="number"
                  value={samplingRate}
                  onChange={(e) => setSamplingRate(Number(e.target.value) || 64000)}
                  className="w-full bg-surface-0 border border-white/10 rounded-lg px-3 py-2 text-sm text-gray-200 focus:outline-none focus:border-brand-400"
                  placeholder="64000"
                />
                <p className="text-[11px] text-gray-500 mt-1">
                  Paderborn benchmark is sampled at 64,000 Hz.
                </p>
              </div>

              <div>
                <label className="text-xs text-gray-400 font-medium block mb-1">
                  Thermal Sensor Override (°C)
                </label>
                <input
                  type="number"
                  step="0.5"
                  value={temperatureOverride}
                  onChange={(e) => setTemperatureOverride(e.target.value)}
                  className="w-full bg-surface-0 border border-white/10 rounded-lg px-3 py-2 text-sm text-gray-200 focus:outline-none focus:border-brand-400"
                  placeholder="Auto-extracted (nominal ~45°C)"
                />
                <p className="text-[11px] text-gray-500 mt-1">
                  Leave empty to extract from file, or test high thermal sweeps (e.g. 85°C).
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Global Controls & Run Button */}
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-2 border-t border-white/5">
          {sourceMode === 'preset' && (
            <div className="w-full sm:w-auto">
              <label className="text-xs text-gray-400 font-medium inline-block mr-2">
                Simulate Temperature (°C):
              </label>
              <input
                type="number"
                step="0.5"
                value={temperatureOverride}
                onChange={(e) => setTemperatureOverride(e.target.value)}
                placeholder="45.0 (Default)"
                className="w-32 bg-surface-0 border border-white/10 rounded-lg px-2.5 py-1.5 text-xs text-gray-200 focus:outline-none focus:border-brand-400"
              />
            </div>
          )}

          <div className="w-full sm:w-auto ml-auto">
            <button
              onClick={() => runAnalysis(sourceMode)}
              disabled={analyzing || (sourceMode === 'upload' && !uploadedFile) || (sourceMode === 'preset' && !selectedFilename)}
              className="w-full sm:w-auto px-6 py-2.5 rounded-lg text-sm font-semibold text-white bg-brand-500 hover:bg-brand-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-md flex items-center justify-center gap-2"
            >
              {analyzing ? (
                <>
                  <Spinner size="sm" />
                  <span>Processing MCSA & Thermal Features...</span>
                </>
              ) : (
                <>
                  <span>⚡</span>
                  <span>Analyse Motor Data</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Error Banner */}
      {error && <ErrorBanner message={error} onDismiss={() => setError(null)} />}

      {/* Diagnostic Verdict & Results */}
      {result && (
        <div className="space-y-6 page-enter">
          {/* Verdict Banner */}
          <div className="glass-card p-6 border-l-4 border-brand-400 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div>
                <div className="flex items-center gap-3">
                  <span
                    className={`inline-block px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider border ${conditionColor(
                      result.predicted_condition,
                    )}`}
                  >
                    Condition: {result.predicted_condition.replace('_', ' ')}
                  </span>
                  <StatusBadge anomalous={result.status === 'ANOMALOUS'} />
                </div>
                <h3 className="text-xl font-bold text-white mt-2">
                  Classification Verdict: {result.predicted_condition.replace('_', ' ')}
                </h3>
                <p className="text-xs text-gray-400 mt-1">
                  Evaluated using <code className="text-brand-300">{result.model_name}</code> (v{result.model_version})
                </p>
              </div>

              {/* Match Indicator if preset with ground truth */}
              {result.metadata?.true_label && (
                <div
                  className={`px-4 py-2.5 rounded-xl border flex items-center gap-2 text-xs font-semibold ${
                    result.metadata.true_label.toLowerCase() ===
                    result.predicted_condition.toLowerCase()
                      ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300'
                      : 'bg-amber-500/10 border-amber-500/30 text-amber-300'
                  }`}
                >
                  <span className="text-base">
                    {result.metadata.true_label.toLowerCase() ===
                    result.predicted_condition.toLowerCase()
                      ? '✅'
                      : '⚠️'}
                  </span>
                  <div>
                    <p className="leading-none">
                      Ground Truth: <strong>{result.metadata.true_label}</strong>
                    </p>
                    <p className="text-[11px] opacity-80 mt-0.5">
                      {result.metadata.true_label.toLowerCase() ===
                      result.predicted_condition.toLowerCase()
                        ? '100% Accurate Classification Match'
                        : 'Condition Discrepancy Observed'}
                    </p>
                  </div>
                </div>
              )}
            </div>

            {/* KPI Summary Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 pt-2">
              <MetricCard
                label="Predicted Class"
                value={result.predicted_condition.replace('_', ' ')}
                sub={`Confidence: ${(result.confidence * 100).toFixed(1)}%`}
                icon="🎯"
                accent={result.status === 'ANOMALOUS' ? 'anomalous' : 'healthy'}
              />
              <MetricCard
                label="Classification Confidence"
                value={`${(result.confidence * 100).toFixed(1)}%`}
                sub={`Winning class probability`}
                icon="📊"
                accent={result.confidence >= 0.8 ? 'healthy' : 'warning'}
              />
              <MetricCard
                label="Anomaly Score"
                value={result.anomaly_score.toFixed(3)}
                sub="Decision Threshold: 0.500"
                icon="🚨"
                accent={result.anomaly_score >= 0.5 ? 'anomalous' : 'healthy'}
              />
              <MetricCard
                label="Thermal State"
                value={
                  result.features.temp_mean !== undefined
                    ? `${result.features.temp_mean.toFixed(1)} °C`
                    : `${(result.metadata?.temperature_celsius ?? 45.0).toFixed(1)} °C`
                }
                sub={
                  (result.features.temp_mean ?? 45.0) > 70.0
                    ? 'Elevated (>70°C)'
                    : 'Regulated Nominal'
                }
                icon="🌡️"
                accent={(result.features.temp_mean ?? 45.0) > 70.0 ? 'warning' : 'brand'}
              />
            </div>
          </div>

          {/* Probabilities Breakdown */}
          <Card title="Class Probability Distribution">
            <div className="space-y-3">
              {Object.entries(result.class_probabilities).map(([className, prob]) => {
                const percentage = +(prob * 100).toFixed(1);
                const isSelected = className === result.predicted_condition;
                return (
                  <div key={className} className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className={`font-medium ${isSelected ? 'text-white' : 'text-gray-400'}`}>
                        {className.replace('_', ' ')} {isSelected && ' (Predicted)'}
                      </span>
                      <span className="font-mono text-gray-300">{percentage}%</span>
                    </div>
                    <div className="w-full bg-surface-0 rounded-full h-2.5 overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-500 ${
                          className === 'Normal'
                            ? 'bg-healthy'
                            : className === 'Outer_Race'
                            ? 'bg-amber-400'
                            : 'bg-anomalous'
                        }`}
                        style={{ width: `${Math.max(percentage, 1)}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </Card>

          {/* Signal Charts: Stator Current Waveforms & MCSA Spectrum */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Waveform Chart */}
            <Card title="Stator Current Waveform (Time Domain)">
              <div className="h-64 w-full">
                {waveformData.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={waveformData} margin={{ top: 10, right: 10, left: -10, bottom: 5 }}>
                      <CartesianGrid stroke={GRID_COLOR} strokeDasharray="3 3" />
                      <XAxis
                        dataKey="timeMs"
                        tick={{ fill: AXIS_COLOR, fontSize: 10 }}
                        label={{
                          value: 'Time (ms)',
                          position: 'insideBottomRight',
                          offset: -5,
                          fill: AXIS_COLOR,
                          fontSize: 10,
                        }}
                      />
                      <YAxis
                        tick={{ fill: AXIS_COLOR, fontSize: 10 }}
                        label={{
                          value: 'Current (A)',
                          angle: -90,
                          position: 'insideLeft',
                          fill: AXIS_COLOR,
                          fontSize: 10,
                        }}
                      />
                      <Tooltip
                        contentStyle={{
                          background: '#1a1d2d',
                          border: '1px solid rgba(255,255,255,0.1)',
                          borderRadius: 8,
                          fontSize: 11,
                        }}
                      />
                      <Legend wrapperStyle={{ fontSize: 11 }} />
                      <Line
                        type="monotone"
                        dataKey="phase1"
                        name="Phase 1 (I_u)"
                        stroke="#6366f1"
                        strokeWidth={1.5}
                        dot={false}
                      />
                      {result.waveform_p2 && (
                        <Line
                          type="monotone"
                          dataKey="phase2"
                          name="Phase 2 (I_v)"
                          stroke="#ec4899"
                          strokeWidth={1.5}
                          dot={false}
                        />
                      )}
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="flex items-center justify-center h-full text-xs text-gray-500">
                    No waveform data available
                  </div>
                )}
              </div>
            </Card>

            {/* MCSA Frequency Spectrum */}
            <Card title="MCSA Power Spectrum (0 – 1000 Hz)">
              <div className="h-64 w-full">
                {spectrumData.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={spectrumData} margin={{ top: 10, right: 10, left: -10, bottom: 5 }}>
                      <CartesianGrid stroke={GRID_COLOR} strokeDasharray="3 3" />
                      <XAxis
                        dataKey="freqHz"
                        tick={{ fill: AXIS_COLOR, fontSize: 10 }}
                        label={{
                          value: 'Frequency (Hz)',
                          position: 'insideBottomRight',
                          offset: -5,
                          fill: AXIS_COLOR,
                          fontSize: 10,
                        }}
                      />
                      <YAxis
                        tick={{ fill: AXIS_COLOR, fontSize: 10 }}
                        label={{
                          value: 'Amplitude (A)',
                          angle: -90,
                          position: 'insideLeft',
                          fill: AXIS_COLOR,
                          fontSize: 10,
                        }}
                      />
                      <Tooltip
                        contentStyle={{
                          background: '#1a1d2d',
                          border: '1px solid rgba(255,255,255,0.1)',
                          borderRadius: 8,
                          fontSize: 11,
                        }}
                      />
                      <Line
                        type="monotone"
                        dataKey="magnitude"
                        name="MCSA Magnitude"
                        stroke="#22d3ee"
                        strokeWidth={1.5}
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="flex items-center justify-center h-full text-xs text-gray-500">
                    No spectrum data available
                  </div>
                )}
              </div>
            </Card>
          </div>

          {/* Extracted Multimodal Features Table */}
          <Card title="Engineered MCSA & Thermal Diagnostic Features">
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
              <div className="bg-surface-0/60 p-3 rounded-lg border border-white/5">
                <span className="text-[10px] text-gray-500 uppercase tracking-wider block font-semibold">
                  RMS Current (P1)
                </span>
                <span className="text-base font-bold text-white mt-1 block">
                  {result.features.rms_p1 !== undefined ? `${result.features.rms_p1.toFixed(3)} A` : 'N/A'}
                </span>
              </div>

              <div className="bg-surface-0/60 p-3 rounded-lg border border-white/5">
                <span className="text-[10px] text-gray-500 uppercase tracking-wider block font-semibold">
                  Crest Factor
                </span>
                <span className="text-base font-bold text-white mt-1 block">
                  {result.features.crest_factor_p1 !== undefined ? result.features.crest_factor_p1.toFixed(3) : 'N/A'}
                </span>
              </div>

              <div className="bg-surface-0/60 p-3 rounded-lg border border-white/5">
                <span className="text-[10px] text-gray-500 uppercase tracking-wider block font-semibold">
                  Kurtosis
                </span>
                <span className="text-base font-bold text-white mt-1 block">
                  {result.features.kurtosis_p1 !== undefined ? result.features.kurtosis_p1.toFixed(3) : 'N/A'}
                </span>
              </div>

              <div className="bg-surface-0/60 p-3 rounded-lg border border-white/5">
                <span className="text-[10px] text-gray-500 uppercase tracking-wider block font-semibold">
                  THD (%)
                </span>
                <span className="text-base font-bold text-brand-300 mt-1 block">
                  {result.features.thd_p1 !== undefined ? `${(result.features.thd_p1 * 100).toFixed(2)}%` : 'N/A'}
                </span>
              </div>

              <div className="bg-surface-0/60 p-3 rounded-lg border border-white/5">
                <span className="text-[10px] text-gray-500 uppercase tracking-wider block font-semibold">
                  Sideband Ratio
                </span>
                <span className="text-base font-bold text-brand-300 mt-1 block">
                  {result.features.sideband_energy_ratio !== undefined ? result.features.sideband_energy_ratio.toFixed(4) : 'N/A'}
                </span>
              </div>

              <div className="bg-surface-0/60 p-3 rounded-lg border border-white/5">
                <span className="text-[10px] text-gray-500 uppercase tracking-wider block font-semibold">
                  Park Modulus Mean
                </span>
                <span className="text-base font-bold text-white mt-1 block">
                  {result.features.park_modulus_mean !== undefined ? result.features.park_modulus_mean.toFixed(3) : 'N/A'}
                </span>
              </div>
            </div>

            {/* Diagnostics Metadata Footer */}
            <div className="mt-4 pt-3 border-t border-white/5 flex flex-wrap items-center justify-between text-xs text-gray-400 gap-2">
              <div>
                Duration: <strong className="text-gray-200">{result.input_diagnostics.duration_seconds}s</strong> ({result.input_diagnostics.samples.toLocaleString()} samples @ {result.input_diagnostics.sampling_rate_hz.toLocaleString()} Hz)
              </div>
              <div>
                Two-Phase Available: <strong className={result.input_diagnostics.two_phase_available ? 'text-emerald-400' : 'text-gray-400'}>{result.input_diagnostics.two_phase_available ? 'Yes (Phase 1 & 2)' : 'No (Phase 1 only)'}</strong>
              </div>
              <div>
                Filename: <strong className="text-gray-200">{result.metadata?.filename ?? 'Custom Upload'}</strong>
              </div>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}
