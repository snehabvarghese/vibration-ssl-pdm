import { useEffect, useState } from 'react';
import { analyseFile, analyseRecording, getEmbeddingMap } from '../api/client';
import { ResultCharts } from '../components/Charts';
import SourceSelector from '../components/SourceSelector';
import type { AnalysisSource } from '../components/SourceSelector';
import { Card, ErrorBanner, MetricCard, StatusBadge } from '../components/ui';
import type { AnalysisResult, EmbeddingMapResponse } from '../types/api';
import type { HistoryEntry } from '../types/api';

export default function AnalysePage() {
  const [embeddingMap, setEmbeddingMap] = useState<EmbeddingMapResponse | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [sourceName, setSourceName] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load background embedding map once
  useEffect(() => {
    getEmbeddingMap().then(setEmbeddingMap).catch(() => {});
  }, []);

  const handleAnalyse = async (source: AnalysisSource) => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      let res: AnalysisResult;
      let name: string;

      if (source.mode === 'upload') {
        res = await analyseFile(source.file, source.samplingRate, source.channel);
        name = source.file.name;
      } else {
        res = await analyseRecording(source.recording.id);
        name = `${source.recording.id}.mat (${source.recording.label})`;
      }

      setResult(res);
      setSourceName(name);

      // Save to history (localStorage) — will move to backend DB later
      saveToHistory(res, source.mode, name);
    } catch (err: unknown) {
      if (err && typeof err === 'object' && 'response' in err) {
        const axiosErr = err as { response?: { data?: { detail?: string } } };
        setError(axiosErr.response?.data?.detail ?? 'An unknown error occurred.');
      } else if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('An unknown error occurred.');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Page header */}
      <div>
        <h2 className="text-2xl font-bold text-white tracking-tight">Vibration Analysis</h2>
        <p className="text-sm text-gray-500 mt-1">
          Upload a vibration recording or select from the CWRU dataset to diagnose bearing health.
        </p>
      </div>

      {/* CWRU notice */}
      <div className="bg-brand-500/5 border border-brand-500/15 rounded-lg px-4 py-3 flex items-start gap-2.5">
        <span className="text-brand-400 text-sm shrink-0 mt-0.5">ℹ️</span>
        <p className="text-xs text-gray-400 leading-relaxed">
          <strong className="text-gray-300">Model trained on CWRU bearing data</strong> — results
          on other machines or sensor configurations are not validated. Use with appropriate
          engineering judgement.
        </p>
      </div>

      {/* Source selector */}
      <SourceSelector onSourceReady={handleAnalyse} disabled={loading} />

      {/* Error */}
      {error && <ErrorBanner message={error} onDismiss={() => setError(null)} />}

      {/* Results */}
      {result && (
        <div className="space-y-5 page-enter">
          {/* Summary cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card>
              <div className="flex items-center justify-between">
                <span className="text-xs text-gray-500 font-medium uppercase">Status</span>
                <StatusBadge anomalous={result.anomalous ?? false} />
              </div>
              {result.true_label && (
                <p className="text-xs text-gray-500 mt-2">
                  True label: <strong className="text-gray-300">{result.true_label}</strong>
                </p>
              )}
            </Card>

            <MetricCard
              label="Predicted Class"
              value={result.predicted_class}
              sub={`${(result.confidence * 100).toFixed(1)}% confidence`}
              icon="🎯"
              accent={result.anomalous ? 'anomalous' : 'healthy'}
            />

            <MetricCard
              label="EDR Score"
              value={result.edr_mean != null ? result.edr_mean.toFixed(3) : 'N/A'}
              sub={
                result.edr_threshold != null
                  ? `Threshold: ${result.edr_threshold.toFixed(3)}`
                  : undefined
              }
              icon="📈"
              accent={result.anomalous ? 'anomalous' : 'brand'}
            />

            <MetricCard
              label="Segments Flagged"
              value={
                result.fraction_segments_flagged != null
                  ? `${(result.fraction_segments_flagged * 100).toFixed(0)}%`
                  : 'N/A'
              }
              sub={`${result.n_segments} total segments`}
              icon="🔍"
              accent={
                (result.fraction_segments_flagged ?? 0) > 0.5 ? 'anomalous' : 'healthy'
              }
            />
          </div>

          {/* Source name */}
          <p className="text-xs text-gray-600">
            Source: <code className="text-gray-400">{sourceName}</code>
          </p>

          {/* Charts */}
          {embeddingMap && (
            <ResultCharts
              result={result}
              bgPoints={embeddingMap.points}
              classes={embeddingMap.classes}
            />
          )}
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  History helper — stores in localStorage                           */
/* ------------------------------------------------------------------ */
// NOTE: This will move to a backend DB later (Phase 2).

function saveToHistory(
  result: AnalysisResult,
  sourceMode: 'upload' | 'recording',
  sourceName: string,
) {
  const entry: HistoryEntry = {
    id: crypto.randomUUID(),
    timestamp: new Date().toISOString(),
    source: sourceMode,
    source_name: sourceName,
    predicted_class: result.predicted_class,
    confidence: result.confidence,
    anomalous: result.anomalous ?? false,
    result,
  };

  try {
    const raw = localStorage.getItem('analysis_history');
    const history: HistoryEntry[] = raw ? JSON.parse(raw) : [];
    history.unshift(entry);
    // Keep last 50 entries
    if (history.length > 50) history.length = 50;
    localStorage.setItem('analysis_history', JSON.stringify(history));
  } catch {
    // Silently fail if storage is full
  }
}
