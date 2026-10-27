/**
 * History page — shows a client-side list of this session's analyses.
 *
 * NOTE: History is stored in localStorage for now. This will move to a
 * backend database in a future iteration so that history persists across
 * devices and sessions.
 */

import { useEffect, useState } from 'react';
import { ResultCharts } from '../components/Charts';
import { Card, StatusBadge } from '../components/ui';
import type { EmbeddingMapResponse, HistoryEntry } from '../types/api';
import { getEmbeddingMap } from '../api/client';

export default function HistoryPage() {
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [selected, setSelected] = useState<HistoryEntry | null>(null);
  const [embeddingMap, setEmbeddingMap] = useState<EmbeddingMapResponse | null>(null);

  useEffect(() => {
    try {
      const raw = localStorage.getItem('analysis_history');
      if (raw) setHistory(JSON.parse(raw));
    } catch {
      // ignore parse errors
    }
    getEmbeddingMap().then(setEmbeddingMap).catch(() => {});
  }, []);

  const clearHistory = () => {
    localStorage.removeItem('analysis_history');
    setHistory([]);
    setSelected(null);
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white tracking-tight">Analysis History</h2>
          <p className="text-sm text-gray-500 mt-1">
            Previous analyses from this browser session.
          </p>
        </div>
        {history.length > 0 && (
          <button
            onClick={clearHistory}
            className="text-xs text-gray-500 hover:text-anomalous border border-white/10 rounded-lg px-3 py-1.5 transition-colors"
          >
            Clear all
          </button>
        )}
      </div>

      {history.length === 0 ? (
        <Card>
          <div className="text-center py-12">
            <p className="text-4xl mb-3">📋</p>
            <p className="text-gray-400">No analyses yet</p>
            <p className="text-sm text-gray-600 mt-1">
              Results will appear here after you run an analysis.
            </p>
          </div>
        </Card>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
          {/* List */}
          <div className="lg:col-span-1 space-y-2 max-h-[70vh] overflow-y-auto pr-1">
            {history.map((entry) => (
              <button
                key={entry.id}
                onClick={() => setSelected(entry)}
                className={`w-full text-left glass-card p-4 transition-all duration-200 hover:bg-surface-2 ${
                  selected?.id === entry.id ? 'glow-ring' : ''
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm font-medium text-gray-200 truncate max-w-[60%]">
                    {entry.source_name}
                  </span>
                  <div className="flex items-center gap-1.5">
                    {entry.is_replay && (
                      <span className="px-1.5 py-0.5 rounded text-[10px] font-mono font-bold bg-amber-500/20 text-amber-300 border border-amber-500/30">
                        REPLAY
                      </span>
                    )}
                    <StatusBadge anomalous={entry.anomalous} />
                  </div>
                </div>
                <div className="flex items-center justify-between text-xs text-gray-500">
                  <span>
                    {entry.modality?.toUpperCase() ?? 'ANALYSIS'} • {entry.predicted_class}
                  </span>
                  <span>{(entry.confidence * 100).toFixed(0)}%</span>
                </div>
                <p className="text-xs text-gray-600 mt-1">
                  {new Date(entry.timestamp).toLocaleString()}
                </p>
              </button>
            ))}
          </div>

          {/* Detail panel */}
          <div className="lg:col-span-2">
            {selected ? (
              <HistoryDetail entry={selected} embeddingMap={embeddingMap} />
            ) : (
              <Card>
                <div className="text-center py-16">
                  <p className="text-gray-500 text-sm">
                    ← Select an analysis to view details
                  </p>
                </div>
              </Card>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function HistoryDetail({
  entry,
  embeddingMap,
}: {
  entry: HistoryEntry;
  embeddingMap: EmbeddingMapResponse | null;
}) {
  const r = entry.result;
  const mr = entry.model_result;
  const fr = entry.fusion_result;

  return (
    <div className="space-y-4 page-enter">
      <div className="flex items-center gap-3 flex-wrap">
        <StatusBadge anomalous={entry.anomalous} />
        {entry.modality && (
          <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-brand-500/15 text-brand-300 border border-brand-500/30 uppercase">
            {entry.modality}
          </span>
        )}
        {entry.is_replay && (
          <span className="px-2 py-0.5 rounded text-xs font-mono font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40 uppercase">
            Recorded Replay Sample
          </span>
        )}
        <span className="text-lg font-bold text-white">{entry.predicted_class}</span>
        <span className="text-sm text-gray-500">
          {(entry.confidence * 100).toFixed(1)}% confidence
        </span>
      </div>

      {/* ModelResult details if present */}
      {mr && (
        <Card title="Diagnostic Metrics">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs mb-3">
            <div>
              <span className="text-gray-500 block">Anomaly Score</span>
              <span className="font-mono font-bold text-gray-200">{mr.anomaly_score.toFixed(3)}</span>
            </div>
            <div>
              <span className="text-gray-500 block">Threshold</span>
              <span className="font-mono font-bold text-gray-200">{mr.threshold.toFixed(3)}</span>
            </div>
            <div>
              <span className="text-gray-500 block">Score Ratio</span>
              <span className={`font-mono font-bold ${mr.score_ratio >= 1.0 ? 'text-rose-400' : 'text-emerald-400'}`}>
                {mr.score_ratio.toFixed(2)}x
              </span>
            </div>
            <div>
              <span className="text-gray-500 block">Source</span>
              <span className="text-gray-300 font-mono truncate block">{entry.source_name}</span>
            </div>
          </div>
        </Card>
      )}

      {/* FusionResult details if present */}
      {fr && (
        <Card title="Multimodal Fusion Verdict">
          <div className="space-y-3 text-xs">
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
              <div>
                <span className="text-gray-500 block">Fused Status</span>
                <span className="font-bold text-gray-100">{fr.fused_status}</span>
              </div>
              <div>
                <span className="text-gray-500 block">Fused Score</span>
                <span className="font-mono font-bold text-gray-100">{fr.fused_score.toFixed(3)}</span>
              </div>
              <div>
                <span className="text-gray-500 block">Method</span>
                <span className="font-mono text-brand-300">{fr.method.toUpperCase()}</span>
              </div>
            </div>
            {fr.reasons && fr.reasons.length > 0 && (
              <div>
                <span className="text-gray-400 font-semibold block mb-1">Reasons:</span>
                <ul className="list-disc list-inside text-gray-300 space-y-0.5">
                  {fr.reasons.map((rs, i) => (
                    <li key={i}>{rs}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </Card>
      )}

      {/* Original vibration result charts if present */}
      {r && embeddingMap && (
        <ResultCharts
          result={r}
          bgPoints={embeddingMap.points}
          classes={embeddingMap.classes}
        />
      )}
    </div>
  );
}

