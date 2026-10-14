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
import type { AnalysisResult, EmbeddingMapResponse, HistoryEntry } from '../types/api';
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
                  <StatusBadge anomalous={entry.anomalous} />
                </div>
                <p className="text-xs text-gray-500">
                  {entry.predicted_class} · {(entry.confidence * 100).toFixed(0)}%
                </p>
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
  const r: AnalysisResult = entry.result;

  return (
    <div className="space-y-4 page-enter">
      <div className="flex items-center gap-3 flex-wrap">
        <StatusBadge anomalous={entry.anomalous} />
        <span className="text-lg font-bold text-white">{r.predicted_class}</span>
        <span className="text-sm text-gray-500">
          {(r.confidence * 100).toFixed(1)}% confidence
        </span>
        {r.true_label && (
          <span className="text-xs bg-healthy/10 text-healthy px-2 py-0.5 rounded-full border border-healthy/20">
            True: {r.true_label}
          </span>
        )}
      </div>

      {embeddingMap && (
        <ResultCharts
          result={r}
          bgPoints={embeddingMap.points}
          classes={embeddingMap.classes}
        />
      )}
    </div>
  );
}
