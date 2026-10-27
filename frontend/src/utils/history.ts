import type { HistoryEntry } from '../types/api';

/**
 * Persists an analysis record to browser localStorage history.
 */
export function saveAnalysisHistory(
  entry: Omit<HistoryEntry, 'id' | 'timestamp'>,
): HistoryEntry {
  const fullEntry: HistoryEntry = {
    ...entry,
    id: crypto.randomUUID(),
    timestamp: new Date().toISOString(),
  };

  try {
    const raw = localStorage.getItem('analysis_history');
    const history: HistoryEntry[] = raw ? JSON.parse(raw) : [];
    history.unshift(fullEntry);
    if (history.length > 50) history.length = 50;
    localStorage.setItem('analysis_history', JSON.stringify(history));
  } catch {
    // Storage quota fallback
  }

  return fullEntry;
}
