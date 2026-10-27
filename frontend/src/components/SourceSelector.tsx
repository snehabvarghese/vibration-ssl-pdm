import { useCallback, useEffect, useState } from 'react';
import type { DragEvent, ChangeEvent } from 'react';
import { getRecordings } from '../api/client';
import type { Recording } from '../types/api';
import { Spinner } from './ui';

export type SourceMode = 'upload' | 'recording';

export interface UploadSource {
  mode: 'upload';
  file: File;
  samplingRate: number;
  channel: string;
}

export interface RecordingSource {
  mode: 'recording';
  recording: Recording;
}

export type AnalysisSource = UploadSource | RecordingSource;

interface SourceSelectorProps {
  onSourceReady: (source: AnalysisSource) => void;
  disabled?: boolean;
}

const ACCEPTED_EXTS = ['.mat', '.npy', '.csv'];

export default function SourceSelector({ onSourceReady, disabled }: SourceSelectorProps) {
  const [mode, setMode] = useState<SourceMode>('upload');

  // Upload state
  const [file, setFile] = useState<File | null>(null);
  const [samplingRate, setSamplingRate] = useState(12000);
  const [channel, setChannel] = useState('DE');
  const [dragOver, setDragOver] = useState(false);

  // Recording state
  const [recordings, setRecordings] = useState<Recording[]>([]);
  const [loadingRec, setLoadingRec] = useState(false);
  const [selectedRecId, setSelectedRecId] = useState('');
  const [recError, setRecError] = useState<string | null>(null);

  const fetchRecordings = () => {
    setLoadingRec(true);
    setRecError(null);
    getRecordings()
      .then((data) => {
        setRecordings(data);
      })
      .catch((err) => {
        setRecError(err?.message || 'Failed to connect to backend server');
      })
      .finally(() => setLoadingRec(false));
  };

  // Load recordings on first switch to recording mode
  useEffect(() => {
    if (mode === 'recording' && recordings.length === 0 && !recError) {
      fetchRecordings();
    }
  }, [mode, recordings.length, recError]);

  // Auto-select first recording when loaded
  useEffect(() => {
    if (recordings.length > 0 && !selectedRecId) {
      setSelectedRecId(recordings[0].id);
    }
  }, [recordings, selectedRecId]);

  const handleDrop = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setDragOver(false);
      const dropped = e.dataTransfer.files[0];
      if (dropped) setFile(dropped);
    },
    [],
  );

  const handleFileChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) setFile(f);
  }, []);

  const handleAnalyse = () => {
    if (mode === 'upload' && file) {
      onSourceReady({ mode: 'upload', file, samplingRate, channel });
    } else if (mode === 'recording' && selectedRecId) {
      const rec = recordings.find((r) => r.id === selectedRecId);
      if (rec) onSourceReady({ mode: 'recording', recording: rec });
    }
  };

  const canAnalyse =
    !disabled &&
    ((mode === 'upload' && file !== null) || (mode === 'recording' && selectedRecId !== ''));

  return (
    <div className="glass-card p-5 space-y-4">
      {/* Mode tabs */}
      <div className="flex gap-1 bg-surface-0 rounded-lg p-1">
        {(['upload', 'recording'] as const).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={`flex-1 py-2 rounded-md text-sm font-medium transition-all duration-200 ${
              mode === m
                ? 'bg-brand-500/20 text-brand-300 shadow-sm'
                : 'text-gray-500 hover:text-gray-300'
            }`}
          >
            {m === 'upload' ? '📁 Upload file' : '💾 Dataset recording'}
          </button>
        ))}
      </div>

      {/* Upload tab content */}
      {mode === 'upload' && (
        <div className="space-y-3">
          {/* Drop zone */}
          <div
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            className={`relative border-2 border-dashed rounded-xl p-6 text-center transition-all duration-200 cursor-pointer ${
              dragOver
                ? 'border-brand-400 bg-brand-500/10'
                : file
                  ? 'border-healthy/30 bg-healthy/5'
                  : 'border-white/10 hover:border-white/20'
            }`}
          >
            <input
              type="file"
              accept={ACCEPTED_EXTS.join(',')}
              onChange={handleFileChange}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            />
            {file ? (
              <div>
                <p className="text-sm font-medium text-healthy">✓ {file.name}</p>
                <p className="text-xs text-gray-500 mt-1">
                  {(file.size / 1024).toFixed(1)} KB · Click or drop to replace
                </p>
              </div>
            ) : (
              <div>
                <p className="text-3xl mb-2">📂</p>
                <p className="text-sm text-gray-400">
                  Drop a <strong>.mat</strong>, <strong>.npy</strong>, or <strong>.csv</strong> file
                </p>
                <p className="text-xs text-gray-600 mt-1">or click to browse</p>
              </div>
            )}
          </div>

          {/* Sampling rate + channel */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-gray-500 mb-1 font-medium">Sampling Rate (Hz)</label>
              <input
                type="number"
                value={samplingRate}
                onChange={(e) => setSamplingRate(Number(e.target.value))}
                min={1}
                className="w-full bg-surface-0 border border-white/10 rounded-lg px-3 py-2 text-sm text-gray-200 focus:outline-none focus:border-brand-400 transition-colors"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-500 mb-1 font-medium">
                Channel <span className="text-gray-600">(for .mat)</span>
              </label>
              <input
                type="text"
                value={channel}
                onChange={(e) => setChannel(e.target.value)}
                placeholder="DE"
                className="w-full bg-surface-0 border border-white/10 rounded-lg px-3 py-2 text-sm text-gray-200 focus:outline-none focus:border-brand-400 transition-colors"
              />
            </div>
          </div>
        </div>
      )}

      {/* Recording tab content */}
      {mode === 'recording' && (
        <div>
          {loadingRec ? (
            <div className="flex items-center gap-2 py-4 justify-center text-gray-500 text-sm">
              <Spinner size="sm" /> Loading recordings…
            </div>
          ) : recError ? (
            <div className="text-center py-3">
              <p className="text-xs text-rose-400 mb-2">Failed to load recordings: {recError}</p>
              <button
                type="button"
                onClick={fetchRecordings}
                className="text-xs text-brand-400 hover:underline cursor-pointer"
              >
                Retry
              </button>
            </div>
          ) : recordings.length === 0 ? (
            <p className="text-sm text-gray-500 text-center py-4">
              No recordings found in <code>data/raw</code>. Make sure the API server is running.
            </p>
          ) : (
            <select
              value={selectedRecId}
              onChange={(e) => setSelectedRecId(e.target.value)}
              className="w-full bg-surface-0 border border-white/10 rounded-lg px-3 py-2.5 text-sm text-gray-200 focus:outline-none focus:border-brand-400 transition-colors"
            >
              {recordings.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.id}.mat — {r.label} (Load {r.load_hp} HP, {r.sampling_rate} Hz)
                </option>
              ))}
            </select>
          )}
        </div>
      )}

      {/* Analyse button */}
      <button
        onClick={handleAnalyse}
        disabled={!canAnalyse}
        className={`w-full py-3 rounded-xl text-sm font-semibold tracking-wide transition-all duration-200 ${
          canAnalyse
            ? 'bg-gradient-to-r from-brand-500 to-brand-600 text-white hover:from-brand-400 hover:to-brand-500 shadow-lg shadow-brand-500/20 active:scale-[0.98]'
            : 'bg-surface-2 text-gray-600 cursor-not-allowed'
        }`}
      >
        {disabled ? (
          <span className="flex items-center justify-center gap-2">
            <Spinner size="sm" /> Analysing…
          </span>
        ) : (
          '🔬 Analyse'
        )}
      </button>
    </div>
  );
}
