/**
 * TypeScript types matching the FastAPI backend JSON responses.
 */

// --------------------------------------------------------------------------
// GET /health
// --------------------------------------------------------------------------
export interface HealthResponse {
  status: string;
  device: string;
  model_loaded: boolean;
}

// --------------------------------------------------------------------------
// GET /recordings
// --------------------------------------------------------------------------
export interface Recording {
  id: string;
  label: string;
  fault_type: string;
  load_hp: number;
  rpm: number;
  sampling_rate: number;
}

// --------------------------------------------------------------------------
// POST /analyse  &  POST /analyse/recording/{file_id}
// --------------------------------------------------------------------------
export interface WaveformData {
  samples: number[];
  sampling_rate: number;
}

export interface AnalysisResult {
  n_segments: number;
  embeddings: number[][];
  predicted_class: string;
  confidence: number;
  class_probabilities: Record<string, number>;

  // Health monitor fields (present when monitor is loaded)
  edr_scores?: number[];
  edr_mean?: number;
  edr_threshold?: number;
  anomalous?: boolean;
  segment_scores?: number[];
  segment_threshold?: number;
  fraction_segments_flagged?: number;

  // Waveform & embedding map (added by the API layer)
  waveform: WaveformData;
  map_xy: number[][];

  // Present only for dataset recording analyses
  recording_id?: string;
  true_label?: string;
  load_hp?: number;
  rpm?: number;
}

// --------------------------------------------------------------------------
// GET /map
// --------------------------------------------------------------------------
export interface MapPoint {
  x: number;
  y: number;
  class_label: string;
}

export interface EmbeddingMapResponse {
  points: MapPoint[];
  classes: string[];
  total_points: number;
}

// --------------------------------------------------------------------------
// History (client-side, stored in localStorage)
// --------------------------------------------------------------------------
export interface HistoryEntry {
  id: string;
  timestamp: string;
  source: 'upload' | 'recording';
  source_name: string;
  predicted_class: string;
  confidence: number;
  anomalous: boolean;
  result: AnalysisResult;
}

// --------------------------------------------------------------------------
// Phase 2: Live monitoring types (Contract for Phase 2)
// --------------------------------------------------------------------------

/** Contract for Phase 2: streaming real-time sensor readings from sensors. */
export interface LiveReading {
  machineId: string;
  timestamp: string;
  samplingRate: number;
  samples: number[];
  status?: string;
  edr?: number;
}

