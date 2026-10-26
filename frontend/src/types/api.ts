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

// --------------------------------------------------------------------------
// Phase 1: Motor Current & Temperature PdM Types (Paderborn Benchmark)
// --------------------------------------------------------------------------

export interface MotorRecording {
  filename: string;
  bearing_id: string;
  condition_label: string;
  fault_type: string;
  operating_condition: string;
  speed_rpm: number;
  load_torque_nm: number;
  file_size_mb: number;
}

export interface MotorSpectrum {
  freqs: number[];
  magnitudes: number[];
}

export interface MotorAnalysisResult {
  model_name: string;
  model_version: string;
  status: 'HEALTHY' | 'ANOMALOUS';
  predicted_condition: 'Normal' | 'Outer_Race' | 'Inner_Race' | string;
  anomaly_score: number;
  confidence: number;
  class_probabilities: Record<string, number>;
  timestamp?: string | null;
  features: {
    rms_p1?: number;
    rms_p2?: number;
    peak_p1?: number;
    peak_p2?: number;
    crest_factor_p1?: number;
    kurtosis_p1?: number;
    skewness_p1?: number;
    thd_p1?: number;
    sideband_energy_ratio?: number;
    spectral_centroid?: number;
    spectral_entropy?: number;
    park_modulus_mean?: number;
    park_modulus_std?: number;
    park_eccentricity?: number;
    temp_mean?: number;
    temp_std?: number;
    temp_drift?: number;
    [key: string]: number | undefined;
  };
  input_diagnostics: {
    samples: number;
    duration_seconds: number;
    sampling_rate_hz: number;
    two_phase_available: boolean;
    temperature_input_available: boolean;
    [key: string]: unknown;
  };
  waveform_p1?: number[];
  waveform_p2?: number[];
  spectrum?: MotorSpectrum;
  metadata?: {
    filename?: string;
    bearing_id?: string;
    true_label?: string;
    operating_condition?: string;
    speed_rpm?: number;
    load_torque_nm?: number;
    temperature_celsius?: number;
    sampling_rate?: number;
  };
}

