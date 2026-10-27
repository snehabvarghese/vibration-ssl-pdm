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

export interface HistoryEntry {
  id: string;
  timestamp: string;
  source: 'upload' | 'recording';
  source_name: string;
  predicted_class: string;
  confidence: number;
  anomalous: boolean;
  modality?: 'vibration' | 'audio' | 'motor' | 'fusion';
  is_replay?: boolean;
  result?: AnalysisResult;
  model_result?: ModelResult;
  fusion_result?: FusionResult;
}

export interface SampleRecord {
  sample_id: string;
  modality: 'vibration' | 'audio' | 'motor';
  label?: string | null;
  description?: string | null;
  filename?: string | null;
  file_size_bytes?: number | null;
  duration_seconds?: number | null;
  sampling_rate?: number | null;
  is_replay: boolean;
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

// --------------------------------------------------------------------------
// Multimodal Model Adapters, Fusion & Replay Types
// --------------------------------------------------------------------------

export interface ModelResult {
  model_name: string;
  modality: 'vibration' | 'audio' | 'motor';
  status: 'NORMAL' | 'ANOMALOUS' | 'UNAVAILABLE';
  anomaly_score: number;
  threshold: number;
  score_ratio: number;
  prediction_label?: string | null;
  confidence?: number | null;
  details: Record<string, any>;
  input_info: Record<string, any>;
  is_replay: boolean;
  timestamp: string;
  error?: string | null;
}

export interface FusionResult {
  fused_score: number;
  fused_status: 'NORMAL' | 'WARNING' | 'ALARM' | 'UNAVAILABLE';
  method: 'max' | 'weighted' | 'vote';
  per_model_ratios: Record<string, number>;
  missing_models: string[];
  reasons: string[];
  confidence: number;
  is_replay: boolean;
  timestamp: string;
  details: Record<string, any>;
}

export interface ReplayMessage {
  step: number;
  timestamp: string;
  scenario_id: string;
  source: string;
  is_replay: boolean;
  models: {
    vibration?: ModelResult;
    audio?: ModelResult;
    motor?: ModelResult;
    [key: string]: ModelResult | undefined;
  };
  fusion: FusionResult;
}

export interface ScenarioMetadata {
  id: string;
  name: string;
  description: string;
  total_steps: number;
  duration_seconds: number;
  is_replay: boolean;
}

export interface ModelInfo {
  model_name: string;
  modality: string;
  is_available: boolean;
  threshold: number;
  threshold_metric?: string | null;
  training_dataset?: string | null;
  checkpoint_path: string;
  monitor_path?: string | null;
  machine_type?: string | null;
  sub_type?: string | null;
}

export interface FusionRequest {
  results?: Partial<ModelResult>[];
  sample_ids?: Record<string, string>;
  method?: 'max' | 'weighted' | 'vote';
  config?: Record<string, any>;
}


