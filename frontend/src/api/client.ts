/**
 * Typed API client for the Vibration SSL PdM backend.
 *
 * Base URL is configured via VITE_API_URL env var (defaults to http://localhost:8000).
 * No URLs are hardcoded in component code — everything goes through this module.
 */

import axios, { type AxiosInstance } from 'axios';
import type {
  AnalysisResult,
  EmbeddingMapResponse,
  HealthResponse,
  MotorAnalysisResult,
  MotorRecording,
  Recording,
} from '../types/api';

const API_BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

const api: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120_000, // analysis can be slow on CPU
});

// --------------------------------------------------------------------------
// Health
// --------------------------------------------------------------------------
export async function getHealth(): Promise<HealthResponse> {
  const { data } = await api.get<HealthResponse>('/health');
  return data;
}

// --------------------------------------------------------------------------
// Recordings catalogue
// --------------------------------------------------------------------------
export async function getRecordings(): Promise<Recording[]> {
  const { data } = await api.get<Recording[]>('/recordings');
  return data;
}

// --------------------------------------------------------------------------
// Analysis
// --------------------------------------------------------------------------
export async function analyseFile(
  file: File,
  samplingRate: number,
  channel?: string,
  clfName: string = 'svm_rbf',
): Promise<AnalysisResult> {
  const form = new FormData();
  form.append('file', file);
  form.append('sampling_rate', String(samplingRate));
  if (channel) form.append('channel', channel);
  form.append('clf_name', clfName);

  const { data } = await api.post<AnalysisResult>('/analyse', form);
  return data;
}

export async function analyseRecording(
  fileId: string,
  clfName: string = 'svm_rbf',
): Promise<AnalysisResult> {
  const { data } = await api.post<AnalysisResult>(
    `/analyse/recording/${fileId}`,
    null,
    { params: { clf_name: clfName } },
  );
  return data;
}

// --------------------------------------------------------------------------
// Motor Current & Temperature PdM (Paderborn Benchmark)
// --------------------------------------------------------------------------
export async function getMotorRecordings(): Promise<MotorRecording[]> {
  const { data } = await api.get<MotorRecording[]>('/recordings/motor');
  return data;
}

export async function analyseMotorFile(
  file: File,
  temperature?: number,
  samplingRate: number = 64_000,
  modelMode: string = 'combined',
): Promise<MotorAnalysisResult> {
  const form = new FormData();
  form.append('file', file);
  if (temperature !== undefined && !isNaN(temperature)) {
    form.append('temperature', String(temperature));
  }
  form.append('sampling_rate', String(samplingRate));
  form.append('model_mode', modelMode);

  const { data } = await api.post<MotorAnalysisResult>('/analyse/motor', form);
  return data;
}

export async function analyseMotorPreset(
  filename: string,
  temperature?: number,
  modelMode: string = 'combined',
): Promise<MotorAnalysisResult> {
  const form = new FormData();
  form.append('filename', filename);
  if (temperature !== undefined && !isNaN(temperature)) {
    form.append('temperature', String(temperature));
  }
  form.append('model_mode', modelMode);

  const { data } = await api.post<MotorAnalysisResult>('/analyse/motor/recording', form);
  return data;
}

// --------------------------------------------------------------------------
// Embedding map
// --------------------------------------------------------------------------
export async function getEmbeddingMap(
  maxPoints: number = 1500,
): Promise<EmbeddingMapResponse> {
  const { data } = await api.get<EmbeddingMapResponse>('/map', {
    params: { max_points: maxPoints },
  });
  return data;
}

export default api;

