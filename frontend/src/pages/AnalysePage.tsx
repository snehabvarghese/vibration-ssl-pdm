import { useEffect, useState } from 'react';
import {
  analyseFile,
  analyseRecording,
  getEmbeddingMap,
  getModels,
  getSamples,
  postFusion,
  predictAudio,
  predictMotorSignal,
  predictSample,
  predictVibrationSignal,
} from '../api/client';
import { ResultCharts, SoundPlot, MotorSeriesPlot } from '../components/Charts';
import SourceSelector from '../components/SourceSelector';
import type { AnalysisSource } from '../components/SourceSelector';
import { Card, ErrorBanner, MetricCard, Spinner, StatusBadge } from '../components/ui';
import type {
  AnalysisResult,
  EmbeddingMapResponse,
  FusionResult,
  ModelInfo,
  ModelResult,
  SampleRecord,
} from '../types/api';
import { saveAnalysisHistory } from '../utils/history';

type TabType = 'vibration' | 'sound' | 'motor' | 'multimodal';

export default function AnalysePage() {
  const [activeTab, setActiveTab] = useState<TabType>('vibration');
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loadingModels, setLoadingModels] = useState(true);

  // Common UI State
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 1. Vibration State
  const [embeddingMap, setEmbeddingMap] = useState<EmbeddingMapResponse | null>(null);
  const [vibResult, setVibResult] = useState<AnalysisResult | null>(null);
  const [vibSourceName, setVibSourceName] = useState('');
  const [lastVibSource, setLastVibSource] = useState<AnalysisSource | null>(null);

  // 2. Sound State
  const [soundMode, setSoundMode] = useState<'upload' | 'sample'>('upload');
  const [soundFile, setSoundFile] = useState<File | null>(null);
  const [soundMachineType, setSoundMachineType] = useState<string>('fan');
  const [soundSamples, setSoundSamples] = useState<SampleRecord[]>([]);
  const [selectedSoundSampleId, setSelectedSoundSampleId] = useState<string>('');
  const [soundResult, setSoundResult] = useState<ModelResult | null>(null);

  // 3. Motor State
  const [motorMode, setMotorMode] = useState<'upload' | 'sample'>('upload');
  const [motorFile, setMotorFile] = useState<File | null>(null);
  const [motorTemp, setMotorTemp] = useState<string>('45.0');
  const [motorSamples, setMotorSamples] = useState<SampleRecord[]>([]);
  const [selectedMotorSampleId, setSelectedMotorSampleId] = useState<string>('');
  const [motorResult, setMotorResult] = useState<ModelResult | null>(null);

  // 4. Multimodal Fusion State
  const [fusionResult, setFusionResult] = useState<FusionResult | null>(null);
  const [fusionResultsMap, setFusionResultsMap] = useState<{
    vibration?: ModelResult | null;
    audio?: ModelResult | null;
    motor?: ModelResult | null;
  }>({});

  // Fetch registered models on mount for dynamic limitation notes
  useEffect(() => {
    getModels()
      .then((data) => setModels(data))
      .catch(() => {})
      .finally(() => setLoadingModels(false));

    getEmbeddingMap()
      .then(setEmbeddingMap)
      .catch(() => {});
  }, []);

  // Fetch sound samples when sound tab is selected
  useEffect(() => {
    if (activeTab === 'sound' || activeTab === 'multimodal') {
      if (soundSamples.length === 0) {
        getSamples('audio')
          .then((s) => {
            setSoundSamples(s);
            if (s.length > 0 && !selectedSoundSampleId) {
              setSelectedSoundSampleId(s[0].sample_id);
            }
          })
          .catch(() => {});
      }
    }
  }, [activeTab, soundSamples.length, selectedSoundSampleId]);

  // Fetch motor samples when motor tab is selected
  useEffect(() => {
    if (activeTab === 'motor' || activeTab === 'multimodal') {
      if (motorSamples.length === 0) {
        getSamples('motor')
          .then((s) => {
            setMotorSamples(s);
            if (s.length > 0 && !selectedMotorSampleId) {
              setSelectedMotorSampleId(s[0].sample_id);
            }
          })
          .catch(() => {});
      }
    }
  }, [activeTab, motorSamples.length, selectedMotorSampleId]);

  // Dynamic model limitation notes from GET /models
  const vibModelInfo = models.find((m) => m.modality === 'vibration');
  const soundModelInfo =
    models.find((m) => m.modality === 'audio' && m.machine_type === soundMachineType) ||
    models.find((m) => m.modality === 'audio');
  const motorModelInfo = models.find((m) => m.modality === 'motor');

  // -------------------------------------------------------------------------
  // Handlers
  // -------------------------------------------------------------------------

  // Vibration Analysis (unchanged workflow)
  const handleVibrationAnalyse = async (source: AnalysisSource) => {
    setLoading(true);
    setError(null);
    setVibResult(null);
    setLastVibSource(source);

    try {
      let res: AnalysisResult;
      let name: string;
      const isReplay = source.mode === 'recording';

      if (source.mode === 'upload') {
        res = await analyseFile(source.file, source.samplingRate, source.channel);
        name = source.file.name;
      } else {
        res = await analyseRecording(source.recording.id);
        name = `${source.recording.id}.mat (${source.recording.label})`;
      }

      setVibResult(res);
      setVibSourceName(name);

      saveAnalysisHistory({
        source: source.mode,
        source_name: name,
        predicted_class: res.predicted_class,
        confidence: res.confidence,
        anomalous: res.anomalous ?? false,
        modality: 'vibration',
        is_replay: isReplay,
        result: res,
      });
    } catch (err: any) {
      handleApiError(err);
    } finally {
      setLoading(false);
    }
  };

  // Sound Analysis
  const handleSoundAnalyse = async () => {
    setError(null);

    // Validate machine type
    if (soundMachineType !== 'fan' && soundMachineType !== 'bearing') {
      setError("Unsupported machine type. Machine type must be 'fan' or 'bearing'.");
      return;
    }

    // Check model availability from GET /models
    if (soundModelInfo && !soundModelInfo.is_available) {
      setError(
        `Acoustic model for '${soundMachineType}' is unavailable or missing checkpoint (${soundModelInfo.checkpoint_path}).`,
      );
      return;
    }

    setLoading(true);
    setSoundResult(null);

    try {
      let res: ModelResult;
      let sourceName: string;
      const isReplay = soundMode === 'sample';

      if (soundMode === 'upload') {
        if (!soundFile) {
          setError('Please select an audio .wav file to upload.');
          setLoading(false);
          return;
        }

        // Validate file extension
        const ext = soundFile.name.split('.').pop()?.toLowerCase();
        if (ext !== 'wav') {
          setError(`Wrong file type (.${ext}). Acoustic analysis strictly requires a .wav audio file.`);
          setLoading(false);
          return;
        }

        res = await predictAudio(soundFile, soundMachineType);
        sourceName = soundFile.name;
      } else {
        if (!selectedSoundSampleId) {
          setError('Please select a recorded sound sample.');
          setLoading(false);
          return;
        }
        res = await predictSample('audio', selectedSoundSampleId);
        const matched = soundSamples.find((s) => s.sample_id === selectedSoundSampleId);
        sourceName = matched ? `${matched.filename} (${matched.label})` : selectedSoundSampleId;
      }

      setSoundResult(res);

      saveAnalysisHistory({
        source: soundMode === 'upload' ? 'upload' : 'recording',
        source_name: sourceName,
        predicted_class: res.prediction_label || res.status,
        confidence: res.confidence ?? 0.85,
        anomalous: res.status === 'ANOMALOUS',
        modality: 'audio',
        is_replay: res.is_replay ?? isReplay,
        model_result: res,
      });
    } catch (err: any) {
      handleApiError(err);
    } finally {
      setLoading(false);
    }
  };

  // Motor Analysis
  const handleMotorAnalyse = async () => {
    setError(null);

    if (motorModelInfo && !motorModelInfo.is_available) {
      setError(
        `Motor PdM model is unavailable or checkpoint missing (${motorModelInfo.checkpoint_path}).`,
      );
      return;
    }

    setLoading(true);
    setMotorResult(null);

    try {
      let res: ModelResult;
      let sourceName: string;
      const isReplay = motorMode === 'sample';
      const tempNum = motorTemp ? parseFloat(motorTemp) : undefined;

      if (motorMode === 'upload') {
        if (!motorFile) {
          setError('Please select a motor recording file (.csv, .mat, or .npy).');
          setLoading(false);
          return;
        }

        const ext = motorFile.name.split('.').pop()?.toLowerCase();
        if (ext !== 'csv' && ext !== 'mat' && ext !== 'npy') {
          setError(
            `Wrong file type (.${ext}). Motor analysis requires a .csv, .mat, or .npy recording file.`,
          );
          setLoading(false);
          return;
        }

        res = await predictMotorSignal(motorFile, tempNum);
        sourceName = motorFile.name;
      } else {
        if (!selectedMotorSampleId) {
          setError('Please select a recorded motor sample.');
          setLoading(false);
          return;
        }
        res = await predictSample('motor', selectedMotorSampleId);
        const matched = motorSamples.find((s) => s.sample_id === selectedMotorSampleId);
        sourceName = matched ? `${matched.filename} (${matched.label})` : selectedMotorSampleId;
      }

      setMotorResult(res);

      saveAnalysisHistory({
        source: motorMode === 'upload' ? 'upload' : 'recording',
        source_name: sourceName,
        predicted_class: res.prediction_label || res.status,
        confidence: res.confidence ?? 0.85,
        anomalous: res.status === 'ANOMALOUS',
        modality: 'motor',
        is_replay: res.is_replay ?? isReplay,
        model_result: res,
      });
    } catch (err: any) {
      handleApiError(err);
    } finally {
      setLoading(false);
    }
  };

  // "Analyse with all models" (Fusion)
  const handleAnalyseWithAllModels = async () => {
    setError(null);
    setLoading(true);
    setFusionResult(null);

    try {
      const activeResults: ModelResult[] = [];
      const resultMap: {
        vibration?: ModelResult | null;
        audio?: ModelResult | null;
        motor?: ModelResult | null;
      } = { vibration: null, audio: null, motor: null };

      // 1. Vibration
      if (lastVibSource) {
        try {
          let vRes: ModelResult;
          if (lastVibSource.mode === 'upload') {
            vRes = await predictVibrationSignal(
              lastVibSource.file,
              lastVibSource.samplingRate,
              lastVibSource.channel,
            );
          } else {
            vRes = await predictSample('vibration', lastVibSource.recording.id);
          }
          activeResults.push(vRes);
          resultMap.vibration = vRes;
        } catch {
          // ignore modality run error
        }
      }

      // 2. Sound
      if (soundFile || (soundMode === 'sample' && selectedSoundSampleId)) {
        try {
          let aRes: ModelResult;
          if (soundMode === 'upload' && soundFile) {
            aRes = await predictAudio(soundFile, soundMachineType);
          } else {
            aRes = await predictSample('audio', selectedSoundSampleId || 'fan_001');
          }
          activeResults.push(aRes);
          resultMap.audio = aRes;
        } catch {
          // ignore
        }
      }

      // 3. Motor
      if (motorFile || (motorMode === 'sample' && selectedMotorSampleId)) {
        try {
          let mRes: ModelResult;
          if (motorMode === 'upload' && motorFile) {
            mRes = await predictMotorSignal(motorFile, motorTemp ? parseFloat(motorTemp) : undefined);
          } else {
            mRes = await predictSample('motor', selectedMotorSampleId || 'paderborn_demo_normal');
          }
          activeResults.push(mRes);
          resultMap.motor = mRes;
        } catch {
          // ignore
        }
      }

      setFusionResultsMap(resultMap);

      if (activeResults.length === 0) {
        setError(
          'No inputs provided for any modality. Please upload a file or select a sample in at least one tab (Vibration, Sound, or Motor).',
        );
        setLoading(false);
        return;
      }

      const fusionVerdict = await postFusion({
        results: activeResults,
        method: 'max',
      });

      setFusionResult(fusionVerdict);

      // Save to history with modality and replay flag
      const hasReplay = activeResults.some((r) => r.is_replay);
      saveAnalysisHistory({
        source: 'upload',
        source_name: `Multimodal Fusion (${activeResults.length}/3 modalities active)`,
        predicted_class: fusionVerdict.fused_status,
        confidence: fusionVerdict.confidence,
        anomalous: fusionVerdict.fused_status !== 'NORMAL',
        modality: 'fusion',
        is_replay: hasReplay,
        fusion_result: fusionVerdict,
      });

      setActiveTab('multimodal');
    } catch (err: any) {
      handleApiError(err);
    } finally {
      setLoading(false);
    }
  };

  const handleApiError = (err: any) => {
    if (err && typeof err === 'object' && 'response' in err) {
      const axiosErr = err as { response?: { data?: { detail?: string } } };
      setError(axiosErr.response?.data?.detail ?? 'An error occurred during inference.');
    } else if (err instanceof Error) {
      setError(err.message);
    } else {
      setError('An unknown error occurred.');
    }
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-12">
      {/* Top Header & Multimodal Action */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-white/5 pb-4">
        <div>
          <h2 className="text-2xl font-bold text-gray-100 tracking-tight">Diagnostic Analysis</h2>
          <p className="text-sm text-gray-400 mt-1">
            Sensor-specific diagnostics and multimodal consensus fusion.
          </p>
        </div>

        {/* Global "Analyse with all models" button */}
        <button
          onClick={handleAnalyseWithAllModels}
          disabled={loading}
          className="inline-flex items-center justify-center gap-2 px-5 py-2.5 rounded-lg bg-gradient-to-r from-brand-500 to-brand-600 hover:from-brand-600 hover:to-brand-700 text-white text-sm font-semibold shadow-lg shadow-brand-500/20 transition-all disabled:opacity-50"
        >
          {loading ? (
            <>
              <Spinner size="sm" /> Analyzing...
            </>
          ) : (
            <>
              <span>⚡</span> Analyse with all models
            </>
          )}
        </button>
      </div>

      {/* Tabs Bar */}
      <div className="flex items-center gap-2 border-b border-white/10 pb-1">
        <button
          onClick={() => setActiveTab('vibration')}
          className={`flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-semibold transition-all ${
            activeTab === 'vibration'
              ? 'bg-brand-500/15 text-brand-300 border-b-2 border-brand-500'
              : 'text-gray-400 hover:text-gray-200 hover:bg-white/5'
          }`}
        >
          <span>📊</span> Vibration
        </button>

        <button
          onClick={() => setActiveTab('sound')}
          className={`flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-semibold transition-all ${
            activeTab === 'sound'
              ? 'bg-brand-500/15 text-brand-300 border-b-2 border-brand-500'
              : 'text-gray-400 hover:text-gray-200 hover:bg-white/5'
          }`}
        >
          <span>🔊</span> Sound
        </button>

        <button
          onClick={() => setActiveTab('motor')}
          className={`flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-semibold transition-all ${
            activeTab === 'motor'
              ? 'bg-brand-500/15 text-brand-300 border-b-2 border-brand-500'
              : 'text-gray-400 hover:text-gray-200 hover:bg-white/5'
          }`}
        >
          <span>⚡</span> Motor
        </button>

        <button
          onClick={() => setActiveTab('multimodal')}
          className={`flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-semibold transition-all ${
            activeTab === 'multimodal'
              ? 'bg-brand-500/15 text-brand-300 border-b-2 border-brand-500'
              : 'text-gray-400 hover:text-gray-200 hover:bg-white/5'
          }`}
        >
          <span>🎛️</span> Multimodal Fusion
        </button>
      </div>

      {/* Error Banner */}
      {error && <ErrorBanner message={error} onDismiss={() => setError(null)} />}

      {/* ==================================================================== */}
      {/* TAB 1: VIBRATION (Unchanged workflow, with dynamic limitation note)   */}
      {/* ==================================================================== */}
      {activeTab === 'vibration' && (
        <div className="space-y-6 page-enter">
          {/* Dynamic Dataset Limitation Note from GET /models */}
          <div className="bg-brand-500/10 border border-brand-500/20 rounded-lg px-4 py-3 flex items-start gap-3">
            <span className="text-brand-400 text-sm shrink-0 mt-0.5">ℹ️</span>
            <p className="text-xs text-gray-300 leading-relaxed">
              <strong className="text-gray-100">Dataset Limitation:</strong>{' '}
              {loadingModels
                ? 'Loading model specifications...'
                : `Model trained on ${
                    vibModelInfo?.training_dataset || 'CWRU Bearing Data'
                  }. Threshold calibrated to ${
                    vibModelInfo?.threshold ?? 1.24
                  }. Inference on other machinery is unvalidated.`}
            </p>
          </div>

          {/* Existing Vibration Source Selector */}
          <SourceSelector onSourceReady={handleVibrationAnalyse} disabled={loading} />

          {/* Vibration Results */}
          {vibResult && (
            <div className="space-y-5 page-enter">
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                <Card>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-500 font-medium uppercase">Status</span>
                    <StatusBadge anomalous={vibResult.anomalous ?? false} />
                  </div>
                  {vibResult.true_label && (
                    <p className="text-xs text-gray-500 mt-2">
                      True label: <strong className="text-gray-300">{vibResult.true_label}</strong>
                    </p>
                  )}
                </Card>

                <MetricCard
                  label="Predicted Class"
                  value={vibResult.predicted_class}
                  sub={`${(vibResult.confidence * 100).toFixed(1)}% confidence`}
                  icon="🎯"
                  accent={vibResult.anomalous ? 'anomalous' : 'healthy'}
                />

                <MetricCard
                  label="EDR Score"
                  value={vibResult.edr_mean != null ? vibResult.edr_mean.toFixed(3) : 'N/A'}
                  sub={
                    vibResult.edr_threshold != null
                      ? `Threshold: ${vibResult.edr_threshold.toFixed(3)}`
                      : undefined
                  }
                  icon="📈"
                  accent={vibResult.anomalous ? 'anomalous' : 'brand'}
                />

                <MetricCard
                  label="Segments Flagged"
                  value={
                    vibResult.fraction_segments_flagged != null
                      ? `${(vibResult.fraction_segments_flagged * 100).toFixed(0)}%`
                      : 'N/A'
                  }
                  sub={`${vibResult.n_segments} total segments`}
                  icon="🔍"
                  accent={
                    (vibResult.fraction_segments_flagged ?? 0) > 0.5 ? 'anomalous' : 'healthy'
                  }
                />
              </div>

              <p className="text-xs text-gray-500">
                Source: <code className="text-gray-300">{vibSourceName}</code>
              </p>

              {embeddingMap && (
                <ResultCharts
                  result={vibResult}
                  bgPoints={embeddingMap.points}
                  classes={embeddingMap.classes}
                />
              )}
            </div>
          )}
        </div>
      )}

      {/* ==================================================================== */}
      {/* TAB 2: SOUND (Acoustic Autoencoder Analysis)                          */}
      {/* ==================================================================== */}
      {activeTab === 'sound' && (
        <div className="space-y-6 page-enter">
          {/* Dynamic Dataset Limitation Note from GET /models */}
          <div className="bg-brand-500/10 border border-brand-500/20 rounded-lg px-4 py-3 flex items-start gap-3">
            <span className="text-brand-400 text-sm shrink-0 mt-0.5">ℹ️</span>
            <p className="text-xs text-gray-300 leading-relaxed">
              <strong className="text-gray-100">Dataset Limitation:</strong>{' '}
              {loadingModels
                ? 'Loading model specifications...'
                : `Acoustic autoencoder trained on ${
                    soundModelInfo?.training_dataset || 'MIMII DG Acoustic Dataset'
                  } (${soundMachineType}). Threshold metric: ${
                    soundModelInfo?.threshold_metric || '95th-percentile error'
                  } (${soundModelInfo?.threshold ?? 0.05}).`}
            </p>
          </div>

          {/* Sound Source Selector Card */}
          <Card title="Acoustic Signal Source Selection">
            <div className="space-y-4">
              {/* Mode Toggle & Machine Type */}
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div className="flex items-center gap-2 bg-surface-2 dark:bg-white/5 p-1 rounded-lg border border-white/10">
                  <button
                    onClick={() => setSoundMode('upload')}
                    className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${
                      soundMode === 'upload'
                        ? 'bg-brand-500 text-white'
                        : 'text-gray-400 hover:text-gray-200'
                    }`}
                  >
                    Upload .wav
                  </button>
                  <button
                    onClick={() => setSoundMode('sample')}
                    className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${
                      soundMode === 'sample'
                        ? 'bg-brand-500 text-white'
                        : 'text-gray-400 hover:text-gray-200'
                    }`}
                  >
                    Recorded Sample
                  </button>
                </div>

                {/* Optional Machine Type Selector */}
                <div className="flex items-center gap-2">
                  <label htmlFor="sound-machine-type" className="text-xs text-gray-400 font-medium">Machine Type:</label>
                  <select
                    id="sound-machine-type"
                    value={soundMachineType}
                    onChange={(e) => setSoundMachineType(e.target.value)}
                    className="bg-surface-2 dark:bg-white/5 border border-white/10 rounded-lg px-3 py-1.5 text-xs text-gray-200 focus:outline-none focus:ring-2 focus:ring-brand-500"
                  >
                    <option value="fan" className="bg-surface-1">
                      Industrial Fan (MIMII DG)
                    </option>
                    <option value="bearing" className="bg-surface-1">
                      Industrial Bearing (MIMII DG)
                    </option>
                  </select>
                </div>
              </div>

              {/* Upload Input */}
              {soundMode === 'upload' ? (
                <div className="border border-dashed border-white/15 rounded-lg p-6 text-center space-y-2">
                  <span className="text-3xl text-gray-500">🎙️</span>
                  <p className="text-xs text-gray-300 font-medium">
                    Select an audio .wav recording file
                  </p>
                  <input
                    type="file"
                    accept=".wav,audio/wav"
                    onChange={(e) => {
                      if (e.target.files && e.target.files[0]) {
                        const f = e.target.files[0];
                        const ext = f.name.split('.').pop()?.toLowerCase();
                        if (ext !== 'wav') {
                          setError('Wrong file type: Acoustic analysis strictly requires a .wav file.');
                          setSoundFile(null);
                        } else {
                          setError(null);
                          setSoundFile(f);
                        }
                      }
                    }}
                    className="text-xs text-gray-400 file:mr-3 file:py-1.5 file:px-3 file:rounded-md file:border-0 file:text-xs file:font-semibold file:bg-brand-500 file:text-white hover:file:bg-brand-600 cursor-pointer"
                  />
                  {soundFile && (
                    <p className="text-xs text-emerald-400 font-mono mt-2">
                      Selected: {soundFile.name} ({(soundFile.size / 1024).toFixed(1)} KB)
                    </p>
                  )}
                </div>
              ) : (
                /* Recorded Sample Select */
                <div className="space-y-2">
                  <label htmlFor="sound-sample-select" className="text-xs text-gray-400 font-medium">
                    Recorded Held-Out MIMII DG Test Clip:
                  </label>
                  <select
                    id="sound-sample-select"
                    value={selectedSoundSampleId}
                    onChange={(e) => setSelectedSoundSampleId(e.target.value)}
                    className="w-full bg-surface-2 dark:bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-xs text-gray-200 focus:outline-none focus:ring-2 focus:ring-brand-500"
                  >
                    {soundSamples.map((s) => (
                      <option key={s.sample_id} value={s.sample_id} className="bg-surface-1">
                        {s.filename} — {s.description} ({s.label})
                      </option>
                    ))}
                  </select>
                </div>
              )}

              {/* Action Button */}
              <div className="flex justify-end pt-2">
                <button
                  onClick={handleSoundAnalyse}
                  disabled={loading || (soundMode === 'upload' && !soundFile)}
                  className="px-5 py-2 rounded-lg bg-brand-500 hover:bg-brand-600 text-white text-xs font-semibold shadow-md transition-colors disabled:opacity-50 flex items-center gap-2"
                >
                  {loading ? (
                    <>
                      <Spinner size="sm" /> Processing Audio...
                    </>
                  ) : (
                    'Diagnose Acoustic Health'
                  )}
                </button>
              </div>
            </div>
          </Card>

          {/* Sound Result View */}
          {soundResult && (
            <ModelResultCard
              title="Acoustic Autoencoder Diagnosis"
              result={soundResult}
              plot={
                <SoundPlot
                  frameScores={soundResult.details?.frame_scores}
                  threshold={soundResult.threshold}
                  samplingRate={soundResult.input_info?.sampling_rate ?? 16000}
                />
              }
            />
          )}
        </div>
      )}

      {/* ==================================================================== */}
      {/* TAB 3: MOTOR (Paderborn Current & Temp Benchmark)                     */}
      {/* ==================================================================== */}
      {activeTab === 'motor' && (
        <div className="space-y-6 page-enter">
          {/* Dynamic Dataset Limitation Note from GET /models */}
          <div className="bg-brand-500/10 border border-brand-500/20 rounded-lg px-4 py-3 flex items-start gap-3">
            <span className="text-brand-400 text-sm shrink-0 mt-0.5">ℹ️</span>
            <p className="text-xs text-gray-300 leading-relaxed">
              <strong className="text-gray-100">Dataset Limitation:</strong>{' '}
              {loadingModels
                ? 'Loading model specifications...'
                : `Motor current and thermal diagnostics calibrated on ${
                    motorModelInfo?.training_dataset || 'Paderborn Motor Benchmark'
                  }. Threshold: ${
                    motorModelInfo?.threshold ?? 0.5
                  }. Calibrated for synchronous motor test bench operations.`}
            </p>
          </div>

          {/* Motor Source Selector Card */}
          <Card title="Motor Telemetry Input Selection">
            <div className="space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div className="flex items-center gap-2 bg-surface-2 dark:bg-white/5 p-1 rounded-lg border border-white/10">
                  <button
                    onClick={() => setMotorMode('upload')}
                    className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${
                      motorMode === 'upload'
                        ? 'bg-brand-500 text-white'
                        : 'text-gray-400 hover:text-gray-200'
                    }`}
                  >
                    Upload File (.csv / .mat)
                  </button>
                  <button
                    onClick={() => setMotorMode('sample')}
                    className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${
                      motorMode === 'sample'
                        ? 'bg-brand-500 text-white'
                        : 'text-gray-400 hover:text-gray-200'
                    }`}
                  >
                    Recorded Sample
                  </button>
                </div>

                {/* Optional Temperature Field */}
                <div className="flex items-center gap-2">
                  <label htmlFor="motor-temp-input" className="text-xs text-gray-400 font-medium">Bearing Temp (°C):</label>
                  <input
                    id="motor-temp-input"
                    type="number"
                    step="0.1"
                    value={motorTemp}
                    onChange={(e) => setMotorTemp(e.target.value)}
                    placeholder="45.0"
                    className="w-24 bg-surface-2 dark:bg-white/5 border border-white/10 rounded-lg px-2.5 py-1.5 text-xs text-gray-200 font-mono focus:outline-none focus:ring-2 focus:ring-brand-500"
                  />
                </div>
              </div>

              {/* Upload Input */}
              {motorMode === 'upload' ? (
                <div className="border border-dashed border-white/15 rounded-lg p-6 text-center space-y-2">
                  <span className="text-3xl text-gray-500">⚡</span>
                  <p className="text-xs text-gray-300 font-medium">
                    Upload motor current recording (.csv, .mat, or .npy)
                  </p>
                  <input
                    type="file"
                    accept=".csv,.mat,.npy"
                    onChange={(e) => {
                      if (e.target.files && e.target.files[0]) {
                        const f = e.target.files[0];
                        const ext = f.name.split('.').pop()?.toLowerCase();
                        if (ext !== 'csv' && ext !== 'mat' && ext !== 'npy') {
                          setError('Wrong file type: Motor analysis requires a .csv, .mat, or .npy file.');
                          setMotorFile(null);
                        } else {
                          setError(null);
                          setMotorFile(f);
                        }
                      }
                    }}
                    className="text-xs text-gray-400 file:mr-3 file:py-1.5 file:px-3 file:rounded-md file:border-0 file:text-xs file:font-semibold file:bg-brand-500 file:text-white hover:file:bg-brand-600 cursor-pointer"
                  />
                  {motorFile && (
                    <p className="text-xs text-emerald-400 font-mono mt-2">
                      Selected: {motorFile.name} ({(motorFile.size / 1024).toFixed(1)} KB)
                    </p>
                  )}
                </div>
              ) : (
                /* Recorded Sample Select */
                <div className="space-y-2">
                  <label htmlFor="motor-sample-select" className="text-xs text-gray-400 font-medium">
                    Recorded Paderborn Benchmark Sample:
                  </label>
                  <select
                    id="motor-sample-select"
                    value={selectedMotorSampleId}
                    onChange={(e) => setSelectedMotorSampleId(e.target.value)}
                    className="w-full bg-surface-2 dark:bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-xs text-gray-200 focus:outline-none focus:ring-2 focus:ring-brand-500"
                  >
                    {motorSamples.map((s) => (
                      <option key={s.sample_id} value={s.sample_id} className="bg-surface-1">
                        {s.filename} — {s.description} ({s.label})
                      </option>
                    ))}
                  </select>
                </div>
              )}

              {/* Action Button */}
              <div className="flex justify-end pt-2">
                <button
                  onClick={handleMotorAnalyse}
                  disabled={loading || (motorMode === 'upload' && !motorFile)}
                  className="px-5 py-2 rounded-lg bg-brand-500 hover:bg-brand-600 text-white text-xs font-semibold shadow-md transition-colors disabled:opacity-50 flex items-center gap-2"
                >
                  {loading ? (
                    <>
                      <Spinner size="sm" /> Analyzing Motor...
                    </>
                  ) : (
                    'Diagnose Electrical & Thermal Health'
                  )}
                </button>
              </div>
            </div>
          </Card>

          {/* Motor Result View */}
          {motorResult && (
            <ModelResultCard
              title="Motor Electrical & Thermal Diagnosis"
              result={motorResult}
              plot={<MotorSeriesPlot features={motorResult.details?.features} />}
            />
          )}
        </div>
      )}

      {/* ==================================================================== */}
      {/* TAB 4: MULTIMODAL FUSION RESULT                                      */}
      {/* ==================================================================== */}
      {activeTab === 'multimodal' && (
        <div className="space-y-6 page-enter">
          <div className="flex items-center justify-between">
            <h3 className="text-base font-bold text-gray-100">
              Multimodal Consensus Fusion Verdict
            </h3>
            <button
              onClick={handleAnalyseWithAllModels}
              disabled={loading}
              className="px-4 py-2 rounded-lg bg-brand-500 hover:bg-brand-600 text-white text-xs font-semibold transition-colors disabled:opacity-50 flex items-center gap-2"
            >
              {loading ? <Spinner size="sm" /> : 'Run Fusion Across All Inputs'}
            </button>
          </div>

          {!fusionResult ? (
            <Card>
              <div className="text-center py-12 text-xs text-gray-400 space-y-2">
                <span className="text-3xl block mb-2">🎛️</span>
                <p className="font-semibold text-gray-300 text-sm">
                  No Multimodal Fusion Executed Yet
                </p>
                <p className="max-w-md mx-auto text-gray-500">
                  Click &quot;Analyse with all models&quot; to execute late consensus fusion across your provided
                  Vibration, Sound, and Motor inputs. Missing modalities will be marked &quot;not provided&quot;.
                </p>
              </div>
            </Card>
          ) : (
            <div className="space-y-5 page-enter">
              {/* Replay Banner if any input was from replay */}
              {fusionResult.is_replay && (
                <div className="w-full bg-amber-500/20 border border-amber-500/40 text-amber-300 px-4 py-2.5 rounded-lg text-xs font-semibold flex items-center justify-between">
                  <span>⚠️ REPLAYED DATA: contains recorded benchmark samples</span>
                  <span className="font-mono text-[10px] uppercase">Held-Out Telemetry</span>
                </div>
              )}

              {/* Fused Overview Cards */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <MetricCard
                  label="Fused Verdict"
                  value={fusionResult.fused_status}
                  sub="Consensus State"
                  accent={
                    fusionResult.fused_status === 'ALARM'
                      ? 'anomalous'
                      : fusionResult.fused_status === 'WARNING'
                      ? 'warning'
                      : 'healthy'
                  }
                  icon={
                    fusionResult.fused_status === 'ALARM'
                      ? '🚨'
                      : fusionResult.fused_status === 'WARNING'
                      ? '⚠️'
                      : '✅'
                  }
                />

                <MetricCard
                  label="Fused Anomaly Score"
                  value={fusionResult.fused_score.toFixed(3)}
                  sub={`Method: ${fusionResult.method.toUpperCase()}`}
                  accent={fusionResult.fused_score >= 1.0 ? 'anomalous' : 'brand'}
                  icon="📈"
                />

                <MetricCard
                  label="Confidence Level"
                  value={`${(fusionResult.confidence * 100).toFixed(0)}%`}
                  sub="Agreement & Modality Coverage"
                  accent="brand"
                  icon="🎯"
                />
              </div>

              {/* Modality Status Breakdown (Marking missing as 'not provided') */}
              <Card title="Modality Input Consensus Breakdown">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {(['vibration', 'audio', 'motor'] as const).map((mod) => {
                    const res = fusionResultsMap[mod];
                    const isMissing = !res;

                    return (
                      <div
                        key={mod}
                        className={`p-4 rounded-lg border text-xs space-y-2 ${
                          isMissing
                            ? 'bg-white/5 border-white/10 opacity-70'
                            : res?.status === 'ANOMALOUS'
                            ? 'bg-rose-500/10 border-rose-500/30'
                            : 'bg-emerald-500/10 border-emerald-500/30'
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <span className="font-bold text-gray-200 uppercase tracking-wider">
                            {mod === 'audio' ? 'Sound (Acoustic)' : mod}
                          </span>
                          {isMissing ? (
                            <span className="px-2 py-0.5 rounded text-[10px] font-mono font-semibold bg-gray-500/20 text-gray-400 border border-gray-500/30">
                              not provided
                            </span>
                          ) : (
                            <StatusBadge anomalous={res.status === 'ANOMALOUS'} />
                          )}
                        </div>

                        {isMissing ? (
                          <p className="text-gray-500 text-[11px]">
                            Modality not provided for this fusion run.
                          </p>
                        ) : (
                          <div className="space-y-1 text-gray-300">
                            <div>
                              Score Ratio:{' '}
                              <span className="font-mono font-bold text-white">
                                {res.score_ratio.toFixed(2)}x
                              </span>
                            </div>
                            <div className="text-[11px] text-gray-400">
                              Label: {res.prediction_label || 'Normal'}
                            </div>
                            {res.is_replay && (
                              <div className="text-[10px] text-amber-300 font-mono">
                                • Replay sample
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </Card>

              {/* Reasons Log */}
              {fusionResult.reasons && fusionResult.reasons.length > 0 && (
                <Card title="Fusion Decision Reasons">
                  <ul className="list-disc list-inside text-xs text-gray-300 space-y-1">
                    {fusionResult.reasons.map((r, i) => (
                      <li key={i}>{r}</li>
                    ))}
                  </ul>
                </Card>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Standard ModelResult Card Component (Sound & Motor)
// ---------------------------------------------------------------------------

interface ModelResultCardProps {
  title: string;
  result: ModelResult;
  plot: React.ReactNode;
}

function ModelResultCard({ title, result, plot }: ModelResultCardProps) {
  const isUnavailable = result.status === 'UNAVAILABLE' || !!result.error;
  const isAnomalous = result.status === 'ANOMALOUS';
  const ratio = result.score_ratio ?? 0;
  const maxRange = Math.max(2.0, ratio * 1.15, 1.2);
  const thresholdPercent = Math.min(100, Math.max(5, (1.0 / maxRange) * 100));
  const fillPercent = isUnavailable
    ? 0
    : Math.min(100, Math.max(0, (ratio / maxRange) * 100));

  return (
    <div className="space-y-5 page-enter">
      {/* Replay Banner if recording sample was used */}
      {result.is_replay && (
        <div className="w-full bg-amber-500/20 border border-amber-500/40 text-amber-300 px-4 py-2.5 rounded-lg text-xs font-semibold flex items-center justify-between">
          <span>⚠️ REPLAY: recorded benchmark sample, not a live sensor</span>
          <span className="font-mono text-[10px] uppercase">Recorded Dataset</span>
        </div>
      )}

      {/* Summary Metrics */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <div className="flex items-center justify-between">
            <span className="text-xs text-gray-500 font-medium uppercase">Status</span>
            <StatusBadge anomalous={isAnomalous} />
          </div>
          <p className="text-xs text-gray-400 mt-2 font-mono">
            Ratio: <strong className={ratio >= 1.0 ? 'text-rose-400' : 'text-emerald-400'}>{ratio.toFixed(2)}x</strong>
          </p>
        </Card>

        <MetricCard
          label="Predicted Label"
          value={result.prediction_label || (isAnomalous ? 'Anomaly' : 'Normal')}
          sub={result.confidence ? `${(result.confidence * 100).toFixed(1)}% confidence` : undefined}
          icon="🎯"
          accent={isAnomalous ? 'anomalous' : 'healthy'}
        />

        <MetricCard
          label="Anomaly Score"
          value={result.anomaly_score.toFixed(3)}
          sub={`Threshold: ${result.threshold.toFixed(3)}`}
          icon="📈"
          accent={isAnomalous ? 'anomalous' : 'brand'}
        />

        <MetricCard
          label="Source Input"
          value={result.input_info?.source_name || 'Signal input'}
          sub={
            result.input_info?.sampling_rate
              ? `${(result.input_info.sampling_rate / 1000).toFixed(1)} kHz`
              : undefined
          }
          icon="📁"
          accent="brand"
        />
      </div>

      {/* Visual Score vs Threshold Bar */}
      <Card title="Score vs Decision Threshold (1.0 Ratio)">
        <div className="space-y-2">
          <div className="relative w-full h-3.5 bg-surface-2 dark:bg-white/5 rounded-full overflow-hidden border border-white/10">
            <div
              className={`h-full rounded-full transition-all duration-300 ${
                ratio >= 1.0
                  ? 'bg-gradient-to-r from-amber-500 to-rose-500'
                  : 'bg-gradient-to-r from-emerald-500 to-teal-400'
              }`}
              style={{ width: `${fillPercent}%` }}
            />
            <div
              className="absolute top-0 bottom-0 w-0.5 bg-rose-400 z-10"
              style={{ left: `${thresholdPercent}%` }}
            />
          </div>

          <div className="flex justify-between text-[10px] text-gray-500 font-mono">
            <span>0.0</span>
            <span className="text-rose-400 font-semibold">| 1.0 Threshold</span>
            <span>{maxRange.toFixed(1)}x</span>
          </div>
        </div>
      </Card>

      {/* Plot Card */}
      <Card title={`${title} — Diagnostic Plot`}>{plot}</Card>

      {/* Input Info Metadata */}
      {result.input_info && (
        <Card title="Input Signal Diagnostics & Acquisition Metadata">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
            {Object.entries(result.input_info).map(([k, v]) => (
              <div key={k}>
                <span className="text-gray-500 block capitalize">{k.replace(/_/g, ' ')}</span>
                <span className="text-gray-300 font-mono font-medium">{String(v)}</span>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
