import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { AnalysisResult, MapPoint } from '../types/api';

/* ------------------------------------------------------------------ */
/*  Colour utilities                                                  */
/* ------------------------------------------------------------------ */

const CLASS_COLORS = [
  '#6366f1', '#8b5cf6', '#a78bfa', '#c084fc',
  '#f472b6', '#fb7185', '#f97316', '#facc15',
  '#4ade80', '#2dd4bf', '#22d3ee', '#38bdf8',
  '#818cf8', '#e879f9', '#fb923c', '#a3e635',
];

function classColor(idx: number): string {
  return CLASS_COLORS[idx % CLASS_COLORS.length];
}

const GRID_COLOR = 'rgba(255,255,255,0.06)';
const AXIS_COLOR = 'rgba(255,255,255,0.3)';
const HIGHLIGHT = '#f59e0b';

/* ------------------------------------------------------------------ */
/*  Waveform                                                          */
/* ------------------------------------------------------------------ */

interface WaveformChartProps {
  samples: number[];
  samplingRate: number;
}

export function WaveformChart({ samples, samplingRate }: WaveformChartProps) {
  const data = samples.map((v, i) => ({
    t: +((i / samplingRate) * 1000).toFixed(2), // ms
    v,
  }));

  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
        <CartesianGrid stroke={GRID_COLOR} strokeDasharray="3 3" />
        <XAxis
          dataKey="t"
          tick={{ fill: AXIS_COLOR, fontSize: 11 }}
          label={{ value: 'Time (ms)', position: 'insideBottomRight', offset: -5, fill: AXIS_COLOR, fontSize: 11 }}
        />
        <YAxis tick={{ fill: AXIS_COLOR, fontSize: 11 }} />
        <Tooltip
          contentStyle={{ background: '#1e1e2e', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }}
          labelStyle={{ color: '#a0a0b0' }}
        />
        <Line type="monotone" dataKey="v" stroke="#6366f1" strokeWidth={1} dot={false} name="Amplitude" />
      </LineChart>
    </ResponsiveContainer>
  );
}

/* ------------------------------------------------------------------ */
/*  EDR Trend                                                         */
/* ------------------------------------------------------------------ */

interface EdrTrendChartProps {
  scores: number[];
  threshold: number;
}

export function EdrTrendChart({ scores, threshold }: EdrTrendChartProps) {
  const data = scores.map((s, i) => ({ idx: i + 1, score: s }));

  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
        <CartesianGrid stroke={GRID_COLOR} strokeDasharray="3 3" />
        <XAxis
          dataKey="idx"
          tick={{ fill: AXIS_COLOR, fontSize: 11 }}
          label={{ value: 'Window', position: 'insideBottomRight', offset: -5, fill: AXIS_COLOR, fontSize: 11 }}
        />
        <YAxis tick={{ fill: AXIS_COLOR, fontSize: 11 }} />
        <Tooltip
          contentStyle={{ background: '#1e1e2e', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }}
          labelStyle={{ color: '#a0a0b0' }}
        />
        <ReferenceLine
          y={threshold}
          stroke="#ef4444"
          strokeDasharray="6 4"
          strokeWidth={2}
          label={{ value: 'Threshold', fill: '#ef4444', fontSize: 11, position: 'right' }}
        />
        <Line type="monotone" dataKey="score" stroke="#22d3ee" strokeWidth={2} dot={false} name="EDR Score" />
      </LineChart>
    </ResponsiveContainer>
  );
}

/* ------------------------------------------------------------------ */
/*  Embedding Scatter                                                 */
/* ------------------------------------------------------------------ */

interface EmbeddingScatterProps {
  bgPoints: MapPoint[];
  classes: string[];
  newPoints: number[][];
}

export function EmbeddingScatter({ bgPoints, classes, newPoints }: EmbeddingScatterProps) {
  // Prepare class index map
  const classIdx: Record<string, number> = {};
  classes.forEach((c, i) => { classIdx[c] = i; });

  // Background data grouped by class for legend
  const bgData = bgPoints.map((p) => ({
    x: p.x,
    y: p.y,
    label: p.class_label,
    fill: classColor(classIdx[p.class_label] ?? 0),
  }));

  // Foreground (new analysis points)
  const fgData = newPoints.map((pt, i) => ({
    x: pt[0],
    y: pt[1],
    label: `Segment ${i + 1}`,
    fill: HIGHLIGHT,
  }));

  return (
    <ResponsiveContainer width="100%" height={320}>
      <ScatterChart margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
        <CartesianGrid stroke={GRID_COLOR} strokeDasharray="3 3" />
        <XAxis
          type="number"
          dataKey="x"
          tick={{ fill: AXIS_COLOR, fontSize: 11 }}
          name="PC-1"
          label={{ value: 'PC-1', position: 'insideBottomRight', offset: -5, fill: AXIS_COLOR, fontSize: 11 }}
        />
        <YAxis
          type="number"
          dataKey="y"
          tick={{ fill: AXIS_COLOR, fontSize: 11 }}
          name="PC-2"
          label={{ value: 'PC-2', angle: -90, position: 'insideLeft', fill: AXIS_COLOR, fontSize: 11 }}
        />
        <Tooltip
          contentStyle={{ background: '#1e1e2e', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }}
          cursor={{ strokeDasharray: '3 3' }}
        />
        <Legend verticalAlign="top" wrapperStyle={{ fontSize: 11 }} />
        {/* Background */}
        <Scatter name="Training data" data={bgData} fill="#6366f1" opacity={0.35} shape="circle">
          {bgData.map((d, i) => (
            <Cell key={`bg-${i}`} fill={d.fill} />
          ))}
        </Scatter>
        {/* Foreground */}
        <Scatter name="Analysed segments" data={fgData} fill={HIGHLIGHT} opacity={0.9} shape="diamond">
          {fgData.map((d, i) => (
            <Cell key={`fg-${i}`} fill={d.fill} />
          ))}
        </Scatter>
      </ScatterChart>
    </ResponsiveContainer>
  );
}

/* ------------------------------------------------------------------ */
/*  Class Probabilities Bar                                           */
/* ------------------------------------------------------------------ */

interface ClassProbBarProps {
  probabilities: Record<string, number>;
  trueLabel?: string;
  topN?: number;
}

export function ClassProbBar({ probabilities, trueLabel, topN = 8 }: ClassProbBarProps) {
  const sorted = Object.entries(probabilities)
    .sort(([, a], [, b]) => b - a)
    .slice(0, topN)
    .map(([name, prob], i) => ({
      name,
      probability: +(prob * 100).toFixed(1),
      isTrue: name === trueLabel,
      fill: classColor(i),
    }));

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={sorted} layout="vertical" margin={{ top: 5, right: 20, left: 60, bottom: 5 }}>
        <CartesianGrid stroke={GRID_COLOR} strokeDasharray="3 3" horizontal={false} />
        <XAxis
          type="number"
          domain={[0, 100]}
          tick={{ fill: AXIS_COLOR, fontSize: 11 }}
          label={{ value: 'Probability (%)', position: 'insideBottomRight', offset: -5, fill: AXIS_COLOR, fontSize: 11 }}
        />
        <YAxis
          type="category"
          dataKey="name"
          tick={{ fill: AXIS_COLOR, fontSize: 11 }}
          width={55}
        />
        <Tooltip
          contentStyle={{ background: '#1e1e2e', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }}
          formatter={(value) => `${value}%`}
        />
        <Bar dataKey="probability" radius={[0, 4, 4, 0]} name="Confidence">
          {sorted.map((d, i) => (
            <Cell
              key={`bar-${i}`}
              fill={d.isTrue ? '#4ade80' : d.fill}
              stroke={d.isTrue ? '#4ade80' : 'none'}
              strokeWidth={d.isTrue ? 2 : 0}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/* ------------------------------------------------------------------ */
/*  Full result dashboard                                             */
/* ------------------------------------------------------------------ */

interface ResultChartsProps {
  result: AnalysisResult;
  bgPoints: MapPoint[];
  classes: string[];
}

export function ResultCharts({ result, bgPoints, classes }: ResultChartsProps) {
  return (
    <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
      {/* Waveform */}
      <div className="glass-card p-4">
        <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
          Waveform
        </h4>
        <WaveformChart
          samples={result.waveform.samples}
          samplingRate={result.waveform.sampling_rate}
        />
      </div>

      {/* EDR Trend */}
      {result.edr_scores && result.edr_threshold != null && (
        <div className="glass-card p-4">
          <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
            EDR Trend
          </h4>
          <EdrTrendChart scores={result.edr_scores} threshold={result.edr_threshold} />
        </div>
      )}

      {/* Embedding Map */}
      <div className="glass-card p-4">
        <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
          Embedding Map
        </h4>
        <EmbeddingScatter bgPoints={bgPoints} classes={classes} newPoints={result.map_xy} />
      </div>

      {/* Class Probabilities */}
      <div className="glass-card p-4">
        <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
          Class Probabilities (Top 8)
        </h4>
        <ClassProbBar probabilities={result.class_probabilities} trueLabel={result.true_label} />
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Multimodal PdM Trend Chart                                       */
/* ------------------------------------------------------------------ */

export interface MultimodalTrendPoint {
  step: number;
  timeLabel: string;
  vibrationRatio?: number | null;
  audioRatio?: number | null;
  motorRatio?: number | null;
  fusedScore?: number | null;
}

interface MultimodalTrendChartProps {
  data: MultimodalTrendPoint[];
}

export function MultimodalTrendChart({ data }: MultimodalTrendChartProps) {
  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={data} margin={{ top: 10, right: 25, left: 0, bottom: 5 }}>
        <CartesianGrid stroke={GRID_COLOR} strokeDasharray="3 3" />
        <XAxis
          dataKey="timeLabel"
          tick={{ fill: AXIS_COLOR, fontSize: 11 }}
          label={{ value: 'Step (s)', position: 'insideBottomRight', offset: -5, fill: AXIS_COLOR, fontSize: 11 }}
        />
        <YAxis
          tick={{ fill: AXIS_COLOR, fontSize: 11 }}
          domain={[0, (dataMax: number) => Math.max(1.5, Math.ceil(dataMax * 1.2))]}
          label={{ value: 'Score / Ratio', angle: -90, position: 'insideLeft', fill: AXIS_COLOR, fontSize: 11 }}
        />
        <Tooltip
          contentStyle={{ background: '#1e1e2e', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }}
          labelStyle={{ color: '#a0a0b0' }}
          formatter={(value) => [typeof value === 'number' ? value.toFixed(3) : value, '']}
        />
        <Legend verticalAlign="top" height={36} wrapperStyle={{ fontSize: 12, paddingBottom: 10 }} />
        <ReferenceLine
          y={1.0}
          stroke="#ef4444"
          strokeDasharray="5 5"
          strokeWidth={2}
          label={{
            value: 'Threshold (1.0)',
            fill: '#ef4444',
            position: 'right',
            fontSize: 11,
          }}
        />
        <Line
          type="monotone"
          dataKey="vibrationRatio"
          stroke="#38bdf8"
          strokeWidth={2}
          dot={false}
          isAnimationActive={false}
          name="Vibration Ratio"
        />
        <Line
          type="monotone"
          dataKey="audioRatio"
          stroke="#c084fc"
          strokeWidth={2}
          dot={false}
          isAnimationActive={false}
          name="Audio Ratio"
        />
        <Line
          type="monotone"
          dataKey="motorRatio"
          stroke="#f59e0b"
          strokeWidth={2}
          dot={false}
          isAnimationActive={false}
          name="Motor Ratio"
        />
        <Line
          type="monotone"
          dataKey="fusedScore"
          stroke="#10b981"
          strokeWidth={3}
          dot={false}
          isAnimationActive={false}
          name="Fused Score"
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

/* ------------------------------------------------------------------ */
/*  Acoustic Sound Plot (Frame Reconstruction Error / Energy)        */
/* ------------------------------------------------------------------ */

interface SoundPlotProps {
  frameScores?: number[];
  threshold?: number;
  samplingRate?: number;
}

export function SoundPlot({ frameScores = [], threshold, samplingRate = 16000 }: SoundPlotProps) {
  if (!frameScores || frameScores.length === 0) {
    return (
      <div className="h-48 border border-dashed border-white/10 rounded-lg flex items-center justify-center text-xs text-gray-500">
        No frame-level acoustic metrics available.
      </div>
    );
  }

  const hopSize = 512;
  const data = frameScores.map((score, i) => ({
    time: +((i * hopSize) / samplingRate).toFixed(2),
    frame: i + 1,
    score: +score.toFixed(4),
  }));

  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
        <CartesianGrid stroke={GRID_COLOR} strokeDasharray="3 3" />
        <XAxis
          dataKey="time"
          tick={{ fill: AXIS_COLOR, fontSize: 11 }}
          label={{ value: 'Time (s)', position: 'insideBottomRight', offset: -5, fill: AXIS_COLOR, fontSize: 11 }}
        />
        <YAxis tick={{ fill: AXIS_COLOR, fontSize: 11 }} />
        <Tooltip
          contentStyle={{ background: '#1e1e2e', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }}
          labelStyle={{ color: '#a0a0b0' }}
        />
        {threshold != null && threshold > 0 && (
          <ReferenceLine
            y={threshold}
            stroke="#ef4444"
            strokeDasharray="5 5"
            strokeWidth={2}
            label={{ value: 'Threshold', fill: '#ef4444', position: 'right', fontSize: 11 }}
          />
        )}
        <Line
          type="monotone"
          dataKey="score"
          stroke="#c084fc"
          strokeWidth={1.5}
          dot={false}
          name="Frame Reconstruction Error"
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

/* ------------------------------------------------------------------ */
/*  Motor Diagnostic Series Plot                                      */
/* ------------------------------------------------------------------ */

interface MotorSeriesPlotProps {
  features?: Record<string, number | undefined>;
  waveform?: number[];
}

export function MotorSeriesPlot({ features = {}, waveform }: MotorSeriesPlotProps) {
  // If waveform exists, plot waveform time series
  if (waveform && waveform.length > 0) {
    const downsampled = waveform.filter((_, i) => i % Math.max(1, Math.floor(waveform.length / 500)) === 0);
    const data = downsampled.map((val, i) => ({ idx: i, current: val }));
    return (
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
          <CartesianGrid stroke={GRID_COLOR} strokeDasharray="3 3" />
          <XAxis dataKey="idx" tick={{ fill: AXIS_COLOR, fontSize: 11 }} />
          <YAxis tick={{ fill: AXIS_COLOR, fontSize: 11 }} />
          <Tooltip
            contentStyle={{ background: '#1e1e2e', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }}
          />
          <Line
            type="monotone"
            dataKey="current"
            stroke="#f59e0b"
            strokeWidth={1.5}
            dot={false}
            name="Stator Current (A)"
          />
        </LineChart>
      </ResponsiveContainer>
    );
  }

  // Otherwise, plot diagnostic feature series bar chart
  const featureEntries = Object.entries(features)
    .filter(([, v]) => typeof v === 'number' && !isNaN(v))
    .slice(0, 8)
    .map(([key, val]) => ({
      name: key.replace(/_/g, ' '),
      value: +(val as number).toFixed(3),
    }));

  if (featureEntries.length === 0) {
    return (
      <div className="h-48 border border-dashed border-white/10 rounded-lg flex items-center justify-center text-xs text-gray-500">
        No motor diagnostic series data available.
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={featureEntries} margin={{ top: 10, right: 20, left: 10, bottom: 25 }}>
        <CartesianGrid stroke={GRID_COLOR} strokeDasharray="3 3" />
        <XAxis
          dataKey="name"
          tick={{ fill: AXIS_COLOR, fontSize: 10 }}
          angle={-25}
          textAnchor="end"
          interval={0}
        />
        <YAxis tick={{ fill: AXIS_COLOR, fontSize: 11 }} />
        <Tooltip
          contentStyle={{ background: '#1e1e2e', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }}
        />
        <Bar dataKey="value" fill="#f59e0b" radius={[4, 4, 0, 0]} name="Metric Value" />
      </BarChart>
    </ResponsiveContainer>
  );
}


