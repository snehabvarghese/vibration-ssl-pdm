import { useCallback, useEffect, useRef, useState } from 'react';
import { WS_BASE_URL } from '../api/client';
import type { ReplayMessage } from '../types/api';

export type ReplayPlaybackStatus =
  | 'idle'
  | 'connecting'
  | 'playing'
  | 'paused'
  | 'stopped'
  | 'completed'
  | 'error';

export interface UseReplayOptions {
  scenarioId?: string;
  speed?: number;
  autoConnect?: boolean;
  onMessage?: (message: ReplayMessage) => void;
  onError?: (error: string) => void;
  onComplete?: () => void;
}

export interface UseReplayReturn {
  status: ReplayPlaybackStatus;
  scenarioId: string | null;
  speed: number;
  latestMessage: ReplayMessage | null;
  history: ReplayMessage[];
  error: string | null;
  isReplay: boolean;
  start: (targetScenarioId?: string) => void;
  pause: () => void;
  resume: () => void;
  togglePause: () => void;
  stop: () => void;
  setSpeed: (speed: number) => void;
  reconnect: () => void;
  clearHistory: () => void;
}

/**
 * React hook wrapping the WebSocket telemetry streaming endpoint /ws/replay/{scenario_id}.
 * Provides playback lifecycle controls (start, pause, stop, speed, reconnect) and
 * maintains an accumulated history of ReplayMessage steps.
 */
export function useReplay(options: UseReplayOptions = {}): UseReplayReturn {
  const [scenarioId, setScenarioId] = useState<string | null>(options.scenarioId ?? null);
  const [status, setStatus] = useState<ReplayPlaybackStatus>('idle');
  const [speed, setSpeedState] = useState<number>(options.speed ?? 1.0);
  const [error, setError] = useState<string | null>(null);
  const [latestMessage, setLatestMessage] = useState<ReplayMessage | null>(null);
  const [history, setHistory] = useState<ReplayMessage[]>([]);

  const wsRef = useRef<WebSocket | null>(null);
  const activeScenarioRef = useRef<string | null>(options.scenarioId ?? null);
  const speedRef = useRef<number>(options.speed ?? 1.0);

  useEffect(() => {
    speedRef.current = speed;
  }, [speed]);

  const closeSocket = useCallback(() => {
    if (wsRef.current) {
      try {
        wsRef.current.close();
      } catch {
        // ignore
      }
      wsRef.current = null;
    }
  }, []);

  const stop = useCallback(() => {
    if (wsRef.current) {
      try {
        if (wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({ action: 'stop' }));
        }
        wsRef.current.close();
      } catch {
        // ignore
      }
      wsRef.current = null;
    }
    setStatus('stopped');
  }, []);

  const start = useCallback(
    (targetScenarioId?: string) => {
      const activeId = targetScenarioId ?? activeScenarioRef.current;
      if (!activeId) {
        setError('No scenario ID specified to start replay.');
        setStatus('error');
        return;
      }

      closeSocket();

      activeScenarioRef.current = activeId;
      setScenarioId(activeId);
      setError(null);
      setStatus('connecting');
      setHistory([]);
      setLatestMessage(null);

      const wsUrl = `${WS_BASE_URL}/ws/replay/${activeId}`;
      let socket: WebSocket;

      try {
        socket = new WebSocket(wsUrl);
        wsRef.current = socket;
      } catch (err: any) {
        const errMsg = `WebSocket connection error: ${err.message || 'Failed to initialize'}`;
        setError(errMsg);
        setStatus('error');
        options.onError?.(errMsg);
        return;
      }

      socket.onopen = () => {
        setStatus('playing');
        if (speedRef.current !== 1.0) {
          socket.send(JSON.stringify({ action: 'set_speed', speed: speedRef.current }));
        }
      };

      socket.onmessage = (event: MessageEvent) => {
        try {
          const raw = JSON.parse(event.data);

          if (raw.type === 'completed') {
            setStatus('completed');
            options.onComplete?.();
            return;
          }

          if (raw.error) {
            setError(raw.error);
            setStatus('error');
            options.onError?.(raw.error);
            return;
          }

          const replayMsg = raw as ReplayMessage;
          setLatestMessage(replayMsg);
          setHistory((prev) => [...prev, replayMsg]);
          options.onMessage?.(replayMsg);
        } catch (err: any) {
          console.error('Failed to parse replay WebSocket message:', err);
        }
      };

      socket.onerror = () => {
        const errMsg = 'WebSocket encountered an unexpected transmission error.';
        setError(errMsg);
        setStatus('error');
        options.onError?.(errMsg);
      };

      socket.onclose = () => {
        setStatus((prev) => (prev === 'playing' || prev === 'paused' ? 'stopped' : prev));
      };
    },
    [closeSocket, options],
  );

  const pause = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      try {
        wsRef.current.send(JSON.stringify({ action: 'pause' }));
      } catch {
        // ignore
      }
    }
    setStatus('paused');
  }, []);

  const resume = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      try {
        wsRef.current.send(JSON.stringify({ action: 'resume' }));
      } catch {
        // ignore
      }
    }
    setStatus('playing');
  }, []);

  const togglePause = useCallback(() => {
    if (status === 'playing') {
      pause();
    } else if (status === 'paused') {
      resume();
    }
  }, [pause, resume, status]);

  const setSpeed = useCallback((newSpeed: number) => {
    setSpeedState(newSpeed);
    speedRef.current = newSpeed;
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      try {
        wsRef.current.send(JSON.stringify({ action: 'set_speed', speed: newSpeed }));
      } catch {
        // ignore
      }
    }
  }, []);

  const reconnect = useCallback(() => {
    if (activeScenarioRef.current) {
      start(activeScenarioRef.current);
    }
  }, [start]);

  const clearHistory = useCallback(() => {
    setHistory([]);
    setLatestMessage(null);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      closeSocket();
    };
  }, [closeSocket]);

  const isReplay = latestMessage?.is_replay ?? (status !== 'idle');

  return {
    status,
    scenarioId,
    speed,
    latestMessage,
    history,
    error,
    isReplay,
    start,
    pause,
    resume,
    togglePause,
    stop,
    setSpeed,
    reconnect,
    clearHistory,
  };
}

export default useReplay;
