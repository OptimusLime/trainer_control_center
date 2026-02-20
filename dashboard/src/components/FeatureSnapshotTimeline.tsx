/**
 * FeatureSnapshotTimeline — React island for animating feature weight evolution.
 *
 * Shows an 8x8 grid of encoder weight features at a specific training step.
 * A timeline scrubber + play button lets you animate through snapshots.
 * Replacement events are marked on the timeline.
 *
 * Data comes from FeatureSnapshotRecorder (captured every 500 steps + on
 * replacement events) via GET /eval/features/snapshots.
 */
import { useEffect, useRef, useState, useCallback } from 'react';
import { useStore } from '@nanostores/react';
import {
  $snapshotTags,
  $snapshotIndex,
  $snapshotFrames,
  $snapshotLoading,
  $snapshotCurrentStep,
  fetchSnapshotTags,
  loadSnapshotTimeline,
  loadSnapshotFrame,
  preloadAllSnapshotFrames,
} from '../lib/store';

const GRID_COLS = 8;
const FEATURE_SCALE = 3; // 3x = 84px per 28px feature

export default function FeatureSnapshotTimeline() {
  const tags = useStore($snapshotTags);
  const index = useStore($snapshotIndex);
  const frames = useStore($snapshotFrames);
  const loading = useStore($snapshotLoading);
  const currentStep = useStore($snapshotCurrentStep);

  const [selectedTag, setSelectedTag] = useState('');
  const [playing, setPlaying] = useState(false);
  const [playSpeed, setPlaySpeed] = useState(200); // ms per frame
  const playRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Fetch tags on mount
  useEffect(() => {
    fetchSnapshotTags();
  }, []);

  // Auto-select first tag
  useEffect(() => {
    if (tags.length > 0 && !selectedTag) {
      setSelectedTag(tags[0]);
    }
  }, [tags, selectedTag]);

  // Stop playback on unmount
  useEffect(() => {
    return () => {
      if (playRef.current) clearInterval(playRef.current);
    };
  }, []);

  // Playback loop
  useEffect(() => {
    if (!playing || !index || index.steps.length === 0) {
      if (playRef.current) {
        clearInterval(playRef.current);
        playRef.current = null;
      }
      return;
    }

    const steps = index.steps.map(s => s.step);
    playRef.current = setInterval(() => {
      const cur = $snapshotCurrentStep.get();
      const curIdx = steps.indexOf(cur);
      const nextIdx = (curIdx + 1) % steps.length;
      const nextStep = steps[nextIdx];
      // Stop at end instead of looping
      if (nextIdx === 0 && curIdx >= 0) {
        setPlaying(false);
        return;
      }
      loadSnapshotFrame(selectedTag, nextStep);
    }, playSpeed);

    return () => {
      if (playRef.current) {
        clearInterval(playRef.current);
        playRef.current = null;
      }
    };
  }, [playing, index, selectedTag, playSpeed]);

  const handleTagChange = useCallback((tag: string) => {
    setSelectedTag(tag);
    setPlaying(false);
    loadSnapshotTimeline(tag);
  }, []);

  const handleScrub = useCallback((stepIdx: number) => {
    if (!index) return;
    const step = index.steps[stepIdx]?.step;
    if (step != null) {
      loadSnapshotFrame(selectedTag, step);
    }
  }, [index, selectedTag]);

  const handlePlay = useCallback(() => {
    if (!index || index.steps.length === 0) return;
    // If at end, restart from beginning
    const steps = index.steps.map(s => s.step);
    const curIdx = steps.indexOf(currentStep);
    if (curIdx === steps.length - 1) {
      loadSnapshotFrame(selectedTag, steps[0]);
    }
    setPlaying(p => !p);
  }, [index, currentStep, selectedTag]);

  const handlePreload = useCallback(() => {
    if (selectedTag) preloadAllSnapshotFrames(selectedTag);
  }, [selectedTag]);

  // No tags available — show nothing
  if (tags.length === 0) return null;

  const frame = currentStep >= 0 ? frames[currentStep] : null;
  const steps = index?.steps ?? [];
  const currentIdx = steps.findIndex(s => s.step === currentStep);
  const currentEvent = steps[currentIdx]?.event ?? '';

  const nativeW = frame?.image_shape?.[1] ?? 28;
  const nativeH = frame?.image_shape?.[0] ?? 28;
  const dispW = nativeW * FEATURE_SCALE;
  const dispH = nativeH * FEATURE_SCALE;

  return (
    <div className="panel" id="snapshot-timeline-panel">
      <div className="panel-header">
        <h3>Feature Evolution</h3>
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <select
            className="compare-select"
            value={selectedTag}
            onChange={e => handleTagChange(e.target.value)}
          >
            {tags.map(t => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
          <button
            className="btn-action"
            onClick={() => handleTagChange(selectedTag)}
            disabled={loading || !selectedTag}
          >
            {loading ? 'Loading...' : 'Load'}
          </button>
        </div>
      </div>

      {/* Timeline controls */}
      {steps.length > 0 && (
        <div style={{ padding: '8px 0', display: 'flex', flexDirection: 'column', gap: '6px' }}>
          {/* Playback controls */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <button
              className="btn-action"
              onClick={handlePlay}
              style={{ minWidth: '60px' }}
            >
              {playing ? 'Pause' : 'Play'}
            </button>
            <button
              className="btn-cp"
              onClick={handlePreload}
              disabled={loading}
              title="Preload all frames for smooth playback"
            >
              Preload All
            </button>
            <label style={{ fontSize: '0.75rem', color: '#8b949e', display: 'flex', alignItems: 'center', gap: '4px' }}>
              Speed:
              <select
                className="compare-select"
                value={playSpeed}
                onChange={e => setPlaySpeed(Number(e.target.value))}
                style={{ width: '70px', fontSize: '0.75rem' }}
              >
                <option value={500}>Slow</option>
                <option value={200}>Normal</option>
                <option value={80}>Fast</option>
              </select>
            </label>
            <span style={{ fontSize: '0.75rem', color: '#8b949e', marginLeft: 'auto' }}>
              {currentIdx >= 0 ? `${currentIdx + 1}/${steps.length}` : '--'}
              {' '}|{' '}Step {currentStep >= 0 ? currentStep.toLocaleString() : '--'}
              {currentEvent && currentEvent !== 'periodic' && (
                <span style={{ color: '#f0883e', marginLeft: '6px' }}>
                  [{currentEvent}]
                </span>
              )}
            </span>
          </div>

          {/* Scrubber */}
          <div style={{ position: 'relative' }}>
            <input
              type="range"
              min={0}
              max={steps.length - 1}
              value={currentIdx >= 0 ? currentIdx : 0}
              onChange={e => handleScrub(Number(e.target.value))}
              style={{ width: '100%' }}
            />
            {/* Event markers on the scrubber track */}
            <div style={{ position: 'relative', height: '6px', margin: '0 8px' }}>
              {steps.map((s, i) => {
                if (!s.event || s.event === 'periodic') return null;
                const pct = steps.length > 1 ? (i / (steps.length - 1)) * 100 : 0;
                const isReplace = s.event.includes('replace');
                return (
                  <div
                    key={`${s.step}-${s.event}`}
                    title={`Step ${s.step}: ${s.event}`}
                    style={{
                      position: 'absolute',
                      left: `${pct}%`,
                      top: 0,
                      width: '4px',
                      height: '6px',
                      background: isReplace ? '#f85149' : '#e3b341',
                      borderRadius: '1px',
                      transform: 'translateX(-2px)',
                    }}
                  />
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Feature grid */}
      {frame && (
        <div className="features-block">
          <div
            className="features-grid"
            style={{
              gridTemplateColumns: `repeat(${GRID_COLS}, ${dispW}px)`,
              gap: '2px',
            }}
          >
            {frame.features.map((b64, i) => (
              <img
                key={i}
                src={`data:image/png;base64,${b64}`}
                width={dispW}
                height={dispH}
                className="feature-img"
                title={`Feature ${i}`}
                alt={`Feature ${i}`}
              />
            ))}
          </div>
        </div>
      )}

      {/* Empty state */}
      {!frame && !loading && steps.length === 0 && index === null && (
        <div className="empty">
          Select a condition tag and click "Load" to see feature weight evolution over training.
          Snapshots are captured every 500 steps during gated training runs.
        </div>
      )}
      {!frame && !loading && steps.length === 0 && index !== null && (
        <div className="empty">No snapshots for this tag.</div>
      )}
    </div>
  );
}
