/**
 * Polling infrastructure.
 *
 * Reusable setInterval-based polling that:
 * - Calls a fetcher on an interval
 * - On success, calls an updater with the data
 * - On failure (null), does nothing (keeps existing DOM)
 * - Supports pause/resume/destroy
 * - Prevents overlapping fetches
 */

export interface Poller {
  start(): void;
  stop(): void;
  /** Force an immediate fetch (resets timer). */
  fetchNow(): void;
  destroy(): void;
}

export function createPoller<T>(
  fetcher: () => Promise<T | null>,
  updater: (data: T) => void,
  intervalMs: number,
): Poller {
  let timer: ReturnType<typeof setInterval> | null = null;
  let fetching = false;
  let destroyed = false;

  async function tick() {
    if (fetching || destroyed) return;
    fetching = true;
    try {
      const data = await fetcher();
      if (data !== null && !destroyed) {
        updater(data);
      }
    } finally {
      fetching = false;
    }
  }

  return {
    start() {
      if (destroyed) return;
      // Fetch immediately, then on interval
      tick();
      timer = setInterval(tick, intervalMs);
    },
    stop() {
      if (timer !== null) {
        clearInterval(timer);
        timer = null;
      }
    },
    fetchNow() {
      if (destroyed) return;
      // Reset timer and fetch now
      if (timer !== null) {
        clearInterval(timer);
      }
      tick();
      timer = setInterval(tick, intervalMs);
    },
    destroy() {
      destroyed = true;
      if (timer !== null) {
        clearInterval(timer);
        timer = null;
      }
    },
  };
}
