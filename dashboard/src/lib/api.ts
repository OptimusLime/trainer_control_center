/**
 * Fetch wrapper for trainer API with silent failure.
 *
 * In dev mode, Vite proxies /api/* → localhost:6060.
 * In production (served by trainer), we hit the same origin directly.
 *
 * On failure: returns null. The caller keeps existing DOM state.
 */

const TRAINER_URL = import.meta.env.DEV ? '/api' : '';

/** GET JSON from trainer. Returns null on any failure. */
export async function fetchJSON<T>(path: string): Promise<T | null> {
  try {
    const resp = await fetch(`${TRAINER_URL}${path}`, {
      signal: AbortSignal.timeout(8000),
    });
    if (!resp.ok) return null;
    return await resp.json() as T;
  } catch {
    return null;
  }
}

/** POST JSON to trainer. Returns null on any failure. */
export async function postJSON<T>(path: string, body?: unknown): Promise<T | null> {
  try {
    const resp = await fetch(`${TRAINER_URL}${path}`, {
      method: 'POST',
      headers: body ? { 'Content-Type': 'application/json' } : {},
      body: body ? JSON.stringify(body) : undefined,
      signal: AbortSignal.timeout(15000),
    });
    if (!resp.ok) return null;
    return await resp.json() as T;
  } catch {
    return null;
  }
}
