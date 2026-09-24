/**
 * The reasoning-effort ladder a given ACP harness actually accepts.
 *
 * Effort levels are per-harness, not universal: Codex offers low→max with no
 * `default` rung, Claude offers `default` plus the same five, and kiro-cli
 * answers the process-global list. A static frontend constant therefore offers
 * levels one adapter will refuse and hides ones another supports, so the
 * Settings controls read the live ladder the same way the model pickers read
 * the live catalog.
 *
 * `backend` is the harness id (`''` is Kiro CLI, a real selection — so it is
 * passed through and only `undefined` means "don't ask about a harness").
 */
import { useQuery } from '@tanstack/react-query'

import { api } from '../api/client'

/** Shown while the ladder is loading or when a harness reports none. Matches
 *  `lib/effort`'s EFFORT_LEVELS minus the '' sentinel, which the Settings
 *  selects prepend themselves as "Model default". */
export const FALLBACK_EFFORT_LEVELS = ['low', 'medium', 'high', 'xhigh', 'max']

export interface EffortLevelsOptions {
  backend?: string
  enabled?: boolean
}

/**
 * Returns the harness's ordered levels, or the fallback while unknown.
 *
 * Never empty: an empty array would collapse the control to nothing, which
 * reads as "this harness has no effort setting" when the truth is "we have not
 * been told yet". The fallback is the union every known harness supports.
 */
export function useEffortLevels({ backend, enabled }: EffortLevelsOptions = {}): string[] {
  const { data } = useQuery({
    queryKey: ['effort-levels', 'backend', backend ?? null],
    queryFn: () =>
      api.effortLevels(undefined, backend).then(rows =>
        Array.isArray(rows) && rows.length > 0 ? rows : FALLBACK_EFFORT_LEVELS
      ),
    ...(enabled === undefined ? {} : { enabled }),
  })
  return data ?? FALLBACK_EFFORT_LEVELS
}
