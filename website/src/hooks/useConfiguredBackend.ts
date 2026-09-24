/**
 * The ACP harness new sessions start on (`agent.acp_backend`), and the harness a
 * picker should ask about for one session.
 *
 * Read from the same `['kirocrewConfig']` query Settings already holds, rather
 * than `/api/acp-backends`, which probes every harness install to answer.
 */
import { useQuery } from '@tanstack/react-query'

import { api } from '../api/client'

interface ConfigWithBackend {
  agent?: { acp_backend?: string }
}

/** `undefined` while the config is loading; `''` is Kiro CLI, a real value. */
export function useConfiguredBackend(): string | undefined {
  const { data } = useQuery<ConfigWithBackend>({
    queryKey: ['kirocrewConfig'],
    queryFn: () => api.kirocrewConfig(),
    staleTime: 30_000,
  })
  const backend = data?.agent?.acp_backend
  return typeof backend === 'string' ? backend : undefined
}

/**
 * The `backend` a model / effort query should name for a session, or
 * `undefined` to ask for the default list.
 *
 * Only a session that picked a harness OTHER than the configured one names it.
 * Everything else shares the default list's cache entry, so a picker on the
 * default harness costs no extra catalog read per session.
 */
export function pickerBackend(
  slotBackend: string | null | undefined,
  configured: string | undefined,
): string | undefined {
  if (slotBackend === null || slotBackend === undefined) return undefined
  if (configured !== undefined && slotBackend === configured) return undefined
  return slotBackend
}

/** The harness a session actually runs on, for display: its pick, else the default. */
export function effectiveBackend(
  slotBackend: string | null | undefined,
  configured: string | undefined,
): string | undefined {
  return slotBackend ?? configured
}
