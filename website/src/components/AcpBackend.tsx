import { Bot, Code2, Sparkles, Terminal } from 'lucide-react'

import { i18nT } from '../i18n/t'

/** ACP backend ids are wire values; the empty string is Kiro CLI, not absence. */
export const KIRO_BACKEND = ''

/**
 * Harnesses that report their own model catalog and effort ladder over ACP,
 * mirroring the backend's `ACP_BACKENDS_CONFIG_MODEL`.
 *
 * Membership, never `id !== KIRO_BACKEND`: a negative test silently grants the
 * next harness a capability nobody decided it has, and fails toward the
 * permissive answer so nothing goes red until an operator pays for it.
 */
export const ADAPTER_BACKENDS: ReadonlySet<string> = new Set(['claude', 'codex', 'opencode'])

/** True when this harness answers "which effort levels do you take" itself. */
export function reportsOwnEffortLadder(id: string | undefined): boolean {
  return id !== undefined && ADAPTER_BACKENDS.has(id)
}

export function acpBackendLabel(id: string): string {
  switch (id) {
    case KIRO_BACKEND:
      return i18nT('components.modelEffortDropdown.kiro_cli')
    case 'claude':
      return i18nT('components.modelEffortDropdown.claude_code')
    case 'codex':
      return i18nT('components.modelEffortDropdown.codex')
    case 'opencode':
      return i18nT('components.modelEffortDropdown.opencode')
    case 'kas':
      return i18nT('components.modelEffortDropdown.kas_kiro_agent')
    default:
      return id
  }
}

export function AcpBackendIcon({ id }: { id: string }) {
  const Icon = id === KIRO_BACKEND
    ? Terminal
    : id === 'claude'
      ? Sparkles
      : id === 'codex'
        ? Code2
        : Bot
  return <Icon className="lucide-inline" aria-hidden="true" />
}
