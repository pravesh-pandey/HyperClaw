import { Bot, Code2, Sparkles, Terminal } from 'lucide-react'

import { ACP_BACKEND_KIRO } from '../api/acpBackend'
import { i18nT } from '../i18n/t'

/**
 * Harnesses that report their own effort ladder over ACP, mirroring the
 * gateway's `ACP_BACKENDS_EFFORT_VIA_CONFIG_OPTION`.
 *
 * Membership, never `id !== ACP_BACKEND_KIRO`: a negative test silently grants
 * the next harness a capability nobody decided it has, and fails toward the
 * permissive answer so nothing goes red until an operator pays for it.
 */
const EFFORT_LADDER_BACKENDS: ReadonlySet<string> = new Set(['claude', 'codex', 'deepseek', 'pi'])

/** True when this harness answers "which effort levels do you take" itself. */
export function reportsOwnEffortLadder(id: string | undefined): boolean {
  return id !== undefined && EFFORT_LADDER_BACKENDS.has(id)
}

/** Display name for a harness id. An id this build does not name falls back to
 *  itself, so a newly registered harness reads plainly rather than as another. */
export function acpBackendLabel(id: string): string {
  switch (id) {
    case ACP_BACKEND_KIRO:
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
  const Icon = id === ACP_BACKEND_KIRO
    ? Terminal
    : id === 'claude'
      ? Sparkles
      : id === 'codex'
        ? Code2
        : Bot
  return <Icon className="lucide-inline" aria-hidden="true" />
}
