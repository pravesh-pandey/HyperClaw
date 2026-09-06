import { useState, useRef, useLayoutEffect } from 'react'
import { motion } from 'framer-motion'
import { Trans } from 'react-i18next'
import { ChevronRight, ChevronLeft, Settings2, Pin, Check, Ban } from 'lucide-react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Input } from './ui'
import ModelDropdownList, { type ModelItem } from './ModelDropdownList'
import ReasoningEffortDropdown from './ReasoningEffortDropdown'
import { effortLabel } from './ChatInput'
import { api } from '../api/client'
import type { AcpBackendResponse } from '../api/client'
import { parseErrorCode } from '../utils/errorReport'
import { useAppDispatch } from '../store'
import { updateSlot } from '../store/dashboardSlice'
import { useProvider } from '../providers'
import { AcpBackendIcon, acpBackendLabel } from './AcpBackend'

import { i18nT } from '../i18n/t'

interface Props {
  anchorRect: DOMRect
  dropdownRef: React.Ref<HTMLDivElement>
  inputRef: React.Ref<HTMLInputElement>
  models: ModelItem[]
  activeModel: string
  onSelectModel: (name: string) => void
  filter: string
  setFilter: (v: string) => void
  onClose: () => void
  hasEffort: boolean
  slot: string | null
  currentEffort: string
  /** ACP backend id from the slot row; '' means Kiro CLI. */
  backendId?: string | null
  /** Configured default effort for new sessions. Shown in the footer when the
   *  slot carries no override, so the row reflects what a turn would run at. */
  defaultEffort?: string
  onListKeyDown: (e: React.KeyboardEvent) => void
  /** Deep-link to the Settings row that sets the GLOBAL fallback model — the
   *  tier that applies to agents pinning no model of their own. Optional so
   *  call sites that have no router (or don't want the link) are unaffected —
   *  the row is simply not rendered. */
  onSetDefault?: () => void
  /** Pin the currently-active model as this agent's own default, in place. Omit
   *  to hide the row (e.g. surfaces with no agent in scope). */
  onPinToAgent?: () => void
  /** KiroCrew agent the pin row acts on; shown in its label. */
  agentName?: string
  /** Model the pin row would WRITE, named in its label. This is the slot's real
   *  model, which is not always the one the composer displays: when a pin is
   *  withheld (the account cannot run it) every display surface reads `auto`
   *  while the write still carries the pin, deliberately, so a degraded model
   *  list cannot clobber a valid pin. Naming it here is what keeps that split
   *  honest — the row states what it persists instead of letting the user assume
   *  it matches the chip. */
  pinModelName?: string
  /** True when the pinned model is withheld, so the row explains that instead of
   *  offering a write. Setting an agent default to a model the account cannot run
   *  has no upside and surfaces later as an unexplained switch, so the action is
   *  withdrawn rather than merely warned about. */
  pinModelUnavailable?: boolean
  /** True when that agent already pins the active model, so the row reports the
   *  state instead of offering a no-op write. */
  pinnedToAgent?: boolean
}

type PickerPage = 'provider' | 'model' | 'effort'

const WIDTH = 340
const SPRING = { type: 'spring' as const, stiffness: 420, damping: 38 }

const statusId = (value: string) => `acp-backend-status-${value || 'kiro'}`

function errorCode(error: unknown): string {
  if (typeof error !== 'object' || error === null) return ''
  const body = (error as { body?: unknown }).body
  return typeof body === 'string' ? (parseErrorCode(body) ?? '') : ''
}

/** Model picker with a drill-in reasoning-effort panel. The model list scrolls;
 *  a fixed (non-scrolling) footer shows the current effort + a chevron. Clicking
 *  the footer springs the effort slider in from the right (the list pushes left);
 *  a back chevron returns. The popover height springs to the active page. The
 *  provider page precedes the model list and switches the slot's backend before
 *  returning to the model page. */
export default function ModelEffortDropdown({
  anchorRect, dropdownRef, inputRef, models, activeModel, onSelectModel,
  filter, setFilter, onClose, hasEffort, slot, currentEffort, backendId = null, onListKeyDown,
  onSetDefault, defaultEffort = '', onPinToAgent, agentName = '', pinModelName = '',
  pinModelUnavailable = false, pinnedToAgent = false,
}: Props) {
  const provider = useProvider()
  const qc = useQueryClient()
  const dispatch = useAppDispatch()
  const [page, setPage] = useState<PickerPage>(() => slot ? 'provider' : 'model')
  const [height, setHeight] = useState<number | undefined>(undefined)
  const [switchError, setSwitchError] = useState('')
  const modelPage = useRef<HTMLDivElement>(null)
  const providerPage = useRef<HTMLDivElement>(null)
  const effortPage = useRef<HTMLDivElement>(null)

  const probeQ = useQuery<AcpBackendResponse>({
    queryKey: ['acp-backends'],
    queryFn: () => api.acpBackends(),
    enabled: !!slot,
    retry: false,
    staleTime: 0,
  })
  const backends = (probeQ.data?.backends ?? []).filter(row => row.selectable)
  const activeBackendId = backendId ?? probeQ.data?.configured ?? ''

  const backendSwitch = useMutation({
    mutationFn: (backend: string) => api.chatSlotBackend(slot!, backend),
    onSuccess: async (result) => {
      // The provider switch re-resolves the namespace. The response is the only
      // authoritative source for all three values; the old model and effort may
      // not exist under the selected provider.
      dispatch(updateSlot({
        key: slot!,
        acp_backend: result.backend,
        model: result.model,
        reasoning_effort: result.effort,
      }))
      await Promise.all([
        // The model query is keyed by the slot's effective backend. The slot
        // update above may change that key, so match all backend variants for
        // this slot rather than refetching only the pre-switch `null` key.
        qc.refetchQueries({ queryKey: ['available-models', provider.id, slot], type: 'all' }),
        qc.refetchQueries({ queryKey: ['effort-levels', slot], type: 'all' }),
      ])
      setSwitchError('')
      setFilter('')
      setPage('model')
    },
    onError: (error, backend) => {
      const code = errorCode(error)
      const probe = backends.find(row => row.id === backend)
      if (code === 'invalid_backend') {
        setSwitchError(i18nT('components.modelEffortDropdown.invalid_backend'))
      } else if (code === 'backend_unavailable') {
        setSwitchError(probe?.install_command
          ? i18nT('components.modelEffortDropdown.backend_unavailable_with_command', { command: probe.install_command })
          : i18nT('components.modelEffortDropdown.backend_unavailable'))
      } else {
        setSwitchError(error instanceof Error && error.message
          ? error.message
          : i18nT('components.errorBoundary.something_went_wrong'))
      }
    },
  })

  // Size the popover to the active page (springs on toggle / list changes).
  useLayoutEffect(() => {
    const el = page === 'provider' ? providerPage.current : page === 'effort' ? effortPage.current : modelPage.current
    if (el) setHeight(el.offsetHeight)
  }, [page, models.length, filter, currentEffort, hasEffort, onSetDefault, onPinToAgent, agentName, pinModelName, pinModelUnavailable, pinnedToAgent, backends.length, switchError])

  // Right-align the dropdown to the button's right edge (clamped to viewport).
  const left = Math.max(8, Math.min(anchorRect.right - WIDTH, window.innerWidth - WIDTH - 8))
  const offset = page === 'provider' ? '0%' : page === 'model' ? '-33.3333%' : '-66.6667%'

  return (
    <div
      ref={dropdownRef}
      tabIndex={-1}
      onKeyDown={page === 'model' ? onListKeyDown : undefined}
      className="fixed z-[9999] bg-bg-elevated border border-border rounded-xl shadow-xl overflow-hidden animate-slide-up"
      style={{ width: WIDTH, bottom: window.innerHeight - anchorRect.top + 4, left }}
    >
      <motion.div animate={{ height }} transition={SPRING} style={{ height }} className="overflow-hidden">
        <motion.div className="flex w-[300%] items-start" animate={{ x: offset }} transition={SPRING}>
          <div ref={providerPage} className="w-1/3 flex flex-col p-1">
            <div className="px-2 py-1.5 text-[13px] text-muted border-b border-border">
              {i18nT('components.modelEffortDropdown.provider')}
            </div>
            {switchError && <div role="alert" className="px-2 py-1.5 text-[12px] text-warn">{switchError}</div>}
            <div role="listbox" aria-label={i18nT('components.modelEffortDropdown.provider_list')} className="overflow-y-auto max-h-[320px]">
              {probeQ.isLoading && <div className="px-3 py-3 text-[13px] text-muted">{i18nT('components.modelEffortDropdown.loading_providers')}</div>}
              {probeQ.isError && <div className="px-3 py-3 text-[13px] text-muted">{i18nT('components.modelEffortDropdown.could_not_load_providers')}</div>}
              {!probeQ.isLoading && !probeQ.isError && backends.map(row => {
                const missing = row.installed === 'missing'
                const restart = row.restart_required === true
                const disabled = missing || restart
                const externalDisclosure = row.id === 'claude' || row.id === 'opencode'
                  ? i18nT('components.modelEffortDropdown.external_adapter_disclosure')
                  : ''
                const hint = missing
                  ? row.install_command
                    ? i18nT('components.modelEffortDropdown.missing_components_with_command', { components: row.missing_components.join(', '), command: row.install_command })
                    : i18nT('components.modelEffortDropdown.missing_components', { components: row.missing_components.join(', ') })
                  : restart
                    ? i18nT('components.modelEffortDropdown.installed_restart_required')
                    : ''
                return (
                  <div key={row.id}>
                    <button
                      type="button"
                      role="option"
                      aria-selected={activeBackendId === row.id}
                      aria-describedby={hint ? statusId(row.id) : undefined}
                      disabled={disabled || backendSwitch.isPending}
                      onClick={() => { setSwitchError(''); backendSwitch.mutate(row.id) }}
                      className={`w-full text-left px-2.5 py-2 flex items-center gap-2 rounded-md cursor-pointer transition-colors border-none bg-transparent ${disabled ? 'opacity-50 cursor-not-allowed' : 'hover:bg-bg-hover'}`}
                    >
                      <AcpBackendIcon id={row.id} />
                      <span className="text-[13px] text-text">{acpBackendLabel(row.id)}</span>
                      {activeBackendId === row.id && <Check className="lucide-inline ml-auto text-accent" />}
                    </button>
                    {hint && <div id={statusId(row.id)} className="px-3 pb-1 text-[11px] leading-relaxed text-warn">{hint}</div>}
                    {externalDisclosure && <div className="px-3 pb-1 text-[11px] leading-relaxed text-muted">{externalDisclosure}</div>}
                  </div>
                )
              })}
            </div>
          </div>

          {/* Page 1 — model list + non-scrolling effort footer */}
          <div ref={modelPage} className="w-1/3 flex flex-col p-1">
            {slot && (
              <button
                type="button"
                onClick={() => { setSwitchError(''); setPage('provider') }}
                className="shrink-0 flex items-center gap-1 px-2 py-1.5 text-[12px] text-muted hover:text-text border-b border-border bg-transparent border-x-0 border-t-0 cursor-pointer self-stretch"
              >
                <ChevronLeft className="lucide-inline" />
                {i18nT('components.modelEffortDropdown.providers')}
              </button>
            )}
            <div className="px-1.5 pt-1.5 pb-1">
              <Input
                ref={inputRef}
                type="text"
                aria-label={i18nT('components.modelEffortDropdown.filter_models')}
                placeholder={i18nT('components.modelEffortDropdown.type_to_filter')}
                value={filter}
                onChange={e => setFilter(e.target.value)}
                className="w-full px-2 py-1 text-[13px]"
              />
            </div>
            <div role="listbox" aria-label={i18nT('components.modelEffortDropdown.model_list')} className="overflow-y-auto max-h-[280px]">
              <ModelDropdownList models={models} activeModel={activeModel} onSelect={onSelectModel} />
            </div>
            {hasEffort && slot && (
              <button
                type="button"
                onClick={() => setPage('effort')}
                className="shrink-0 mt-0.5 border-t border-border rounded-b-lg flex items-center justify-between gap-2 px-3 py-2.5 text-[13px] cursor-pointer bg-transparent border-x-0 border-b-0 hover:bg-bg-hover transition-colors"
              >
                <span className="text-muted">{i18nT('components.modelEffortDropdown.reasoning')}</span>
                <span className="flex items-center gap-1 text-text font-medium">
                  {effortLabel(currentEffort || defaultEffort)}
                  <ChevronRight className="lucide-inline text-muted" />
                </span>
              </button>
            )}
            {onPinToAgent && agentName && (
              <button
                type="button"
                onClick={pinnedToAgent || pinModelUnavailable ? undefined : onPinToAgent}
                disabled={pinnedToAgent || pinModelUnavailable}
                aria-pressed={pinnedToAgent}
                className="shrink-0 border-t border-border flex items-center justify-between gap-2 px-3 py-2 text-[12px] cursor-pointer bg-transparent border-x-0 border-b-0 text-muted hover:text-text hover:bg-bg-hover transition-colors disabled:cursor-default disabled:hover:bg-transparent"
              >
                {/* Wraps rather than truncates. The label's whole job is to name
                    WHICH agent and WHICH model the write targets, and both
                    identifiers sit at the ends — an ellipsis eats exactly the
                    part that carries the meaning. English fits on one line, but
                    the disambiguating word costs 8-14 characters in the Romance
                    locales ("modelo predeterminado", "modèle par défaut"), so
                    those overflow 340px. The popover already springs its height
                    to the measured page, so a second line is free. min-w-0 lets
                    the flex item shrink below its content; break-words is the
                    backstop for a model id longer than one line. */}
                <span className="min-w-0 text-left break-words">
                  {pinModelUnavailable
                    ? <Trans i18nKey="components.modelEffortDropdown.pin_model_unavailable" components={{ model: <span className="font-mono">{pinModelName}</span> }} />
                    : pinnedToAgent
                    ? <Trans i18nKey="components.modelEffortDropdown.default_for_agent" components={{ agent: <span className="font-mono">{agentName}</span> }} />
                    : <Trans i18nKey="components.modelEffortDropdown.set_default_for_agent" components={{ model: <span className="font-mono">{pinModelName}</span>, agent: <span className="font-mono">{agentName}</span> }} />}
                </span>
                {pinnedToAgent ? <Check className="lucide-inline text-accent" /> : pinModelUnavailable ? <Ban className="lucide-inline" /> : <Pin className="lucide-inline" />}
              </button>
            )}
            {onSetDefault && (
              <button
                type="button"
                onClick={onSetDefault}
                className="shrink-0 border-t border-border rounded-b-lg flex items-center justify-between gap-2 px-3 py-2 text-[12px] cursor-pointer bg-transparent border-x-0 border-b-0 text-muted hover:text-text hover:bg-bg-hover transition-colors"
              >
                <span>{i18nT('components.modelEffortDropdown.set_default_for_new_sessions')}</span>
                <Settings2 className="lucide-inline" />
              </button>
            )}
          </div>

          {/* Page 2 — reasoning effort slider, reached via the footer */}
          <div ref={effortPage} className="w-1/3 flex flex-col p-1">
            <button
              type="button"
              onClick={() => setPage('model')}
              className="shrink-0 flex items-center gap-1 px-2 py-1.5 text-[12px] text-muted hover:text-text border-b border-border bg-transparent border-x-0 border-t-0 cursor-pointer self-stretch"
            >
              <ChevronLeft className="lucide-inline" />
              {i18nT('components.modelEffortDropdown.models')}
            </button>
            {slot && <ReasoningEffortDropdown slot={slot} currentEffort={currentEffort} defaultEffort={defaultEffort} onClose={onClose} embedded />}
          </div>
        </motion.div>
      </motion.div>
    </div>
  )
}
