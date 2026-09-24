import { useState } from 'react'
import { Trans } from 'react-i18next'
import { Settings2, Pin, Check, Ban, ChevronRight, ChevronDown, LoaderCircle } from 'lucide-react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Btn, Input } from './ui'
import ErrorNotice from './ErrorNotice'
import ModelDropdownList, { type ModelItem } from './ModelDropdownList'
import { AcpBackendIcon, acpBackendLabel } from './AcpBackend'

import { api } from '../api/client'
import { useImeGuard } from '../hooks/useImeGuard'
import { i18nT } from '../i18n/t'
import { useProvider } from '../providers'
import { useAppDispatch } from '../store'
import { updateSlot } from '../store/dashboardSlice'
import { parseErrorCode } from '../utils/errorReport'

interface Props {
  anchorRect: DOMRect
  dropdownRef: React.Ref<HTMLDivElement>
  inputRef: React.Ref<HTMLInputElement>
  models: ModelItem[]
  activeModel: string
  onSelectModel: (name: string) => void
  /** True while the model list's source is still being fetched (a remote-bound
   *  session's peer capability read in flight or re-polling). Forwarded to the
   *  list so an empty picker shows a loading row rather than "No matches". */
  modelsLoading?: boolean
  /** True when the model list's source READ failed outright (the peer
   *  capability request errored, not a per-field miss inside a good reply).
   *  Renders an `ErrorNotice` row: without it the failure would wear the
   *  empty list's "No matches" clothes and read as "this crew has no models". */
  modelsFailed?: boolean
  /** In-place retry for that failure; omit to hide the button. */
  onRetryModels?: () => void
  /** True while that retry is in flight. The failed state persists until the
   *  refetch settles, so the Retry button disables and spins instead of
   *  sitting apparently dead under the unchanged error row. */
  retryingModels?: boolean
  filter: string
  setFilter: (v: string) => void
  modelVisibilityError?: boolean
  onRetryModelVisibility?: () => void
  /** Chat slot the Provider row switches; omit (or null) to hide that row. */
  slot?: string | null
  /** The ACP harness this session runs on (`''` is Kiro CLI). Omit to hide the
   *  Provider row -- a surface with no session, or one whose harness this
   *  machine cannot name (a remote-bound session runs on the peer's). */
  backendId?: string
  /** False shows the harness without offering a switch. */
  providerSwitchable?: boolean
  onListKeyDown: (e: React.KeyboardEvent) => void
  /** Deep-link to the Settings row that sets the GLOBAL fallback model — the
   *  tier that applies to agents pinning no model of their own. Optional so
   *  call sites that have no router (or don't want the link) are unaffected —
   *  the row is simply not rendered. */
  onSetDefault?: () => void
  /** First-use shortcut to the persistent visible-model setting. */
  onManageModels?: () => void
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

const WIDTH = 340

const statusId = (value: string) => `acp-backend-status-${value || 'kiro'}`

function errorCode(error: unknown): string {
  if (typeof error !== 'object' || error === null) return ''
  const body = (error as { body?: unknown }).body
  return typeof body === 'string' ? (parseErrorCode(body) ?? '') : ''
}

/** Searchable model picker. Effort lives in its own composer control.
 *  A Provider row above the list names the session's ACP harness and, when
 *  switchable, expands in place into the harness list. A harness switch resets
 *  the session (there is no live switch across harnesses), so it lives here,
 *  beside the model it scopes, rather than as a separate control. */
export default function ModelEffortDropdown({
  anchorRect, dropdownRef, inputRef, models, activeModel, onSelectModel,
  filter, setFilter, slot, backendId, providerSwitchable = true,
  onListKeyDown, onSetDefault, onManageModels,
  modelVisibilityError = false, onRetryModelVisibility,
  onPinToAgent, agentName = '', pinModelName = '',
  pinModelUnavailable = false, pinnedToAgent = false, modelsLoading = false,
  modelsFailed = false, onRetryModels, retryingModels = false,
}: Props) {
  const ime = useImeGuard()
  const provider = useProvider()
  const qc = useQueryClient()
  const dispatch = useAppDispatch()
  const [showProviders, setShowProviders] = useState(false)
  const [switchError, setSwitchError] = useState('')
  const showProviderRow = !!slot && backendId !== undefined
  // Probing every harness install is not free, so it runs only while the
  // harness list is open.
  const probeQ = useQuery({
    queryKey: ['acpBackends'],
    queryFn: () => api.acpBackends(),
    enabled: showProviderRow && showProviders,
    retry: false,
    staleTime: 0,
  })
  const backends = (probeQ.data?.backends ?? []).filter(row => row.selectable)
  const backendSwitch = useMutation({
    mutationFn: (backend: string) => api.chatSlotBackend(slot!, backend),
    onSuccess: async result => {
      // The response is the only authoritative source for all three values:
      // the switch clears a model the new harness cannot name.
      dispatch(updateSlot({
        key: slot!,
        acp_backend: result.backend,
        model: result.model,
        reasoning_effort: result.effort,
      }))
      // A switch also moves the configured default for new sessions.
      await Promise.all([
        qc.invalidateQueries({ queryKey: ['kirocrewConfig'] }),
        qc.invalidateQueries({ queryKey: ['available-models', provider.id] }),
        qc.invalidateQueries({ queryKey: ['effort-levels', slot] }),
        qc.invalidateQueries({ queryKey: ['slot-selection-capabilities', slot] }),
      ])
      setSwitchError('')
      setFilter('')
      setShowProviders(false)
    },
    onError: (error, backend) => {
      const code = errorCode(error)
      const row = backends.find(candidate => candidate.id === backend)
      if (code === 'invalid_backend') {
        setSwitchError(i18nT('components.modelEffortDropdown.invalid_backend'))
      } else if (code === 'backend_unavailable') {
        setSwitchError(row?.install_command
          ? i18nT('components.modelEffortDropdown.backend_unavailable_with_command', { command: row.install_command })
          : i18nT('components.modelEffortDropdown.backend_unavailable'))
      } else if (code === 'turn_in_flight') {
        setSwitchError(i18nT('components.modelEffortDropdown.backend_switch_busy'))
      } else {
        setSwitchError(error instanceof Error && error.message
          ? error.message
          : i18nT('components.errorBoundary.something_went_wrong'))
      }
    },
  })
  // Right-align the dropdown to the button's right edge (clamped to viewport).
  const width = Math.min(WIDTH, window.innerWidth - 16)
  const left = Math.max(8, Math.min(anchorRect.right - width, window.innerWidth - width - 8))
  const maxHeight = Math.max(0, anchorRect.top - 12)

  return (
    // The dialog delegates list navigation from its filter and option rows.
    // eslint-disable-next-line jsx-a11y/no-noninteractive-element-interactions
    <div
      ref={dropdownRef}
      role="dialog"
      aria-label={i18nT('components.modelEffortDropdown.model_list')}
      tabIndex={-1}
      onKeyDown={event => {
        const target = event.target as HTMLElement
        // Tab advances into the optional management action when present.
        if (event.key === 'Tab') {
          if (!ime.claimKey(event)) return
          if (!event.shiftKey && target.tagName === 'INPUT') {
            const nextControl = event.currentTarget.querySelector<HTMLElement>(
              '[data-model-picker-manage]',
            )
            if (nextControl) {
              event.preventDefault()
              event.stopPropagation()
              nextControl.focus()
            }
          }
          return
        }
        if (target.closest('[data-model-picker-manage]')) {
          if (event.key === 'ArrowUp') {
            const options = Array.from(
              event.currentTarget.querySelectorAll<HTMLElement>('[role="option"]'),
            )
            const lastOption = options[options.length - 1]
            if (lastOption) {
              event.preventDefault()
              event.stopPropagation()
              lastOption.focus()
            }
          }
          return
        }
        // Native buttons handle their own keys: Enter on a harness row must
        // switch the harness, not pick the highlighted model.
        if (target.closest('[data-model-picker-provider]')) return
        if (event.key === 'ArrowDown' && target.getAttribute('role') === 'option') {
          const options = Array.from(event.currentTarget.querySelectorAll<HTMLElement>('[role="option"]'))
          if (target === options[options.length - 1]) {
            const nextControl = event.currentTarget.querySelector<HTMLElement>(
              '[data-model-picker-manage]',
            )
            if (nextControl) {
              event.preventDefault()
              event.stopPropagation()
              nextControl.focus()
              return
            }
          }
        }
        onListKeyDown(event)
      }}
      className="fixed z-[9999] flex flex-col bg-bg-elevated border border-border rounded-xl shadow-xl overflow-hidden animate-slide-up"
      style={{ width, maxHeight, bottom: window.innerHeight - anchorRect.top + 4, left }}
    >
          <div className="flex min-h-0 flex-1 flex-col p-1">
            {showProviderRow && (
              <Btn
                type="button"
                data-model-picker-provider
                onClick={providerSwitchable ? () => { setSwitchError(''); setShowProviders(open => !open) } : undefined}
                disabled={!providerSwitchable}
                aria-expanded={providerSwitchable ? showProviders : undefined}
                className="w-full shrink-0 justify-between rounded-lg border-0 px-2.5 py-1.5 text-[12px] disabled:hover:bg-transparent"
              >
                <span className="text-muted">{i18nT('components.modelEffortDropdown.provider')}</span>
                <span className="flex items-center gap-1 text-text">
                  <AcpBackendIcon id={backendId} />
                  {acpBackendLabel(backendId)}
                  {providerSwitchable && <ChevronDown className={`lucide-inline text-muted transition-transform ${showProviders ? 'rotate-180' : ''}`} />}
                </span>
              </Btn>
            )}
            {showProviderRow && showProviders && (
              <div
                role="listbox"
                aria-label={i18nT('components.modelEffortDropdown.provider_list')}
                className="shrink-0 max-h-[240px] overflow-y-auto border-y border-border py-0.5"
              >
                {switchError && <div role="alert" className="px-2.5 py-1.5 text-[12px] text-warn">{switchError}</div>}
                {probeQ.isLoading && <div className="px-2.5 py-2 text-[12px] text-muted">{i18nT('components.modelEffortDropdown.loading_providers')}</div>}
                {probeQ.isError && <div className="px-2.5 py-2 text-[12px] text-muted">{i18nT('components.modelEffortDropdown.could_not_load_providers')}</div>}
                {backends.map(row => {
                  const missing = row.installed === 'missing'
                  const restart = row.restart_required === true
                  const hint = missing
                    ? row.install_command
                      ? i18nT('components.modelEffortDropdown.missing_components_with_command', { components: row.missing_components.join(', '), command: row.install_command })
                      : i18nT('components.modelEffortDropdown.missing_components', { components: row.missing_components.join(', ') })
                    : restart
                      ? i18nT('components.modelEffortDropdown.installed_restart_required')
                      : ''
                  return (
                    <div key={row.id || 'kiro'}>
                      <button
                        type="button"
                        role="option"
                        data-model-picker-provider
                        aria-selected={backendId === row.id}
                        aria-describedby={hint ? statusId(row.id) : undefined}
                        disabled={missing || restart || backendSwitch.isPending}
                        onClick={() => { setSwitchError(''); backendSwitch.mutate(row.id) }}
                        className={`w-full text-left px-2.5 py-1.5 flex items-center gap-2 rounded-md border-none bg-transparent text-[13px] text-text transition-colors ${missing || restart ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer hover:bg-bg-hover'}`}
                      >
                        <AcpBackendIcon id={row.id} />
                        {acpBackendLabel(row.id)}
                        {backendSwitch.isPending && backendSwitch.variables === row.id
                          ? <LoaderCircle className="lucide-inline ml-auto animate-spin" aria-hidden />
                          : backendId === row.id && <Check className="lucide-inline ml-auto text-accent" />}
                      </button>
                      {hint && <div id={statusId(row.id)} className="px-3 pb-1 text-[11px] leading-relaxed text-warn">{hint}</div>}
                    </div>
                  )
                })}
              </div>
            )}
            <div className="shrink-0 px-1.5 pt-1.5 pb-1">
              <Input
                ref={inputRef}
                type="text"
                aria-label={i18nT('components.modelEffortDropdown.filter_models')}
                placeholder={i18nT('components.modelEffortDropdown.type_to_filter')}
                value={filter}
                {...ime.bindComposition<HTMLInputElement>()}
                onChange={e => setFilter(e.target.value)}
                className="w-full px-2 py-1 text-[13px]"
              />
            </div>
            {modelVisibilityError && (
              <div className="flex shrink-0 items-center gap-2 px-1.5 py-1">
                {/* No hand-off: the chat composer may contain an unsent draft.
                    Retrying in place preserves it. */}
                <ErrorNotice
                  className="min-w-0 flex-1"
                  variant="inline"
                  message={i18nT('pages.settings.chatPanel.failed_to_load_dashboard_config')}
                />
                {onRetryModelVisibility && (
                  <Btn type="button" className="shrink-0" onClick={onRetryModelVisibility}>
                    {i18nT('pages.settings.chatPanel.retry')}
                  </Btn>
                )}
              </div>
            )}
            {modelsFailed && (
              <div className="flex shrink-0 items-center gap-2 px-1.5 py-1">
                {/* No hand-off: the chat composer may contain an unsent draft.
                    Retrying in place preserves it. */}
                <ErrorNotice
                  className="min-w-0 flex-1"
                  variant="inline"
                  message={i18nT('components.modelEffortDropdown.models_failed')}
                />
                {onRetryModels && (
                  <Btn type="button" className="shrink-0" onClick={onRetryModels} disabled={retryingModels}>
                    {retryingModels && <LoaderCircle className="lucide-inline animate-spin" aria-hidden />}
                    {i18nT('pages.settings.chatPanel.retry')}
                  </Btn>
                )}
              </div>
            )}
            <div role="listbox" aria-label={i18nT('components.modelEffortDropdown.model_list')} className="min-h-0 flex-1 max-h-[240px] overflow-y-auto">
              <ModelDropdownList models={models} activeModel={activeModel} onSelect={onSelectModel} loading={modelsLoading} failed={modelsFailed} />
            </div>
            {onManageModels && <ManageModelsFooter onManage={onManageModels} />}
            {onPinToAgent && agentName && (
              <Btn
                type="button"
                onClick={pinnedToAgent || pinModelUnavailable ? undefined : onPinToAgent}
                disabled={pinnedToAgent || pinModelUnavailable}
                aria-pressed={pinnedToAgent}
                className="w-full shrink-0 justify-between rounded-none border-x-0 border-b-0 px-3 py-2 text-[12px] text-muted disabled:hover:bg-transparent"
              >
                {/* Wraps rather than truncates. The label's whole job is to name
                    WHICH agent and WHICH model the write targets, and both
                    identifiers sit at the ends — an ellipsis eats exactly the
                    part that carries the meaning. English fits on one line, but
                    the disambiguating word costs 8-14 characters in the Romance
                    locales ("modelo predeterminado", "modèle par défaut"), so
                    those overflow 340px. The popover grows with its content, so
                    a second line is free. min-w-0 lets
                    the flex item shrink below its content; break-words is the
                    backstop for a model id longer than one line. */}
                <span className="min-w-0 text-left break-words">
                  {pinModelUnavailable
                    ? <Trans
                        i18nKey="components.modelEffortDropdown.pin_model_unavailable"
                        components={{ model: <span className="font-mono">{pinModelName}</span> }}
                      />
                    : pinnedToAgent
                    ? <Trans
                        i18nKey="components.modelEffortDropdown.default_for_agent"
                        components={{ agent: <span className="font-mono">{agentName}</span> }}
                      />
                    : <Trans
                        i18nKey="components.modelEffortDropdown.set_default_for_agent"
                        components={{
                          model: <span className="font-mono">{pinModelName}</span>,
                          agent: <span className="font-mono">{agentName}</span>,
                        }}
                      />}
                </span>
                {pinnedToAgent ? <Check className="lucide-inline text-accent" /> : pinModelUnavailable ? <Ban className="lucide-inline" /> : <Pin className="lucide-inline" />}
              </Btn>
            )}
            {onSetDefault && (
              <Btn
                type="button"
                onClick={onSetDefault}
                className="w-full shrink-0 justify-between rounded-b-lg rounded-t-none border-x-0 border-b-0 px-3 py-2 text-[12px] text-muted"
              >
                <span>{i18nT('components.modelEffortDropdown.set_default_for_new_sessions')}</span>
                <Settings2 className="lucide-inline" />
              </Btn>
            )}
          </div>
    </div>
  )
}

/** First-use shortcut; the permanent visibility control lives in Settings. */
export function ManageModelsFooter({ onManage }: { onManage: () => void }) {
  return (
    <Btn
      type="button"
      onClick={onManage}
      data-model-picker-manage
      data-option
      aria-label={i18nT('components.modelEffortDropdown.manage_visible_models')}
      className="w-full shrink-0 justify-between rounded-none border-x-0 border-b-0 px-3 py-2 text-[12px] text-muted"
    >
      <span className="flex items-center gap-2 text-left">
        <Settings2 className="lucide-inline shrink-0" />
        {i18nT('components.modelEffortDropdown.manage_visible_models')}
      </span>
      <span className="flex items-center gap-0.5 text-right text-[10px]">
        {i18nT('components.modelEffortDropdown.manage_in_settings')}
        <ChevronRight className="lucide-inline shrink-0" />
      </span>
    </Btn>
  )
}
