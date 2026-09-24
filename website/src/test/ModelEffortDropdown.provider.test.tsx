import React from 'react'
import { describe, expect, it, beforeEach, vi } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Provider } from 'react-redux'
import { configureStore } from '@reduxjs/toolkit'

import ModelEffortDropdown from '../components/ModelEffortDropdown'
import { api, type AcpBackendProbe } from '../api/client'
import chatReducer from '../store/chatSlice'
import dashboardReducer, { addSlotOptimistic } from '../store/dashboardSlice'
import notificationsReducer from '../store/notificationsSlice'

const SLOT = 'dashboard:provider-test'

const probe = (id: string, overrides: Partial<AcpBackendProbe> = {}): AcpBackendProbe => ({
  id,
  policy_id: id || 'kiro',
  selectable: true,
  installed: 'installed',
  missing_components: [],
  install_command: '',
  restart_required: false,
  ...overrides,
})

const baseProps = {
  anchorRect: { right: 400, top: 300 } as DOMRect,
  dropdownRef: React.createRef<HTMLDivElement>(),
  inputRef: React.createRef<HTMLInputElement>(),
  models: [{ name: 'auto', description: '' }],
  activeModel: 'auto',
  onSelectModel: vi.fn(),
  filter: '',
  setFilter: vi.fn(),
  onClose: vi.fn(),
  hasEffort: true,
  slot: SLOT,
  currentEffort: 'low',
  backendId: '',
  onListKeyDown: vi.fn(),
}

function setup(
  backends: AcpBackendProbe[],
  backendId: string | undefined = '',
  extra: Partial<React.ComponentProps<typeof ModelEffortDropdown>> = {},
) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const store = configureStore({
    reducer: { dashboard: dashboardReducer, chat: chatReducer, notifications: notificationsReducer },
  })
  store.dispatch(addSlotOptimistic({
    key: SLOT,
    title: 'Provider test',
    messages: 1,
    running: false,
    model: 'old-model',
    reasoning_effort: 'old-effort',
    acp_backend: backendId ?? null,
  }))
  vi.spyOn(api, 'acpBackends').mockResolvedValue({ backends })
  vi.spyOn(api, 'effortLevels').mockResolvedValue(['low', 'high'] as never)
  return { queryClient, store, render: () => render(
    <Provider store={store}>
      <QueryClientProvider client={queryClient}>
        <ModelEffortDropdown {...baseProps} backendId={backendId} {...extra} />
      </QueryClientProvider>
    </Provider>,
  ) }
}

beforeEach(() => {
  vi.restoreAllMocks()
})

/** Open the in-place harness list from the Provider row. */
async function openProviders() {
  fireEvent.click(await screen.findByRole('button', { name: /Provider/ }))
}

describe('ModelEffortDropdown provider row', () => {
  it('names the session harness and lists selectable providers on demand', async () => {
    const { render: renderPicker } = setup([
      probe('', { policy_id: 'kiro' }),
      probe('claude'),
      probe('codex'),
      probe('kas', { selectable: false }),
    ], 'codex')
    renderPicker()

    const row = await screen.findByRole('button', { name: /Provider/ })
    expect(row).toHaveTextContent('Codex')
    // The install probe is not free, so nothing is fetched until the list opens.
    expect(api.acpBackends).not.toHaveBeenCalled()
    await openProviders()

    expect(await screen.findByRole('option', { name: 'Kiro CLI' })).toBeInTheDocument()
    expect(screen.getByRole('option', { name: 'Claude Code' })).toBeInTheDocument()
    expect(screen.getByRole('option', { name: 'Codex' })).toHaveAttribute('aria-selected', 'true')
    expect(screen.queryByRole('option', { name: 'KAS (kiro-agent)' })).not.toBeInTheDocument()
  })

  it('hides the row when the harness is not known here', async () => {
    const { render: renderPicker } = setup([probe('')], '', { backendId: undefined })
    renderPicker()
    await screen.findByRole('dialog')
    expect(screen.queryByRole('button', { name: /Provider/ })).not.toBeInTheDocument()
  })

  it('shows but does not open the list when switching is not offered', async () => {
    const { render: renderPicker } = setup([probe('')], 'codex', { providerSwitchable: false })
    renderPicker()
    const row = await screen.findByRole('button', { name: /Provider/ })
    expect(row).toBeDisabled()
    fireEvent.click(row)
    expect(screen.queryByRole('option', { name: 'Kiro CLI' })).not.toBeInTheDocument()
  })

  it('disables a missing adapter and shows its install command', async () => {
    const { render: renderPicker } = setup([
      probe('', { policy_id: 'kiro' }),
      probe('codex', {
        installed: 'missing',
        missing_components: ['codex-acp'],
        install_command: 'npm install -g codex-acp',
      }),
    ])
    renderPicker()
    await openProviders()

    const row = await screen.findByRole('option', { name: 'Codex' })
    expect(row).toBeDisabled()
    expect(screen.getByText('Missing on this machine: codex-acp. Install with: npm install -g codex-acp')).toBeInTheDocument()
    expect(row).toHaveAttribute('aria-describedby', 'acp-backend-status-codex')
  })

  it('applies the backend, model, and effort returned by a successful switch', async () => {
    const { render: renderPicker, store } = setup([probe(''), probe('codex')])
    const switchMock = vi.spyOn(api, 'chatSlotBackend').mockResolvedValue({
      ok: true,
      backend: 'codex',
      model: 'gpt-6-sol',
      effort: 'high',
    })
    renderPicker()
    await openProviders()

    fireEvent.click(await screen.findByRole('option', { name: 'Codex' }))
    await waitFor(() => {
      const slot = store.getState().dashboard.slots.find(row => row.key === SLOT)
      expect(slot).toMatchObject({ acp_backend: 'codex', model: 'gpt-6-sol', reasoning_effort: 'high' })
    })
    expect(switchMock).toHaveBeenCalledWith(SLOT, 'codex')
  })

  it('Enter on a harness row switches the harness, not the highlighted model', async () => {
    const { render: renderPicker } = setup([probe(''), probe('codex')])
    const switchMock = vi.spyOn(api, 'chatSlotBackend').mockResolvedValue({
      ok: true, backend: 'codex', model: '', effort: '',
    })
    renderPicker()
    await openProviders()
    const row = await screen.findByRole('option', { name: 'Codex' })
    fireEvent.keyDown(row, { key: 'Enter' })
    expect(baseProps.onListKeyDown).not.toHaveBeenCalled()
    fireEvent.click(row)
    await waitFor(() => expect(switchMock).toHaveBeenCalled())
  })

  it('surfaces invalid_backend without changing the slot', async () => {
    const { render: renderPicker, store } = setup([probe(''), probe('codex')])
    vi.spyOn(api, 'chatSlotBackend').mockRejectedValue({
      body: JSON.stringify({ error: 'rejected', code: 'invalid_backend' }),
    })
    renderPicker()
    await openProviders()

    fireEvent.click(await screen.findByRole('option', { name: 'Codex' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('This provider is not selectable on this build.')
    expect(store.getState().dashboard.slots.find(row => row.key === SLOT)).toMatchObject({
      acp_backend: '', model: 'old-model', reasoning_effort: 'old-effort',
    })
  })

  it('surfaces a turn in flight as its own message', async () => {
    const { render: renderPicker } = setup([probe(''), probe('codex')])
    vi.spyOn(api, 'chatSlotBackend').mockRejectedValue({
      body: JSON.stringify({ error: 'busy', code: 'turn_in_flight' }),
    })
    renderPicker()
    await openProviders()

    fireEvent.click(await screen.findByRole('option', { name: 'Codex' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('Finish or stop the current turn before switching provider.')
  })

  it('surfaces backend_unavailable with the probe install command', async () => {
    const { render: renderPicker } = setup([
      probe(''),
      probe('codex', { install_command: 'npm install -g codex-acp' }),
    ])
    vi.spyOn(api, 'chatSlotBackend').mockRejectedValue({
      body: JSON.stringify({ error: 'missing binary', code: 'backend_unavailable' }),
    })
    renderPicker()
    await openProviders()

    fireEvent.click(await screen.findByRole('option', { name: 'Codex' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('Install with: npm install -g codex-acp')
  })
})
