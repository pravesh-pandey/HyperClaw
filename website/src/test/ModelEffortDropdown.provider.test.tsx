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

function setup(backends: AcpBackendProbe[], configured = '', backendId: string | null = '') {
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
    acp_backend: backendId,
  }))
  vi.spyOn(api, 'acpBackends').mockResolvedValue({ backends, configured })
  vi.spyOn(api, 'effortLevels').mockResolvedValue(['low', 'high'] as never)
  return { queryClient, store, render: () => render(
    <Provider store={store}>
      <QueryClientProvider client={queryClient}>
        <ModelEffortDropdown {...baseProps} backendId={backendId} />
      </QueryClientProvider>
    </Provider>,
  ) }
}

beforeEach(() => {
  vi.restoreAllMocks()
})

describe('ModelEffortDropdown provider level', () => {
  it('renders selectable providers and filters out unselectable rows', async () => {
    const { render: renderPicker } = setup([
      probe('', { policy_id: 'kiro' }),
      probe('claude'),
      probe('codex'),
      probe('kas', { selectable: false }),
    ])
    renderPicker()

    expect(await screen.findByRole('option', { name: 'Kiro CLI' })).toBeInTheDocument()
    expect(screen.getByRole('option', { name: 'Claude Code' })).toBeInTheDocument()
    expect(screen.getByRole('option', { name: 'Codex' })).toBeInTheDocument()
    expect(screen.queryByRole('option', { name: 'KAS (kiro-agent)' })).not.toBeInTheDocument()
  })

  it('uses the configured provider when the slot has no backend pick', async () => {
    const { render: renderPicker } = setup([probe(''), probe('codex')], 'codex', null)
    renderPicker()

    expect(await screen.findByRole('option', { name: 'Codex' })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('option', { name: 'Kiro CLI' })).toHaveAttribute('aria-selected', 'false')
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
      model: 'gpt-5.6-sol',
      effort: 'high',
    })
    renderPicker()

    fireEvent.click(await screen.findByRole('option', { name: 'Codex' }))
    await waitFor(() => {
      const slot = store.getState().dashboard.slots.find(row => row.key === SLOT)
      expect(slot).toMatchObject({ acp_backend: 'codex', model: 'gpt-5.6-sol', reasoning_effort: 'high' })
    })
    expect(switchMock).toHaveBeenCalledWith(SLOT, 'codex')
  })

  it('surfaces invalid_backend without changing the slot', async () => {
    const { render: renderPicker, store } = setup([probe(''), probe('codex')])
    vi.spyOn(api, 'chatSlotBackend').mockRejectedValue({
      body: JSON.stringify({ error: 'rejected', code: 'invalid_backend' }),
    })
    renderPicker()

    fireEvent.click(await screen.findByRole('option', { name: 'Codex' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('This provider is not selectable on this build.')
    expect(store.getState().dashboard.slots.find(row => row.key === SLOT)).toMatchObject({
      acp_backend: '', model: 'old-model', reasoning_effort: 'old-effort',
    })
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

    fireEvent.click(await screen.findByRole('option', { name: 'Codex' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('Install with: npm install -g codex-acp')
  })
})
