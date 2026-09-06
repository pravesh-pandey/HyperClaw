# External code harnesses — selectable ACP adapters

## Public provider boundary

`AgentConfig.provider` admits only the ACP provider, and
`KiroCrewConfig.create_provider_factory()` constructs `AcpProvider`. Harness
choice is a separate field, `agent.acp_backend`, and the public build now offers
every harness it knows: `acp_backends.BASELINE_SELECTABLE_BACKENDS` contains
`ACP_BACKEND_KIRO` (the empty string), `ACP_BACKEND_CLAUDE`,
`ACP_BACKEND_CODEX`, and `ACP_BACKEND_KAS` — i.e. all of
`ACP_BACKENDS_KNOWN`.
`test_baseline_ships_every_known_backend` pins that equality.

`DefaultProviderRegistry` therefore registers no extra backend: there is nothing
left in `ACP_BACKENDS_KNOWN` to add. `register_selectable_backend` stays because
the `ProviderRegistry` protocol declares the hook and an edition overrides it, but
it is **not** an extension point for a harness the core does not ship: it rejects
any id outside `ACP_BACKENDS_KNOWN`, and every id inside that set is now already
selectable. Adding a genuinely new harness therefore means widening
`ACP_BACKENDS_KNOWN` — a core edit — not just calling the register. What the hook
does buy an edition is reach once the id is known: it lands in the config gate, the
dashboard PATCH allowlist and `GET /api/config/schema` together
(`test_a_registered_backend_reaches_the_allowlist`,
`test_a_registered_backend_reaches_the_schema_endpoint`).

`acp_backends.resolve_selected_backend()` normalizes an `agent.acp_backend` value
this deployment cannot select to the Kiro harness. This boundary is load-bearing:
`AcpProvider` rejects unknown harnesses, so normalization prevents a persisted or
hand-edited value from becoming a startup failure.
`TestConfigRoundTrip.test_unselectable_values_degrade_to_the_default` exercises
that path, and `test_harness_parity.test_unselectable_backend_degrades_to_kiro`
asserts the outcome against the live registry rather than a hardcoded verdict —
which is why `claude` now *survives* that gate instead of degrading.

Selectable is not the same as usable, and it is not the same as permitted.
Whether a *deployment* may pick a registered harness is answered by the
`agent_backend` governance scope narrowing the registry
(`apply_selectable_denials`, floored at `GOVERNANCE_FLOOR_BACKEND` = kiro-cli).
Whether a *machine* can run it is answered by `agent_sdk.probe_backend`.

## The Codex harness

Codex is adapted through the public `@agentclientprotocol/codex-acp` package and
uses the same `AcpClient` lifecycle as Claude Code:

- `ACP_BACKEND_CODEX` selects `codex-acp`; `_resolve_codex_acp_bin()` searches
  the explicit `CODEX_AGENT_ACP_BIN`, the project/vendor install, mise, and the
  augmented PATH. The resolver returns a Node argv for a JavaScript entry point,
  so the adapter works in daemon environments whose shebang PATH is incomplete.
- The adapter package carries its compatible Codex dependency. `CODEX_PATH` is
  an optional adapter override; the backend probe therefore has one component,
  `codex-acp`, and reports the same install command the spawn path uses.
- Codex uses ACP protocol version `1`, skips Kiro's `session/set_mode`, and
  changes models through `session/set_config_option` with `configId: "model"`.
  Its `models.availableModels` payload can contain `model[effort]` display ids;
  Crew exposes the base ids from the adapter's model config option so a picker
  never sends a composite id back as a model selection.
- Reasoning effort is a separate `reasoning_effort` config option. Crew maps its
  provider-neutral `effort` control to that wire id, refreshes the advertised
  levels after a model change, and applies only values the live session reports.
  A cold Codex session exposes `Auto` until the adapter advertises concrete
  models; no model id is guessed for an account whose entitlements are unknown.
- Codex owns its session store and accepts the ACP `session/load` flow. It
  receives Crew's always-emitted managed MCP servers through the same ACP
  session parameters as Claude; pooled broker stubs remain independently
  injectable when configured.

Codex uses its own authentication and model entitlement state. The Crew backend
probe checks whether the adapter can be launched, not whether the operator has
completed Codex sign-in; an authentication failure is surfaced by the ACP session
as a runtime error rather than misreported as a missing executable.

## The Claude harness

`acp/client.py` owns the whole Claude spawn path, and it is a live path on a
plain public build:

- `AcpClient._is_claude` recognizes `ACP_BACKEND_CLAUDE`, and `AcpClient._spawn`
  takes the adapter branch for it.
- `_resolve_claude_acp_bin()` finds the `claude-agent-acp` Node entry script and
  returns `(argv, searched_path)`; the result is memoized process-wide in
  `_claude_acp_argv_cache`, so the search runs once and the "not found" message
  names exactly the directories that were searched.
- `_resolve_claude_code_executable()` finds the `claude` CLI and `_spawn` exports
  it as `CLAUDE_CODE_EXECUTABLE` when the caller has not set one. The adapter
  forwards it to `@anthropic-ai/claude-agent-sdk` as
  `pathToClaudeCodeExecutable`; without it the SDK fails `session/new` with
  "Claude native binary not found", because it does not search `PATH` for
  `claude` on its own.
- The adapter is a **public** npm package, `CLAUDE_ACP_NPM_PKG =
  "@agentclientprotocol/claude-agent-acp"`. Nothing on this path is
  edition-private.

### Models: the adapter owns its catalog

Claude is a member of both `ACP_BACKENDS_CONFIG_MODEL_WIRE_IDS` and
`ACP_BACKENDS_OWN_MODEL_CATALOG`, which settles three things that used to be
answered differently for it than for Codex:

- **The catalog is the `model` config option.** `claude-agent-acp` returns no
  `models` block in `session/new` — its `configOptions` entry with `id: "model"`
  carries both the offered values and the `currentValue` the session is running.
  A picker fed from `models.availableModels` therefore showed nothing but
  `Auto`, permanently, on a completely healthy session.
- **Its ids are the wire values, and they travel verbatim.** The option offers
  short aliases (`default`, `sonnet`, `opus`, `haiku`) alongside full ids
  (`claude-fable-5-1[1m]`), and `session/set_config_option` accepts exactly
  those. Translating through `model_registry.to_provider_id(…, "claude_code")`
  produces `global.anthropic.claude-opus-4-8[1m]`, which the adapter refuses
  with `-32603 Invalid value for config option model`. No call site between the
  picker, `_wire_model_id`, `acp_effective_model`, the cron normalizer, and the
  warm-pool post-claim switch translates an adapter model id. `auto` is likewise
  not a value it accepts, so returning to the default needs a session reset.
- **It never reads kiro-cli's agent spec.** `agent.model` defaults to `auto`, and
  collapsing that sentinel through `_resolve_agent_model` reads
  `~/.kiro/agents/kirocrew.json` — kiro-cli's file, holding whatever model the
  operator runs *kiro-cli* on. A GPT pin there was handed to `claude-agent-acp`
  on every cold start, which is where the reported `-32603` came from.

Because the advertised ids and the wire ids are the same namespace,
`model_is_unusable` is a valid pre-wire check here: an unservable pin is refused
against the account's real catalog, and the error names the models it does have
rather than surfacing a raw JSON-RPC fault. A refusal that still arrives from the
adapter (entitlement unknown at send time) degrades to the harness default rather
than ending the session.

Availability is therefore a property of the operator's machine, not of the build:
Claude Code needs **two** locally-installed binaries, the `claude-agent-acp`
adapter and the `claude` CLI. `agent_sdk.backend_install._probe_claude` reports
which of the two is absent (`COMPONENT_CLAUDE_ACP_ADAPTER`,
`COMPONENT_CLAUDE_CODE_CLI`) plus the command that installs the adapter, so a
half-install does not read as a total one. A probe that itself fails reads
`UNKNOWN`, never `MISSING`.

Both operator-facing surfaces read that one verdict:

- `kirocrew doctor` prints Claude Code as an optional backend — present, absent
  with the missing components named, or uncheckable. It is never a hard failure;
  kiro-cli is the floor.
- The dashboard's agent-backend control **hides** a harness the deployment may
  not select instead of dimming it, because under a managed policy there is
  nothing the reader can do about it and advertising a forbidden option is the
  opposite of what a restriction is for. The currently-selected value is always
  kept visible. A rendered-but-disabled row therefore always names something the
  user can act on: install a binary, or restart the gateway. Covered by
  `hides a backend the deployment may not select, rather than dimming it`,
  `keeps the selected backend visible even if it reads as unselectable` and
  `saves the Claude Code selection the shipped build offers`.

### What Crew gates on this harness

Start from what is NOT broken, because the difference is narrow and easy to overstate.

**By default, Claude asks and Crew decides.** In `default` permission mode with no
matching rule, every tool call reaches the SDK's `canUseTool` callback — which is
exactly what `claude-agent-acp` turns into ACP `session/request_permission`. That
arrives as a `permission_request` event and runs Crew's own approval path:
`hooks.on_tool_call`, its deny rules, its sensitive-path and write-protected-config
checks, and its SEL decision record. A Claude session is governed like any other on
that path.

**Managed Crew calls are forced through that path.** The ACP session metadata sets
`settingSources: []`, so project and user `.claude/settings.json` files cannot inject
pre-approved rules. It also disables Claude's native `Agent` and `Task` tools and
adds scoped `permissions.ask` rules for every managed Crew MCP name. The adapter
therefore asks ACP before a Crew MCP call, even when the cloned project carries its
own Claude settings. The external-adapter regression tests pin these options because
they are part of Crew's security boundary.

Other Claude-native tools still use the adapter's normal permission flow. Kiro CLI and
KAS remain governed by their own harness controls and do not read Claude settings.


### Crew MCP tools in external adapter sessions

The external adapters do not read `kirocrew.json` on their own. Kiro CLI still
receives managed servers from its rendered agent file; Claude and Codex receive
the always-emitted `kirocrew-core` and `kirocrew-cron` entries explicitly in both
the `session/new` and `session/load` ACP requests. The entries are built from
the same managed-server registry and emission gates as the Kiro spec, with no
`autoApprove` metadata. User-configured MCP servers remain an edition seam and
are not copied into an external adapter session.

Claude sessions also set `settingSources: []`, disable the native `Agent` and
`Task` tools, and install scoped `ask` rules for every managed Crew MCP tool.
This keeps a project `.claude/settings.json` from pre-approving Crew calls and
ensures `spawn_run` reaches ACP's permission request before the MCP server
executes it. The ACP child inherits the session key already used by Crew's MCP
identity resolver, so `spawn_run` resolves the originating chat and applies
`agent.role_backends.subagent` and `agent.role_models.subagent`.

### Companion-owned glue stays out of the core

The public client accepts edition-supplied Claude settings behavior without
owning it, hooked through `getattr` so the core is byte-identical when the hook
is absent:

- `_spawn` calls an optional `_write_claude_local_settings` on the **primary**
  spawn path, not only on the model-substitution retry — a session that skips it
  collapses to the 200K context default.
- `_spawn` merges `extra_env` into the child environment, which is how a
  caller-supplied `CLAUDE_CONFIG_DIR` reaches the adapter
  (`test_spawn_forwards_claude_config_dir_from_extra_env`).
- `AcpClient._reset_state` removes `<work_dir>/.claude/settings.local.json` for a
  Claude client. This is load-bearing because no caller retries teardown, so a
  session-scoped elevated permission setting must not outlive its client.

Standing rule, unchanged by Claude Code becoming selectable: `agent.provider`
stays single-valued and **no provider selector is re-added**. The harness switch
is `agent.acp_backend`, and it is gated in exactly one place
(`resolve_selected_backend`).

## Model registry

`src/kiro_crew/model_registry.json` is the shared model data source for
`model_registry.py` and `website/src/model_registry.json`.
`test_frontend_registry_matches_python_source` compares their parsed JSON, and
`website/src/providers/modelRegistry.ts` imports the frontend copy. The
per-entry `claude_code` provider IDs are registry mappings for the adapter's
advertised ids, not values accepted by `AgentConfig.provider`.

`model_registry._build_indices` indexes canonical keys, provider IDs, and
aliases. `from_provider_id` uses that index to recover a canonical key from an
advertised adapter ID. `TestModelRegistry.test_bare_advertised_ids_fold_to_canonical_key`
pins the bare-ID case.

`model_registry.available_models` and `display_list` sort default entries first
rather than trusting JSON object order. This is load-bearing because the adapter
uses the resulting allowlist when an automatic selection omits an explicit
model. `TestModelRegistry.test_fable_5_not_default` and
`TestModelRegistry.test_available_models_is_default_first` pin the default and
ordering behavior.
