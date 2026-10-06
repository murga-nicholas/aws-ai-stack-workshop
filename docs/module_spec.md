# Module specification — read this before writing any Python

This is the **grounding contract** for the AWS AI stack workshop. Codex implements
every Python file against it; Claude writes the deck content, the data files and the
docs against it. If the code and this file disagree, one of them is a bug: fix the
code or propose a change to this file — never let them drift silently.

Verified API facts in §9 were measured from installed wheels and live read-only calls
on **2026-10-05**. Do not guess an API. If a fact is not in §9, read the installed
source, then add it to §9 and to `docs/research.md` §9.

---

## 0. Ownership and how the pieces fit

| Area | Owner | Files |
|---|---|---|
| Demo package, CLI, contracts, runtime | Codex | `src/awsai_demo/**` |
| Tests (100% statements and branches) | Codex | `tests/**` |
| Project config | Codex | `pyproject.toml`, `uv.lock`, `.env.example`, `.gitignore`, `.gitattributes`, `docker-compose.localstack.yml` |
| Deck renderer, validators, facts collector, llms-full generator, price snapshot tool | Codex | `deck/**/*.py`, `scripts/*.py` |
| Deck content (declarative) | Claude | `deck/content/*.yaml`, `deck/speaker_notes.md`, `deck/assets/**` |
| Data content | Claude | `data/aws_ai_stack_notes.md`, `data/lineage.yaml`, `data/components.yaml` |
| Generated data (committed) | Codex's tool, reviewed by Claude | `data/pricing_snapshot.json`, `deck/facts.json`, `llms-full.txt`, `aws_ai_stack.pptx` |
| Docs | Claude | `README.md`, `CONTRIBUTING.md`, `docs/**`, `llms.txt` |

Rule of thumb: **facts and words are Claude's, code is Codex's.** A fact lives in one
data file; code loads and validates it; the deck, the glossary and the CLI all read
the same record.

---

## 1. House style (enforced, not advisory)

- Python **3.12+** (`requires-python = ">=3.12,<3.14"`); `from __future__ import
  annotations` in every module.
- **Ruff at 79 columns**, docstring lines at **72**, the same 26 rule families as the
  siblings (`A ANN ARG B C4 D DTZ E EM F I ICN LOG N PERF PIE PLC PLE PLW PTH RET RSE
  S SIM T10 TC TRY UP W RUF`), ignoring `ANN401` and `PLC0415` only.
- **Google-style docstrings** on every public function, class and module.
- `mypy --strict` over `src/awsai_demo` and `deck/`. `Any` only at an SDK boundary,
  converted at once to a local `TypedDict`, `Protocol` or dataclass.
- Optional SDKs are imported **inside functions**, so a missing package is a reported
  result (`sdk_missing`), never an `ImportError` at start-up.
- **Console output is ASCII** (Windows code pages). Rich is fine; em dashes are not.
- Every demo module opens with a docstring stating: the technology, its lane, its
  lifecycle record ids (§6), and the exact `uv run awsai-demo <command>` lines.
- Clear for a junior: one public `run_<name>_demo()` per module, small named helpers,
  no metaprogramming, no clever decorators of our own, comments explain *why*.

---

## 2. The result contract (`awsai_demo.contracts`)

### 2.1 Envelope

`result(...)` returns a `DemoResult` TypedDict:

| Field | Meaning |
|---|---|
| `schema_version` | `1` |
| `demo` | CLI name, e.g. `"agentcore-harness"` |
| `technology` | Product name shown on the slide |
| `lane` | `"models" \| "agents" \| "data" \| "platform" \| "operations" \| "local"` |
| `lifecycle_refs` | list of `data/lineage.yaml` record ids (§6); at least one |
| `requested_execution` | `"offline" \| "emulator" \| "live"` — what the user asked for |
| `mode` | how the result was actually produced (§2.2) |
| `status` | `"ok" \| "paused" \| "blocked" \| "error"` |
| `headline` | one ASCII sentence for the console and the slide |
| `operations` | list of `OperationOutcome` (§2.3), in call order |
| `evidence` | `Evidence` (§2.4) |
| `data` | technology-specific payload (allowlisted, redacted) or `None` |
| `error` | `{"code", "message"}` or `None` (§2.5) |
| `next_steps` | concrete missing prerequisites |
| `children` | only when `mode == "batch"` |

### 2.2 `mode` — pick exactly one, the true one

| `mode` | Use when |
|---|---|
| `local_execution` | Real local computation or real protocol execution on loopback (Strands with a local tool, MCP over stdio, A2A on 127.0.0.1, `BedrockAgentCoreApp` on localhost, Cedar via `cedarpy`, OpenTelemetry in-memory). Nothing simulated. |
| `local_contract` | Real SDK/application code with an **explicitly synthetic** external response (botocore `Stubber`, a scripted Strands model, an `httpx.MockTransport`). Requires `fixture_id`. |
| `local_emulator` | **LocalStack** executed the AWS API on loopback. Not AWS. Requires `emulator` evidence (§2.4). |
| `live_model` | An AWS (or OpenAI) model was actually invoked and answered. |
| `live_identity` | Credentials were actually verified by AWS STS. |
| `live_service` | A non-model AWS service actually answered (including a real 403). |
| `attempt_failed` | A call was dispatched and **no response arrived** (connection reset, timeout, DNS). Nothing can be claimed about the service. |
| `not_run` | A prerequisite stopped it before any call. Use `missing_configuration(...)` or `not_run(...)`. |
| `batch` | Container only; children carry their own modes. Never an operation mode. |

**Labelling rule for mixed demos.** Scripted inference (`ScriptedModel`, Stubber,
MockTransport) is always `local_contract`, even when a real framework or protocol
runs around it. A demo that combines a real local protocol with scripted inference
(Runtime on localhost, A2A, evaluation, observability) records **separate operation
outcomes**: the protocol calls as `local_execution`, the model calls as
`local_contract`; the result's top-level `mode` is the *weakest* of its operations in
the order `local_contract < local_emulator < local_execution` for local lanes. An
emulator answering STS is `local_emulator`, never `live_identity`.

`contracts.result()` **raises `ValueError`** when evidence contradicts the mode:
a live mode must have `network_attempted=True`, a non-loopback endpoint and
`response_received=True` on at least one operation; a non-live mode must not claim
`aws_executed=True`; `local_contract` must name a `fixture_id`; `local_emulator` must
name an emulator endpoint on loopback; `attempt_failed` must have an operation with
`response_received=False`. Pure local computations (`cost`, `lineage`) and `batch`
containers may have **zero** operations. **No silent fallback**: a live run never
replays a fixture. If nothing in the requested lane runs, the result is `not_run`; if
a live read (such as `CountTokens`) runs and the billable call is refused, the result
keeps that live mode with status `blocked`.

**Live provenance.** AWS error responses, including a rejected inference call, are
`live_service`; `live_model` requires a model answer; `live_identity` requires a
successful STS verification. A mixed live result takes `live_model` if any operation
established it, otherwise `live_service`, otherwise `live_identity`. Local
operations (the SQLite sink, local retrieval) never erase live provenance. Failed
operations stay in `operations`; `status` is derived separately. `attempt_failed`
is the result mode only when calls were dispatched and **no** operation established
execution.

**Deliberate skips do not degrade status.** Blocked `not_run` operations with
`contract_only`, `create_not_allowed`, `invoke_not_allowed`,
`not_supported_by_emulator` or `not_supported_by_model` do not affect aggregate
status. `missing_configuration`
is neutral only when the operation is marked `optional=True`. A result with no
execution in the requested lane remains blocked; actual local provenance is
preserved; a top-level error imposes at least `blocked`.

**Batch aggregation.** A batch's `status` is the worst child status in the order
`ok < paused < blocked < error`; `all` exits `0` when no child is `error`, `1`
otherwise, and prints a per-child summary. `blocked` (a real 403, a missing token) is
a successful demonstration, not a failure of `all`.

### 2.3 Operation outcomes

Every external call (and every loopback protocol call) is recorded:

```python
class OperationOutcome(TypedDict):
    service: str             # "bedrock-runtime", "sts", "mcp", "localstack", ...
    operation: str           # "Converse", "GetCallerIdentity", "tools/call", ...
    phase: Literal["setup", "main", "teardown"]
    execution_target: Literal["fixture", "local", "emulator", "aws", "openai"]
    mode: Mode               # per-operation mode (never "batch")
    status: Status
    effect: Effect           # §3.5 vocabulary
    transport: Literal["none", "loopback", "aws", "external"]  # stdio = "none"
    endpoint_url: str | None # sanitized: scheme + host + path, no query
    response_received: bool
    request_validated: bool  # params validated against the botocore model
    fixture_id: str | None
    http_status: int | None
    error_code: str | None   # AWS error code or our code (§2.5)
    duration_ms: int | None
    usage: Usage | None      # tokens; None when unknown - never invent 0
    reserved_usd: float | None  # budget reserved before dispatch (§3.4)
    optional: bool           # default False; marks optional missing configuration
```

### 2.4 Evidence (allowlisted fields only)

`sdk_invoked`, `network_attempted`, `aws_executed`, `provider`, `region`,
`fixture_id`, `requested_model`, `observed_model`, `credential_source`
(`"none" | "profile" | "env" | "csv_file" | "default_chain"`; never a value),
`packages` (name → version for the SDKs the demo touched, including `botocore`
whose service model validated the request), `emulator` (`{"endpoint": "http://
localhost:4566", "image": tag, "digest": sha256, "edition": ..., "plan": ...}` or
`None`), `estimated_cost_usd` (float or `None`) and `cost_basis`
(`"estimated" | "measured" | None`).

### 2.5 Error codes (stable vocabulary)

`missing_configuration`, `sdk_missing`, `authorization_denied`,
`maintenance_mode` (only when AWS's documented maintenance diagnostic matches, §8.8),
`entitlement_missing` (LocalStack plan), `not_supported_by_emulator`,
`emulator_unavailable`, `contract_only`, `budget_exceeded`, `create_not_allowed`,
`invoke_not_allowed`, `throttled`, `timeout`, `validation_failed`,
`model_unavailable` (the model is unexpectedly absent), `cleanup_incomplete`,
`subscription_required` (normalized `SubscriptionRequiredException`; an answered
service refusal remains `blocked`), `model_access_required` (the provider's
one-time, per-account access step, such as Anthropic's use-case form, is missing),
`not_supported_by_model` (the catalog shows the model and the API is a documented
gap for it), `max_tokens_reached` (the model answered and hit
`max_output_tokens`; operation mode `live_model`, status `blocked`, partial `usage`
kept; neither `model_unavailable` nor a neutral skip). Map known AWS errors
(`AccessDeniedException`, `ThrottlingException`, `ValidationException`,
`ResourceNotFoundException`, `ServiceQuotaExceededException`) to these; **let unknown
exceptions raise**. Never catch broadly and manufacture success.

### 2.6 Redaction

- Evidence and `data` are **allowlisted**; raw SDK responses and headers never enter a
  result.
- `redact()` runs recursively at **every serialization boundary**: result builder,
  CLI printer (pretty and JSON), facts collector, subprocess output capture, log
  formatter. It masks 12-digit account ids, ARNs' account segments, access key ids
  (`AKIA…`, `ASIA…`), secret-looking 40-char base64 strings, bearer tokens,
  `Authorization` headers and session tokens.
- ARNs are redacted in their account **and** resource segments
  (`arn:aws:bedrock-agentcore:us-east-1:<account>:harness/<redacted>`); a stable,
  non-secret short hash may replace a resource id when the demo must show that two
  outputs refer to the same thing. Fingerprints and proposal-version hashes are not
  secrets and are preserved.
- **Private vs public.** The cleanup manifest (§3.6) is private state under
  `.awsai_runs/` and keeps exact ids; it is never printed, never copied into facts,
  and is git-ignored. Everything printed or written to `deck/` is public evidence.
- **Unknown exceptions**: library code re-raises them unchanged; the CLI entry point
  and every subprocess entry point catch at the top, print only a sanitized
  one-line diagnostic (exception class + redacted message), and exit non-zero.
- The CLI isolates execution, envelope-validation and serialization failures per
  demo as sanitized error results, preserving validated operation evidence where
  available, and continues the batch.
- Tests prove redaction on success **and** failure paths, including exception text
  from botocore, for every channel above.

---

## 3. Execution lanes, credentials and the network

### 3.1 The three lanes

`--execution offline` (default) · `--execution emulator` · `--execution live`.

| Lane | Network | Credentials | What may answer |
|---|---|---|---|
| offline | none, except registered loopback endpoints (§3.3) | never loads real ones; fixed dummy `AKIDEXAMPLE`-style values for clients | Stubber fixtures, scripted models, local protocols |
| emulator | loopback to LocalStack only | fixed `test`/`test`, region `us-east-1` | LocalStack |
| live | AWS endpoints (and OpenAI only for the explicit cross-vendor path) | §3.2 | AWS |

### 3.2 Credential selection (live only)

Precedence, first match wins, **no switching after selection**:

1. `--profile NAME` (CLI flag)
2. `AWS_PROFILE`
3. `AWS_CREDS_FILE_PATH` — an IAM console CSV. Exactly one data row; header names
   normalised case-insensitively (`Access key ID`, `Secret access key`, optional
   `Session token`); UTF-8 BOM tolerated. Loaded into an in-memory `boto3.Session`;
   never copied, printed, logged or written.
4. The standard boto3 chain.

Region: `--region` → `AWS_REGION` → `AWS_DEFAULT_REGION` → `us-east-1`.
Once selected, authentication or authorisation failures **fail**, they do not fall
through to the next source. `doctor` (no flag) inspects configuration only and makes
no credential network call; `doctor --probe` calls STS and one read per service.

### 3.3 Network rules

- Offline egress is **enforced**, not hoped for, during normal runs *and* tests,
  including subprocesses: a process-wide guard (installed by the CLI for the offline
  and emulator lanes, and by a pytest fixture) allows only registered endpoints.
- Stubber-only demos open **no sockets** at all.
- Registered loopback endpoints: `agentcore-runtime` (local app on a free
  127.0.0.1 port), `a2a` (127.0.0.1 ephemeral port), LocalStack
  (`localhost:4566`, emulator lane only). MCP over stdio opens no socket and is
  recorded as `transport: "none"`. Redirects off loopback are refused.
- SDK callbacks, Strands callback handlers and telemetry exporters obey the same
  network and redaction rules.

### 3.4 Execution policy (application-enforced budget)

`ExecutionPolicy` (frozen dataclass) with defaults:

| Limit | Default |
|---|---|
| `max_input_tokens` | 2,000 per model call (see the counting rule below) |
| `max_output_tokens` | 512 per model call (256 made Claude Haiku 4.5 stop mid-answer in the multi-step demos; the USD caps are unchanged) |
| `max_model_calls` | 6 per command |
| `max_agent_iterations` | 4 per agent; `max_model_calls` also applies across the command |
| retries | **one** retry: botocore `Config(retries={"total_max_attempts": 2, "mode": "standard"})`; Strands `ModelRetryStrategy` and the OpenAI client (`max_retries=1`) configured explicitly so retry layers never multiply |
| `max_wall_seconds` | 120 per command |
| `max_estimated_usd` | 0.05 per command, 0.50 for `all --execution live` |
| `cleanup_allowance_usd` | 0.01, reserved separately so teardown is never refused |

**Counting input tokens.** Live input limits and reservations use, in order:
(1) exact counting with `bedrock-runtime.CountTokens` for the complete request when
the model supports it; (2) otherwise a documented upper bound: the UTF-8 byte length
of the complete serialized request (system prompt, full history, tool schemas,
structured-output schema) plus 256 tokens of protocol overhead, labelled
`upper_bound`; this holds because the Bedrock text tokenizers we use are byte-level
or byte-fallback and emit at most one token per input byte. If a model is not known
to satisfy (1) or (2) (a list kept in `policy.py` with its evidence), the live call is
refused. Character heuristics may appear only as labelled estimates in prose.

**Reservation before dispatch.** Every billable live operation reserves a
conservative upper bound **before** it is sent: model calls from complete input +
`max_output_tokens`, times (1 + retries); service operations from unit price x
units (characters, text units, events, vectors, session-seconds) taken from
`data/pricing_snapshot.json`. Reservations are shared through one locked ledger per
run (`.awsai_runs/<run-id>/budget.json`) across parent and child processes,
concurrent agents and `all`. If a reservation would exceed the limit, the operation
is refused with `budget_exceeded`. **An operation with no price row, or one whose
downstream charges we cannot see (`invoke_named` into user code, Payments, batch,
async, customization, ingestion), cannot be reserved and is therefore not executed
live, whatever flags are passed.** Synthetic contracts for them are still allowed
offline.

**Harness live execution is read-only.** CPU, memory and token estimates do not bound session-lifetime charges, automatic CloudWatch telemetry or internally constructed prompts ([AWS harness cost guidance](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/harness-operations.html)). `CreateHarness` and `InvokeHarness` therefore return `not_run` / `budget_exceeded` live, including with `--allow-create`; `ListHarnesses` runs live.

Cost notices go to **stderr** so `--format json` stays valid JSON. Wording
everywhere: "application-enforced limits, not an AWS billing ceiling".

Tested default live model: `global.anthropic.claude-haiku-4-5-20251001-v1:0`, counted with the foundation id
`anthropic.claude-haiku-4-5-20251001-v1:0` (no Nova model supports `CountTokens`,
section 9.6) (overridable with
`--model`). Call it the **tested default**, not the cheapest.

### 3.5 Effects, phases and the flags that permit them

Each operation has exactly one `effect`:

| `effect` | Meaning | Live only when |
|---|---|---|
| `none` | no external effect (fixture, pure local, loopback protocol) | - |
| `read` | List/Get/Describe; no state change; free or negligible | `--execution live` |
| `infer` | billable model or AI-service inference, reserved per 3.4 | `--execution live` |
| `write` | creates or changes state (create resource, put item, create event) | `--allow-create` |
| `delete` | removes state we created (teardown) | follows the matching `write` |
| `session_start` / `session_stop` | billable session (Code Interpreter) | `--allow-create` |
| `export` | sends spans to an OTLP collector; v1 supports a registered loopback collector only | `--trace otlp` |
| `invoke_named` | invokes a resource the user named (a deployed runtime, a gateway, a knowledge base); downstream effects invisible to us | **never** in this workshop (3.4); request shapes are shown offline |

Emulator-lane writes are real writes **into LocalStack**: they carry `effect: write`
and `execution_target: emulator`; LocalStack state is disposable.

`phase` is `setup`, `main` or `teardown`. A lane that a row marks "not executed"
returns `not_run` with `code="contract_only"` when requested.

Our code **never** creates IAM roles, deploys containers, starts ingestion jobs,
calls `CreateAgent` or `InvokeInlineAgent`, or creates anything not listed as a live
`write` in section 7.

### 3.6 Resource manifest and cleanup

- **Intent first.** Before a `write`/`session_start` request is *sent*, append an
  intent record to `.awsai_runs/<run-id>/manifest.json` (private, git-ignored):
  run id, account fingerprint (truncated sha256 of the account id), region, service,
  operation, exact intended name, client token, tags. After the response, update it
  with the exact id/ARN and status; on an ambiguous failure record `unknown` so
  cleanup reconciles it.
- **Service-specific names.** Names obey each service's pattern (9.3): harness
  `[a-zA-Z][a-zA-Z0-9_]{0,39}` -> `awsai_<12hex>_h`; memory
  `[a-zA-Z][a-zA-Z0-9_]{0,47}` -> `awsai_<12hex>_mem`; guardrail
  `[0-9a-zA-Z-_]{1,50}` -> `awsai-<12hex>-guard`; S3 vector bucket (lowercase,
  hyphens) -> `awsai-<12hex>-vec`. `<12hex>` is the run id. The manifest records the
  naming scheme; cleanup validates it.
- **Idempotent creates.** Every API with `clientToken`/`clientRequestToken` gets one
  derived from run id + suffix. Where none exists, reconcile by exact name before any
  retry.
- **Teardown** runs in `finally`, reverse order, children first (index before vector
  bucket; harness endpoint before harness), with retries inside the cleanup
  allowance; outcomes are written to the manifest.
- **Ownership checks.** Cleanup requires either verified ownership of a **tagged**
  resource (exact id from the manifest **and** tag `run-id` **and** name scheme
  **and** account fingerprint **and** region) or one of these documented exceptions
  for manifest-owned, untaggable things:
  - *Child resources* (memory events, guardrail versions, vector indexes) may be
    created only beneath a parent created and ownership-verified in the same run;
    the manifest records exact parent and child ids. They are removed by deleting the
    owned parent (or by exact-id deletion), never by search.
  - *Sessions* (Code Interpreter): before `StartCodeInterpreterSession` is sent, the
    manifest stores a deterministic run-specific session `name` and the
    `clientToken`, and both are sent. After an ambiguous response, reconcile with
    `ListCodeInterpreterSessions` / `GetCodeInterpreterSession` using the exact
    name, interpreter identifier, account and region; persist the recovered session
    id before stopping it. Never match by prefix alone. If ownership cannot be
    established, leave the manifest entry `unresolved` and report
    `cleanup_incomplete`; never claim success.
- `awsai-demo cleanup [--run-id ID] [--execute]` is **dry-run by default**; outside
  the exceptions above it never deletes untagged resources, and it never deletes
  shared or user-supplied resources (for example `AWSAI_GUARDRAIL_ID`). A create path whose cleanup is
  not fully specified here is **not executed live**, even with `--allow-create`.

---

## 4. The shape of a module

### 4.1 Ports and adapters

```python
class HarnessPort(Protocol):
    """The external calls this demo makes, and nothing else."""

    async def create(self, spec: HarnessSpec) -> HarnessHandle: ...
    async def invoke(self, handle: HarnessHandle, prompt: str) -> HarnessTurn: ...
    async def delete(self, handle: HarnessHandle) -> None: ...


async def run_agentcore_harness_demo(
    *,
    execution: Execution = "offline",
    port: HarnessPort | None = None,
    policy: ExecutionPolicy | None = None,
    settings: Settings | None = None,
) -> DemoResult:
    ...
```

- One public `run_<name>_demo()` per module; tests inject a port. **Nothing in our
  code is monkey-patched.** `monkeypatch` is allowed only for the standard library,
  environment variables, the working directory and third-party SDK entry points
  (for example a clock, package metadata or an SDK client class); a test that needs
  to replace one of our own functions means that function should be an injected port.
- Adapters per lane: `Stub…Port` (offline, botocore Stubber or scripted),
  `Emulator…Port` (LocalStack), `Live…Port`. Construct the live adapter only after
  configuration checks pass.
- Missing optional SDK ⇒ `sdk_missing` for that lane; offline paths that need no SDK
  still run.
- Missing token/usage counts are `None`, never `0`.

### 4.2 Stubber rules (`awsai_demo.stubs`)

- Real `boto3` client, fixed region, dummy credentials, `Stubber(client)`.
- Every `add_response` supplies **`expected_params`**; every run ends with
  `assert_no_pending_responses()`.
- Timestamp-shaped response fields in fixtures contain Python `datetime` values,
  exactly as botocore returns them; SDK timestamps are normalized recursively to ISO
  8601 strings at the response boundary.
- Evidence records `fixture_id`, `botocore` version, service and operation, and
  `request_validated=True`.
- **Event-stream outputs** (`InvokeHarness`, `InvokeAgent`, `InvokeCodeInterpreter`,
  `ConverseStream`, `InvokeAgentRuntime` streaming) cannot be stubbed faithfully. For
  those, validate the request with
  `botocore.validate.validate_parameters(params, operation_model.input_shape)` and
  replay a named fixture stream through the port; record `request_validated=True`,
  `fixture_id`.
- Slides and docs describe it exactly: *"validates the request against the service
  model; proves nothing about IAM, provider semantics, streaming or service
  behaviour."*

### 4.3 The offline model (`awsai_demo.offline`)

`ScriptedModel(strands.models.model.Model)` implements the four abstract methods
(`get_config`, `update_config`, `stream`, `structured_output`) and replays a named
script of Converse-shaped stream events (text deltas, `toolUse`, `stop_reason`,
usage). The **real** `strands.Agent` loop, tool execution, hooks, interrupts and
session managers run around it. Mode `local_contract`, fixture id
`strands-script-<name>-v1`.

### 4.4 Providers (`awsai_demo.providers`)

The whole model switch (all `BedrockModel` arguments are keyword-only):

- `offline` -> `ScriptedModel`.
- `bedrock` (live) -> `BedrockModel(model_id=..., boto_session=<selected session>,
  boto_client_config=<one-retry Config>, max_tokens=policy.max_output_tokens)`.
- `emulator` -> `BedrockModel(model_id="ollama.<name>",
  endpoint_url="http://localhost:4566", boto_session=<dummy test/test session>,
  streaming=False, use_native_token_count=False, max_tokens=...)`. `streaming=False`
  because LocalStack has no `ConverseStream`; the explicit dummy session keeps the
  real credential chain out of the emulator lane.
- `openai` -> `OpenAIModel` with the user's `OPENAI_API_KEY`, `max_retries=1`
  (cross-vendor path only, selected with `--provider openai`).

Nothing downstream knows the vendor.

---

## 5. Shared scenario and the centrepiece

### 5.1 Scenario (`awsai_demo.scenario`)

Identical business task **and numbers** to the siblings, so the three workshops can
be compared line by line: cost a **six-week customer-support pilot** against a
**25,000 USD** budget and refuse to accept it without a named human approval.

| Input | Value (illustrative, as in the siblings) |
|---|---|
| `PILOT_WEEKS` | 6 |
| `PILOT_BUDGET_USD` | 25,000 |
| `RATE_CARD_USD_PER_WEEK` | solution-architect 2,400 · python-engineer 1,900 · qa-engineer 1,500 |
| default team (FTE) | solution-architect 0.5 · python-engineer 1.0 |
| `PLATFORM_COST_USD_PER_WEEK` | 180 (model, hosting, telemetry) |

Expected `price_pilot()` result: staffing 7,200 + 11,400 = **18,600**; platform
**1,080**; total **19,680**; within budget; headroom **5,320**. Tests assert these
exact numbers.

Deterministic functions: `BRIEF` (AWS wording: "AWS resources and permissions are not
provisioned yet"), `price_pilot(weeks=…, team=…, budget=…) -> PilotCost` (rejects
non-positive weeks and unknown roles), `proposal_is_acceptable(cost) -> (bool,
reasons)`, `proposal_version(cost) -> str` (sha256 of the canonical JSON),
`action_key(cost) -> str` = `"accept-proposal:" + proposal_version(cost)`. Every demo
that needs numbers uses these.

### 5.2 `decision` — one decision, three implementations

| Lane | Implementation | Lanes | Honest label |
|---|---|---|---|
| A | **Agents Classic**: action group `price_pilot` + return of control for `accept_proposal` (`InvokeAgent` with `returnControl` → `sessionState.returnControlInvocationResults`) | offline (event-stream fixture) | contract — never live (§3.5) |
| B | **AgentCore harness**: `CreateHarness` with an inline function tool `accept_proposal`; the `InvokeHarness` stream ends with a `toolUse` content block and `messageStop.stopReason = "tool_use"`; the client resumes by sending a `toolResult` content block keyed by that `toolUseId` | offline (fixture stream); live is read-only (3.4) | contract |
| C | **Strands, code-defined** (the principal executable proof): `accept_proposal` tool calls `tool_context.interrupt("approval", reason=…)`; session persisted with `FileSessionManager` under `.awsai_checkpoints/`; a **separate OS process** (`python -m awsai_demo.resume_worker`) loads the session and answers the interrupt | offline (scripted model), emulator (DynamoDB sink), live (Bedrock model) | real framework execution |

The business effect is written to an **idempotent sink**, deduplicated on the
**business action key** (`scenario.action_key(cost)`), independent of who approved:
approval id, approver and timestamp are **audit data** stored with the effect, never
part of the uniqueness key.

- offline **and live** lanes: SQLite (`.awsai_checkpoints/ledger.sqlite3`, local
  even when the model is live) — one transaction inserts the effect row with
  `action_key TEXT PRIMARY KEY`; a second insert for the same key is a no-op that
  returns the original row;
- emulator lane: DynamoDB on LocalStack — `PutItem` with
  `ConditionExpression="attribute_not_exists(action_key)"`.

Approval is bound to `proposal_version`. Checkpoint writes are coordinated too: the
resume worker takes an exclusive file lock on the session directory before loading,
so two resumes cannot interleave checkpoint writes.

CLI:

```text
awsai-demo decision                                         # A and B as labelled; C pauses and prints its run id (status "paused")
awsai-demo decision --simulate-approval                     # offline only: pause, simulated approval in a separate process, one replay
awsai-demo decision --resume RUN_ID --approve --approver NAME
awsai-demo decision --resume RUN_ID --deny --approver NAME
awsai-demo decision --replay RUN_ID                         # must find the effect and do nothing
```

**Default execution and `all` never synthesize a human approval.** The deck's
evidence comes from `--simulate-approval`, which is refused outside the offline lane
and records `approval_source="simulated"`. `--approve/--deny` record
`approval_source="cli"` and the caller-supplied `approver` label, which is **not** an
authenticated identity and the output says so. The same approval gate (bound to
`proposal_version`) runs before the business effect in every lane. `--resume`
validates the persisted settings (execution lane, model, policy) and the proposal
version before answering the interrupt.

Required tests: approve, deny, **stale approval** (proposal changed → rejected),
**two concurrent resumes** (exactly one effect), **crash after the effect commits but
before the checkpoint is saved** (replay finds the effect and does not repeat it),
plain replay. Claim "at most once" **only for the sink effect**.

Output `data`: per lane what ran; process ids of the pausing and resuming processes;
files read by the resume process; the computed checks; and a **responsibility
matrix** with rows `model choice`, `orchestration loop`, `tool authorisation`,
`approval pause`, `state persistence`, `resume after crash`, `deduplication`,
`tracing`, `cost limits` and columns lane A/B/C, each cell one of `application`,
`framework`, `aws_service`, `not_available`, with a one-line justification.

---

## 6. Data files Claude curates (Codex loads and validates)

### 6.1 `data/lineage.yaml` — lifecycle and lineage records

```yaml
schema_version: 1
verified: 2026-10-05
records:
  - id: agents-classic                 # kebab-case, unique
    name: Amazon Bedrock Agents Classic
    former_names: [Agents for Amazon Bedrock]
    entity_type: service               # service|feature|model|api|package|endpoint|product
    scope: CreateAgent and InvokeInlineAgent for accounts without prior usage
    official_status: "Maintenance (no longer open to new customers)"
    normalized_status: maintenance     # active|preview|maintenance|sunset|legacy|end_of_life|renamed|moved
    launched: 2023-11-28
    announced: 2026-06-30
    effective: 2026-07-30
    end_of_support: null
    successors: [agentcore-harness, agentcore-runtime]
    relation: successor                # rename|successor|migration|merged|moved|none
    sources: [https://docs.aws.amazon.com/bedrock/latest/userguide/agents-classic-maintenance-mode.html]
    note: Existing agents keep working; model catalog frozen at 2026-07-30.
```

Validation (Codex): unique ids; every `successors` id exists; dates are ISO; every
record has ≥1 `https://` source; `normalized_status` in the enum; a test asserts that
every demo's `lifecycle_refs` exist.

### 6.2 `data/components.yaml` — the component → code → slide matrix

```yaml
schema_version: 1
components:
  - id: agentcore-harness
    name: AgentCore harness
    layer: agent-platform
    lineage: [agentcore-harness]
    demos: [agentcore-harness, decision]
    slides: [agentcore.harness]
    covered: true
```

Tests: every registry demo appears in ≥1 component; every `covered: true` component
names ≥1 demo **and** ≥1 slide id that exists in `deck/content/*.yaml`; every lineage
id exists. This is how "every component is covered in code and deck" is enforced.

### 6.3 `data/aws_ai_stack_notes.md`

The retrieval corpus. Sections `## id: <id>` + `### Title` + prose. Citations in
`knowledge-bases` output use these ids.

### 6.4 `data/pricing_snapshot.json` (generated)

Written by `scripts/snapshot_prices.py` (Codex) from the **AWS Price List API**
(`pricing` client, us-east-1) for **every billable unit a live row in section 7 uses**:
model tokens (service codes `AmazonBedrock`, `AmazonBedrockFoundationModels`),
Guardrails text units, AgentCore runtime compute, Memory events, Code Interpreter
session time, S3 Vectors request tiers, put/query bytes and storage, Comprehend
operation-specific 100-character units (minimum three per request), and synchronous
Translate and standard Polly characters. Price List service codes are
case-sensitive (`comprehend`, `translate` are lowercase; S3 Vectors rows are under
`AmazonS3` with `Vectors-*` usage types). The script discovers the service codes with `DescribeServices`
and records what it could not find. Records: service code, model/feature, region,
usagetype, tier (standard/flex/priority/batch; in-Region vs `cross-region-global`),
unit, USD, `read_at`. Offline estimates use it; the `cost` demo's live lane reads the
API directly. Never hand-edit. **A live row whose unit is missing here is not executed
live** (3.4).

For one SKU's complete, contiguous Comprehend volume schedule, reservations use the
highest marginal rate. Conflicting SKUs or incomplete schedules remain unavailable.

---

## 7. The registry: 30 demos and their operations

★ = presented in the talk (run **offline** on stage; slides quote recorded live
facts where a live run succeeded). Columns: **O** offline, **E** emulator, **L** live.
A cell says how that lane executes the row; `-` means the lane does **not execute**
it and returns `not_run` with `code="contract_only"` (or
`not_supported_by_emulator`) if requested. Effects and phases per section 3.5.
"Fixture" = Stubber, scripted model, MockTransport or a named fixture stream.

Live `write`/`session_start` rows additionally need `--allow-create`, a price row in
`data/pricing_snapshot.json` for every billable unit they use, and the cleanup rule
of section 3.6; if any of those is missing, the row is not executed live.

### 7.1 Models - lane `models`

**★ `bedrock-runtime`** - Converse vs InvokeModel - lineage `bedrock`, `converse-api`, `invoke-model-api`

| Operation | Phase | O | E | L | Effect |
|---|---|---|---|---|---|
| `bedrock-runtime.Converse` (text) | main | fixture | LocalStack | AWS | infer |
| `bedrock-runtime.Converse` with `toolConfig` (`price_pilot`), result validated by pydantic | main | fixture | LocalStack | AWS | infer |
| `bedrock-runtime.InvokeModel` (Anthropic Messages body for Haiku; Nova-compatible body for explicit Nova and emulator examples) | main | fixture | LocalStack | AWS | infer |
| `bedrock-runtime.ConverseStream` | main | fixture stream | - (`not_supported_by_emulator`) | AWS | infer |
| `bedrock-runtime.CountTokens` | setup | fixture | - | AWS | read |
| `bedrock-runtime.StartAsyncInvoke` | main | fixture | - | - | none |
| `bedrock.CreateModelInvocationJob` | main | fixture | - | - | none |

**`bedrock-openai`** - OpenAI-compatible APIs and API keys - lineage `bedrock-mantle`, `bedrock-runtime-openai`, `bedrock-api-keys`

| Operation | Phase | O | E | L | Effect |
|---|---|---|---|---|---|
| Mint a short-term Bedrock API key from the selected session (`aws-bedrock-token-generator`; signing is local) | setup | fixture token | - | local | none |
| `openai` `chat.completions.create` at `https://bedrock-runtime.{region}.amazonaws.com/openai/v1` | main | MockTransport | - | AWS | infer |
| `openai` `responses.create`, same base URL | main | MockTransport | - | AWS | infer |

Never read `OPENAI_API_KEY` here (on the workshop machine it is a real OpenAI key);
pass `api_key=` explicitly. Model selection is **per API**: Chat Completions and
Responses each take a model id that `ListFoundationModels` (filtered to provider
OpenAI) reports for the region and that the snapshot prices; defaults are chosen at
implementation time from that list and recorded in section 9.6. If a model does not
support Responses, that row is `not_run` with `model_unavailable`.

**★ `model-lifecycle`** - catalog and lifecycle - lineage `bedrock-model-lifecycle`, `titan-text`, `nova-v1`, `nova-2`, `bedrock-inference-profiles`

| Operation | Phase | O | E | L | Effect |
|---|---|---|---|---|---|
| `bedrock.ListFoundationModels` -> counts by provider and `modelLifecycle.status` (`ACTIVE`/`LEGACY`) | main | fixture | LocalStack | AWS | read |
| `bedrock.GetFoundationModel` with the **foundation-model id** `amazon.nova-2-lite-v1:0` (not the `global.` profile) -> `modelLifecycle` incl. `legacyTime`, `endOfLifeTime` when present | main | fixture | LocalStack | AWS | read |
| `bedrock.ListInferenceProfiles` -> `us.` / `eu.` / `apac.` / `global.` split | main | fixture | - | AWS | read |
| Join with `data/lineage.yaml` model records | main | local | local | local | none |

**`guardrails`** - Bedrock Guardrails - lineage `guardrails`, `comprehend-prompt-safety`

| Operation | Phase | O | E | L | Effect |
|---|---|---|---|---|---|
| `bedrock.CreateGuardrail` (PII email anonymize, one denied topic, prompt-attack filter; `clientRequestToken`, tags) | setup | fixture | - | AWS | write |
| `bedrock.CreateGuardrailVersion` | setup | fixture | - | AWS | write |
| `bedrock-runtime.ApplyGuardrail` (`source=INPUT`, text containing an email address) | main | fixture | - | AWS | infer |
| `bedrock-runtime.ApplyGuardrail` (`source=OUTPUT`, safe text) | main | fixture | - | AWS | infer |
| `bedrock.DeleteGuardrail` (deletes its versions) | teardown | fixture | - | AWS | delete |

Live without `--allow-create`: if `AWSAI_GUARDRAIL_ID` and `AWSAI_GUARDRAIL_VERSION`
are set, only the two `ApplyGuardrail` rows run (`infer`); otherwise `not_run`.

**`bedrock-flows`** - Prompt management and Flows - lineage `bedrock-flows`, `prompt-management`

| Operation | Phase | O | E | L | Effect |
|---|---|---|---|---|---|
| `bedrock-agent.CreatePrompt`, `CreatePromptVersion` | setup | fixture | - | - | none |
| `bedrock-agent.CreateFlow`, `PrepareFlow`, `CreateFlowVersion`, `CreateFlowAlias` | setup | fixture | - | - | none |
| `bedrock-agent-runtime.InvokeFlow` | main | fixture stream | - | - | none |
| `bedrock-agent.ListPrompts` | main | fixture | - | AWS | read |
| `bedrock-agent.ListFlows` | main | fixture | - | AWS | read |

**`nova-act`** - Amazon Nova Act - lineage `nova-act`

| Operation | Phase | O | E | L | Effect |
|---|---|---|---|---|---|
| `nova-act.ListModels` | main | fixture | - | AWS | read |
| `nova-act.CreateWorkflowDefinition`, `CreateWorkflowRun` | main | fixture | - | - | none |

**`model-customization`** - fine-tuning, RFT, distillation, import - lineage `model-customization`

| Operation | Phase | O | E | L | Effect |
|---|---|---|---|---|---|
| `bedrock.CreateModelCustomizationJob` (`FINE_TUNING`, `REINFORCEMENT_FINE_TUNING`, `DISTILLATION`) | main | fixture | - | - | none |
| `bedrock.CreateModelImportJob`, `CreateCustomModelDeployment` | main | fixture | - | - | none |
| `bedrock.ListCustomModels` | main | fixture | - | AWS | read |

### 7.2 Agents - lane `agents`

**`agents-classic`** - Bedrock Agents Classic - lineage `agents-classic`, `agents-classic-multi-agent`

| Operation | Phase | O | E | L | Effect |
|---|---|---|---|---|---|
| `bedrock-agent-runtime.InvokeAgent` ending in `returnControl`; answered via `sessionState.returnControlInvocationResults` | main | fixture stream | - | - | none |
| `bedrock-agent.ListAgents` | main | fixture | - | AWS | read |
| Classify any `AccessDeniedException` (section 8.8) | main | local | - | local | none |

Live output states only what was observed: "N agents returned in this region;
historical allowlisting is not observable from a list call."

**★ `strands-agent`** - Strands Agents - lineage `strands-agents`

| Operation | Phase | O | E | L | Effect |
|---|---|---|---|---|---|
| Strands `Agent` loop with `tools=[price_pilot]`, callback handler, `structured_output_model=PilotSummary` (framework execution) | main | local | local | local | none |
| Model calls made by that loop | main | ScriptedModel | BedrockModel -> LocalStack (Ollama) | BedrockModel -> AWS | infer |
| Same agent on `OpenAIModel` (`--provider openai`) | main | - | - | OpenAI | infer |

**`strands-multiagent`** - Graph, Swarm, agents-as-tools - lineage `strands-multiagent`, `agents-classic-multi-agent`, `agent-squad`, `multi-agent-orchestrator`

| Operation | Phase | O | E | L | Effect |
|---|---|---|---|---|---|
| `GraphBuilder` researcher -> costing -> reviewer; `Swarm` of two agents (`max_handoffs=3`); one agent used as another's tool (framework execution) | main | local | - | local | none |
| Model calls made by those agents (total bounded by `max_model_calls`) | main | ScriptedModel | - | BedrockModel -> AWS | infer |

**`agentcore-runtime`** - AgentCore Runtime - lineage `agentcore-runtime`

| Operation | Phase | O | E | L | Effect |
|---|---|---|---|---|---|
| Start `BedrockAgentCoreApp` (wrapping the Strands agent) in a subprocess on 127.0.0.1:<free port> | setup | local | - | - | none |
| `GET /ping` | main | local (loopback) | - | - | none |
| `POST /invocations` | main | local (loopback) | - | - | none |
| Model calls inside the app | main | ScriptedModel | - | - | none |
| `bedrock-agentcore-control.CreateAgentRuntime` (request shape) | main | fixture | - | - | none |
| `bedrock-agentcore.InvokeAgentRuntime` (request shape) | main | fixture | - | - | none |
| `bedrock-agentcore-control.ListAgentRuntimes` | main | fixture | - | AWS | read |

Evidence text: "AgentCore Runtime contract on localhost, not AWS." Invoking a deployed
runtime is not executed live: it runs code whose downstream charges we cannot see.

**★ `agentcore-harness`** - AgentCore harness - lineage `agentcore-harness`, `agents-classic`

| Operation | Phase | O | E | L | Effect |
|---|---|---|---|---|---|
| `bedrock-agentcore-control.CreateHarness` (`harnessName`, `executionRoleArn`=`AWSAI_HARNESS_ROLE_ARN`, `model`, `systemPrompt`, `tools` incl. inline function `accept_proposal`, `maxIterations`, `maxTokens`, `timeoutSeconds`, `tags`, `clientToken`) | setup | fixture | - | AWS | write |
| `bedrock-agentcore-control.GetHarness` until ready (bounded by wall time) | setup | fixture | - | AWS | read |
| `bedrock-agentcore.InvokeHarness` -> stream ends with `toolUse` + `messageStop.stopReason="tool_use"` | main | fixture stream | - | AWS | infer |
| `bedrock-agentcore.InvokeHarness` with a `toolResult` for that `toolUseId` (approval answered) | main | fixture stream | - | AWS | infer |
| `bedrock-agentcore-control.DeleteHarness` | teardown | fixture | - | AWS | delete |
| `bedrock-agentcore-control.ListHarnesses` | main | fixture | - | AWS | read |

**Harness live execution is read-only.** CPU, memory and token estimates do not bound session-lifetime charges, automatic CloudWatch telemetry or internally constructed prompts ([AWS harness cost guidance](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/harness-operations.html)). `CreateHarness` and `InvokeHarness` therefore return `not_run` / `budget_exceeded` live, including with `--allow-create`; `ListHarnesses` runs live. The rows above marked AWS for create, invoke and delete describe the request shapes the offline lane validates; they are not dispatched live.

**`agentcore-memory`** - AgentCore Memory - lineage `agentcore-memory`

| Operation | Phase | O | E | L | Effect |
|---|---|---|---|---|---|
| `bedrock-agentcore-control.CreateMemory` (`name`, `eventExpiryDuration=7`, no strategies, tags, `clientToken`) | setup | fixture | - | AWS | write |
| `bedrock-agentcore-control.GetMemory` until `ACTIVE` (bounded) | setup | fixture | - | AWS | read |
| `bedrock-agentcore.CreateEvent` (user turn) | main | fixture | - | AWS | write |
| `bedrock-agentcore.CreateEvent` (assistant turn) | main | fixture | - | AWS | write |
| `bedrock-agentcore.ListEvents` | main | fixture | - | AWS | read |
| `bedrock-agentcore.RetrieveMemoryRecords` (long-term; extraction is asynchronous) | main | fixture | - | - | none |
| `bedrock-agentcore-control.DeleteMemory` (removes its events) | teardown | fixture | - | AWS | delete |
| Name the Strands integration module `bedrock_agentcore.memory.integrations.strands.session_manager` | main | local | - | local | none |

**★ `agentcore-gateway`** - Gateway and Policy - lineage `agentcore-gateway`, `agentcore-policy`

| Operation | Phase | O | E | L | Effect |
|---|---|---|---|---|---|
| Real MCP server (stdio, `mcp` `MCPServer` (`mcp.server.mcpserver`)) exposing `price_pilot` and `accept_proposal`; Strands `MCPClient` lists and calls them | main | local (`transport: none`) | - | - | none |
| Cedar policy evaluated with `cedarpy`: `accept_proposal` **denied** unless `context.approved == true` and `context.proposal_version` matches | main | local | - | - | none |
| `CreateGateway`, `CreateGatewayTarget` (MCP server target), `CreatePolicyEngine`, `CreatePolicy` (request shapes) | main | fixture | - | - | none |
| `bedrock-agentcore-control.ListGateways` | main | fixture | - | AWS | read |
| `bedrock-agentcore-control.ListPolicyEngines` | main | fixture | - | AWS | read |

**`agentcore-identity`** - AgentCore Identity - lineage `agentcore-identity`

| Operation | Phase | O | E | L | Effect |
|---|---|---|---|---|---|
| `CreateWorkloadIdentity`, `CreateOauth2CredentialProvider`, `GetWorkloadAccessToken`, `GetResourceOauth2Token` (request shapes) | main | fixture | - | - | none |
| `bedrock-agentcore-control.ListWorkloadIdentities` | main | fixture | - | AWS | read |
| `bedrock-agentcore-control.ListOauth2CredentialProviders` | main | fixture | - | AWS | read |

The output explains inbound (who may call the agent) versus outbound (what the agent
may call) and names `bedrock_agentcore.identity.requires_access_token`.

**`agentcore-tools`** - Code Interpreter, Browser, Web Search, Payments - lineage `agentcore-code-interpreter`, `agentcore-browser`, `agentcore-web-search`, `agentcore-payments`

| Operation | Phase | O | E | L | Effect |
|---|---|---|---|---|---|
| `bedrock-agentcore.StartCodeInterpreterSession` (`aws.codeinterpreter.v1`, `sessionTimeoutSeconds=300`, `clientToken`) | setup | fixture | - | AWS | session_start |
| `bedrock-agentcore.InvokeCodeInterpreter` (`executeCode`: Python computing the pilot cost) | main | fixture stream | - | AWS | infer |
| `bedrock-agentcore.StopCodeInterpreterSession` | teardown | fixture | - | AWS | session_stop |
| `StartBrowserSession` / `StopBrowserSession` (request shapes) | main | fixture | - | - | none |
| Web Search Gateway connector target (request shape) | main | fixture | - | - | none |
| Payments `CreatePaymentSession`, `ProcessPayment` (request shapes) | main | fixture | - | - | none |

**`mcp`** - Model Context Protocol - lineage `mcp`, `aws-mcp-server`

| Operation | Phase | O | E | L | Effect |
|---|---|---|---|---|---|
| Serve: our `MCPServer` over stdio | main | local | - | - | none |
| Consume: Strands `MCPClient` `list_tools` and `call_tool` | main | local | - | - | none |

The AWS MCP Server (managed) is described in the output, not called.

**`a2a`** - A2A and AWS Agent Registry - lineage `a2a`, `agent-registry`

| Operation | Phase | O | E | L | Effect |
|---|---|---|---|---|---|
| Start Strands `A2AServer` (a2a-sdk 0.3.x) on 127.0.0.1 ephemeral port | setup | local | - | - | none |
| Client fetches the agent card (`/.well-known/agent-card.json`) | main | local (loopback) | - | - | none |
| Client sends one task; the served agent uses ScriptedModel | main | local (loopback) + fixture model | - | - | none |
| `agent-registry.SearchDiscoverableRegistryRecords` (`registryIds=[AWSAI_REGISTRY_ID]`) | main | fixture | - | AWS | read |
| `bedrock-agentcore.GetAgentCard` (request shape) | main | fixture | - | - | none |

**★ `decision`** - the centrepiece, section 5.2. O, E (DynamoDB sink on LocalStack,
model on LocalStack), L (Bedrock model; SQLite sink; the harness lane stays a contract, see 3.4).

### 7.3 Data - lane `data`

**★ `knowledge-bases`** - (Managed) Knowledge Bases - lineage `knowledge-bases`, `managed-knowledge-base`, `kendra`

| Operation | Phase | O | E | L | Effect |
|---|---|---|---|---|---|
| Local retrieval over `data/aws_ai_stack_notes.md`: chunk by `## id:` section, BM25 (pure Python), top-k with section-id citations; cases: answerable, unanswerable (refuses), **injected document retrieved and not obeyed** (fixture in code, not in the corpus) | main | local | local | local | none |
| `bedrock-agent-runtime.Retrieve`, `RetrieveAndGenerate` (managed KB request shapes) | main | fixture | - | - | none |
| `bedrock-agent.CreateKnowledgeBase` (`type: MANAGED`), `CreateDataSource`, `StartIngestionJob` (request shapes) | main | fixture | - | - | none |
| `bedrock-agent.ListKnowledgeBases` | main | fixture | - | AWS | read |

**`kendra`** - Amazon Kendra (maintenance) - lineage `kendra`, `managed-knowledge-base`

| Operation | Phase | O | E | L | Effect |
|---|---|---|---|---|---|
| `kendra.Retrieve` (request shape) | main | fixture | - | - | none |
| Translate that request into the BMKB `Retrieve` request using AWS's published mapping (pure function) | main | local | - | local | none |
| `kendra.ListIndices` | main | fixture | - | AWS | read |

**`s3-vectors`** - Amazon S3 Vectors - lineage `s3-vectors`

| Operation | Phase | O | E | L | Effect |
|---|---|---|---|---|---|
| Local cosine search over deterministic 8-dimension embeddings (reference result) | main | local | - | local | none |
| `s3vectors.CreateVectorBucket` | setup | fixture | - | AWS | write |
| `s3vectors.CreateIndex` (dimension 8, cosine) | setup | fixture | - | AWS | write |
| `s3vectors.PutVectors` (same embeddings, no model call) | main | fixture | - | AWS | write |
| `s3vectors.QueryVectors` (compare with the reference) | main | fixture | - | AWS | read |
| `s3vectors.DeleteIndex` | teardown | fixture | - | AWS | delete |
| `s3vectors.DeleteVectorBucket` | teardown | fixture | - | AWS | delete |

**`ai-services`** - batch, one child per service

| Child | Operation | Lineage | O | E | L | Effect |
|---|---|---|---|---|---|---|
| comprehend | `DetectPiiEntities`, `DetectSentiment` (one 200-char text) | `comprehend` (context: `comprehend-prompt-safety`) | fixture | - | AWS | infer |
| translate | `TranslateText` (same text) | `translate` | fixture | - | AWS | infer |
| polly | `SynthesizeSpeech` (<= 200 chars; report byte length only) | `polly` | fixture | - | AWS | infer |
| transcribe | `StartTranscriptionJob`, `GetTranscriptionJob` | `transcribe` | fixture | LocalStack (Hobby) | - | none |
| textract | `DetectDocumentText` | `textract` | fixture | LocalStack (Ultimate) | - | none |
| rekognition | `DetectLabels` | `rekognition-maintenance-features` (context) | fixture | - | - | none |
| data-automation | `InvokeDataAutomationAsync` | `bedrock-data-automation` | fixture | - | - | none |

### 7.4 Platform - lane `platform`

**`sagemaker-ai`** - SageMaker AI - lineage `sagemaker-ai`, `sagemaker-ai-maintenance-features`, `sagemaker-profiler`

| Operation | Phase | O | E | L | Effect |
|---|---|---|---|---|---|
| `sagemaker-runtime.InvokeEndpoint` (request shape) | main | fixture | - | - | none |
| `sagemaker.ListEndpoints` | main | fixture | LocalStack (Ultimate) | AWS | read |
| `sagemaker.ListModels` | main | fixture | LocalStack (Ultimate) | AWS | read |

### 7.5 Operations - lane `operations`

**`aws-identity`** - IAM and STS - lineage `iam-sts`

| Operation | Phase | O | E | L | Effect |
|---|---|---|---|---|---|
| `sts.GetCallerIdentity` -> principal **type** and redacted ARN (`live_identity` on AWS; `local_emulator` on LocalStack) | main | fixture | LocalStack | AWS | read |
| Authorisation probe `bedrock.ListFoundationModels` -> `ok` or `blocked` / `authorization_denied` | main | fixture | - | AWS | read |

Teaching point: a principal can authenticate and still be denied.

**`evaluation`** - Evaluations and optimization - lineage `agentcore-evaluations`, `agentcore-optimization`

| Operation | Phase | O | E | L | Effect |
|---|---|---|---|---|---|
| Run the Strands agent over `src/awsai_demo/data/eval_cases.jsonl` (runtime data, packaged) and score with code evaluators: correct cost, approval requested, no action before approval; **one case fails on purpose** | main | local framework + ScriptedModel | - | - | none |
| `bedrock-agentcore.Evaluate` (built-in evaluator id; request shape) | main | fixture | - | - | none |
| Recommendations, batch evaluation, A/B test (request shapes) | main | fixture | - | - | none |
| `bedrock-agentcore-control.ListEvaluators` | main | fixture | - | AWS | read |

**`observability`** - OpenTelemetry -> AgentCore Observability - lineage `agentcore-observability`

| Operation | Phase | O | E | L | Effect |
|---|---|---|---|---|---|
| Strands agent run (ScriptedModel) with an in-memory span exporter; print the span tree with the span names Strands 1.57 actually emits (record them in 9.4) | main | local | - | - | none |
| OTLP export of the same spans to a **registered loopback collector** (`--trace otlp`, `OTEL_EXPORTER_OTLP_ENDPOINT` must be `http://127.0.0.1:<port>` or `http://localhost:<port>`), in the same offline run: `execution_target="local"`, `transport="loopback"`, `mode="local_execution"` | main | local | - | - | export |
| OTLP export to a remote endpoint | main | - | - | - | none |

**`cost`** - cost per successful task - lineage `bedrock-service-tiers`, `bedrock-inference-profiles`

| Operation | Phase | O | E | L | Effect |
|---|---|---|---|---|---|
| Cost per successful decision run across standard / flex / priority / batch and in-Region vs `global.`, retries counted, from `data/pricing_snapshot.json` | main | local | local | local | none |
| `pricing.GetProducts` (live price read, same filters as the snapshot) | main | fixture | - | AWS | read |

### 7.6 Local - lane `local`

**★ `localstack`** - LocalStack core - lineage `localstack-auth-token`, `localstack-community-image`

| Operation | Phase | O | E | L | Effect |
|---|---|---|---|---|---|
| `GET /_localstack/health`, `GET /_localstack/info` | setup | - | LocalStack | - | read |
| S3 `CreateBucket`, `PutObject` | main | fixture | LocalStack | - | write |
| DynamoDB `CreateTable`, conditional `PutItem` twice (the approval sink: second write rejected) | main | fixture | LocalStack | - | write |
| Lambda `CreateFunction` (zipped `price_pilot`) and `Invoke` | main | fixture | LocalStack | - | write |

Missing `LOCALSTACK_AUTH_TOKEN` -> `not_run` / `missing_configuration`; emulator down
-> `attempt_failed` / `emulator_unavailable`.

**`localstack-ai`** - LocalStack AI services - lineage `localstack-bedrock`, `localstack-ai-services`

| Operation | Phase | O | E | L | Effect |
|---|---|---|---|---|---|
| `bedrock.ListFoundationModels` | main | - | LocalStack (Ultimate) | - | read |
| `bedrock-runtime.Converse`, `InvokeModel` (`ollama.<DEFAULT_BEDROCK_MODEL>`) | main | - | LocalStack (Ultimate) | - | infer |
| `transcribe.StartTranscriptionJob` on a deterministically generated two-second, 16 kHz mono PCM silence WAV in an emulated S3 bucket | main | - | LocalStack (Hobby) | - | write |
| `textract.DetectDocumentText` | main | - | LocalStack (Ultimate) | - | infer |
| `sagemaker.ListEndpoints` | main | - | LocalStack (Ultimate) | - | read |

Output: a **coverage table** with `advertised` (from `data/lineage.yaml`), `attempted`,
`passed`, `error_code` (`entitlement_missing` when the plan lacks the service).
Offline: the advertised table only (`local_execution`), every row `attempted=false`.

### 7.7 Utilities (not technologies)

`list`, `doctor [--probe]` (`--probe` is explicitly live, read-only and bounded:
STS plus one List call per service), `lineage [--status S]`, `all [--lane L]
[--execution ...]`, `cleanup [--run-id ID] [--execute]`.

Every demo command accepts `--execution offline|emulator|live`,
`--format pretty|json`, `--region`, `--profile`, `--model`, `--provider
offline|bedrock|openai`, `--trace off|memory|otlp` (default `off`) and
`--allow-create`. A lane a command does not execute returns `not_run` with
`contract_only` or `not_supported_by_emulator`.

---

## 8. Cross-cutting implementation notes

1. `runtime.load_env_file()` reads `.env`; real environment variables win.
2. `.env.example` documents every variable the code reads (all blank), including
   `AWS_CREDS_FILE_PATH`, `AWS_PROFILE`, `AWS_REGION`, `AWSAI_HARNESS_ROLE_ARN`,
   `AWSAI_GUARDRAIL_ID`, `AWSAI_GUARDRAIL_VERSION`, `AWSAI_REGISTRY_ID`,
   `OTEL_EXPORTER_OTLP_ENDPOINT`,
   `LOCALSTACK_AUTH_TOKEN`, `LOCALSTACK_ENDPOINT` (default `http://localhost:4566`),
   `DEFAULT_BEDROCK_MODEL`, `OPENAI_API_KEY` (cross-vendor only).
3. `docker-compose.localstack.yml`: `localstack/localstack:2026.09.0@sha256:<digest>`,
   `LOCALSTACK_AUTH_TOKEN=${LOCALSTACK_AUTH_TOKEN:?…}`, `SERVICES` unset (plan
   decides), `DEFAULT_BEDROCK_MODEL=qwen2.5:0.5b` (small), `BEDROCK_PREWARM=1`,
   port 4566, Docker socket mounted for Lambda.
4. Telemetry off by default: Strands/OTel exporters configured only with `--trace`.
5. Agent loops bounded by `ExecutionPolicy.max_agent_iterations`; Swarm `max_handoffs`.
6. `resume_worker` is a module runnable as `python -m awsai_demo.resume_worker`;
   its stdout/stderr pass through `redact()`.
7. Deterministic output: sort keys, fixed fixtures, no wall-clock values in offline
   `data` except durations (excluded from snapshot comparisons).
8. **Maintenance diagnostic**: classify `maintenance_mode` only if an
   `AccessDeniedException` message contains `"Bedrock Agents is in Maintenance Mode"`;
   otherwise `authorization_denied`.
9. Windows: paths via `pathlib`; subprocesses with `sys.executable`; no shell=True.
10. Ports: find a free loopback port by binding to `127.0.0.1:0`.
11. Pricing: estimates read `data/pricing_snapshot.json`; if a model is missing from
    the snapshot, the live call is refused (`budget_exceeded`: "cost cannot be
    bounded"), never estimated as 0.

---

## 9. Verified API facts (2026-10-05)

### 9.1 Packages (PyPI) and pins

| Package | Version | Note |
|---|---|---|
| boto3 / botocore | 1.43.108 | all clients below present |
| strands-agents | 1.57.2 | requires `mcp>=1.23,<2.2`; extra `[a2a]` pins **`a2a-sdk>=0.3,<0.4`** (0.3.26 resolves) |
| bedrock-agentcore | 1.24.0 | `pydantic<2.41.3`; extra `[strands-agents]` pins `mcp<2.0` — **do not use that extra**; `[a2a]` pins a2a-sdk 0.3, `[a2a-v1]` pins 1.x |
| a2a-sdk | 1.2.2 latest | **incompatible** with Strands' A2A module (`ImportError: DataPart`); use 0.3.x via `strands-agents[a2a]` |
| mcp | 2.3.0 latest | 2.1.1 resolves under Strands' pin |
| strands-agents-evals | 1.4.0 | optional; our evaluators are plain functions |
| openai | 3.24.0 | for `bedrock-openai` |
| aws-bedrock-token-generator | 1.1.0 (2025-07-29) | short-term Bedrock API keys; verify API before use |
| cedarpy | 4.12.1 (2026-09-24) | local Cedar evaluation |
| opentelemetry-sdk | 1.45.0 | in-memory exporter |
| python-pptx | 1.0.2 | deck group |
| pyyaml | 6.0.3 | deck + data loading (also a Strands dependency) |

### 9.2 boto3 clients and operation counts

`bedrock-runtime` (11): `ApplyGuardrail, Converse, ConverseStream, CountTokens,
GetAsyncInvoke, InvokeGuardrailChecks, InvokeModel, InvokeModelWithBidirectionalStream,
InvokeModelWithResponseStream, ListAsyncInvokes, StartAsyncInvoke`.
`bedrock-agentcore` (67), `bedrock-agentcore-control` (171), `agent-registry` (3:
`BatchGetDiscoverableRegistryRecord, ListDiscoverableRegistryRecords,
SearchDiscoverableRegistryRecords`), `s3vectors` (21), `nova-act` (16). Full lists:
`docs/research.md` §4.3.

### 9.3 Operation shapes used (required → optional of interest → output)

- `bedrock-agentcore-control.CreateHarness`: **`harnessName`, `executionRoleArn`** →
  `clientToken, environment, model, systemPrompt, tools, skills, allowedTools, memory,
  truncation, hooks, maxIterations, maxTokens, timeoutSeconds, tags` → `harness`.
- `bedrock-agentcore.InvokeHarness`: **`harnessArn`, `runtimeSessionId`, `messages`** →
  `qualifier, model, systemPrompt, tools, allowedTools, maxIterations, maxTokens,
  timeoutSeconds, actorId` → `stream` (event stream).
- `bedrock-agentcore.InvokeAgentRuntime`: **`agentRuntimeArn`, `payload`** →
  `runtimeSessionId, qualifier, contentType, accept, mcp*` → `response, statusCode,
  runtimeSessionId, contentType`.
- `bedrock-runtime.Converse`: **`modelId`** → `messages, system, inferenceConfig,
  toolConfig, guardrailConfig, additionalModelRequestFields, requestMetadata,
  performanceConfig, serviceTier, outputConfig` → `output, stopReason, usage, metrics,
  serviceTier`.
- `bedrock-runtime.CountTokens`: **`modelId`, `input`** → `inputTokens`.
- `bedrock-runtime.ApplyGuardrail`: **`guardrailIdentifier`, `guardrailVersion`,
  `source`, `content`** → `outputScope` → `action, actionReason, outputs, assessments,
  usage, guardrailCoverage`.
- `bedrock.CreateGuardrail`: **`name`, `blockedInputMessaging`,
  `blockedOutputsMessaging`** → policy configs, `tags`, `clientRequestToken` →
  `guardrailId, guardrailArn, version`.
- `bedrock-agentcore-control.CreateMemory`: **`name`, `eventExpiryDuration`** →
  `clientToken, memoryStrategies, tags` → `memory`.
- `bedrock-agentcore.CreateEvent`: **`memoryId`, `actorId`, `eventTimestamp`,
  `payload`** → `sessionId, clientToken, metadata` → `event`.
- `bedrock-agentcore.RetrieveMemoryRecords`: **`memoryId`, `searchCriteria`** →
  `namespace, maxResults`.
- `bedrock-agentcore.StartCodeInterpreterSession`: **`codeInterpreterIdentifier`** →
  `name, sessionTimeoutSeconds, clientToken` → `sessionId`.
- `bedrock-agentcore.InvokeCodeInterpreter`: **`codeInterpreterIdentifier`, `name`** →
  `sessionId, arguments` → `stream`.
- `bedrock-agentcore.Evaluate`: **`evaluatorId`, `evaluationInput`** →
  `evaluationTarget, evaluationReferenceInputs` → `evaluationResults`.
- `bedrock-agentcore-control.CreatePolicy`: **`name`, `definition`,
  `policyEngineId`** → `validationMode, enforcementMode, clientToken`.
- `bedrock-agentcore-control.CreateGatewayTarget`: **`gatewayIdentifier`,
  `targetConfiguration`** → `name, credentialProviderConfigurations, clientToken`.
- `bedrock-agent-runtime.Retrieve`: **`knowledgeBaseId`, `retrievalQuery`** →
  `retrievalConfiguration, guardrailConfiguration` → `retrievalResults`.
- `bedrock-agent-runtime.InvokeAgent`: **`agentId`, `agentAliasId`, `sessionId`** →
  `inputText, sessionState, enableTrace` → `completion` (event stream).
- `s3vectors.QueryVectors`: **`topK`, `queryVector`** → `vectorBucketName,
  indexName|indexArn, filter, returnMetadata, returnDistance` → `vectors`.
- `agent-registry.SearchDiscoverableRegistryRecords`: **`searchQuery`,
  `registryIds`** → `maxResults, filters` → `registryRecords`.

### 9.4 Strands 1.57.2

- `from strands import Agent, tool, ToolContext`; `Agent(model, tools=…,
  system_prompt=…, structured_output_model=…, session_manager=…, hooks=…,
  agent_id=…, name=…, callback_handler=…)`.
- `Model` abstract methods: `get_config`, `update_config`, `stream(messages,
  tool_specs=None, system_prompt=None, *, tool_choice=None, …)`,
  `structured_output(output_model, prompt, system_prompt=None, **kw)`.
- `BedrockModel(*, boto_session=None, boto_client_config=None, region_name=None,
  endpoint_url=None, api_key=None, **model_config)`.
- `ToolContext.interrupt(name, reason=None, response=None)`; `AgentResult` fields:
  `stop_reason, message, metrics, state, interrupts, structured_output, checkpoint`.
- `strands.multiagent`: `GraphBuilder` (`add_node, add_edge, set_entry_point,
  set_max_node_executions, set_execution_timeout, build`), `Swarm(nodes, *,
  entry_point, max_handoffs=20, max_iterations=20, execution_timeout=900.0,
  node_timeout=300.0, …)`.
- `strands.session`: `FileSessionManager, S3SessionManager, RepositorySessionManager,
  SnapshotSessionManager`.
- `strands.tools.mcp.MCPClient`; `strands.multiagent.a2a.A2AServer` (needs the
  `a2a` extra).
- `strands.models.openai.OpenAIModel` (has a `BedrockMantleConfig`).
- Span names Strands 1.57.2 emits for one agent turn with one tool and structured
  output, distinct, in tree order: `invoke_agent Strands Agents`,
  `execute_event_loop_cycle`, `chat`, `execute_tool price_pilot`,
  `execute_tool PilotSummary` (structured output is implemented as a tool call).

### 9.5 bedrock-agentcore 1.24.0

- `from bedrock_agentcore import BedrockAgentCoreApp`; `app.entrypoint` decorator;
  `app.run(port=8080, host=None)`; Starlette-based; serves `/invocations` and `/ping`.
- `bedrock_agentcore.memory`: `MemoryClient, MemorySessionManager`;
  Strands integration under `bedrock_agentcore.memory.integrations.strands`.
- `bedrock_agentcore.identity`: `requires_access_token, requires_api_key,
  requires_wat`.
- `bedrock_agentcore.tools`: `CodeInterpreter, BrowserClient, WebSearchBackend,
  GatewayMcpBackend`; `code_interpreter_client.code_session`.
- `bedrock_agentcore.policy.PolicyEngineClient`; `bedrock_agentcore.evaluation`;
  `bedrock_agentcore.payments`.

### 9.6 Endpoints and live measurements

- OpenAI-compatible base URL: `https://bedrock-runtime.{region}.amazonaws.com/openai/v1`
  (recommended since 2026-09-15); legacy `https://bedrock-mantle.{region}.api.aws/v1`.
- With the CSV credentials (us-east-1): `ListFoundationModels` = 120 models, statuses
  `ACTIVE`, `LEGACY` (`modelLifecycle` = `status` in `ACTIVE|LEGACY` plus optional
  `startOfLifeTime`, `legacyTime`, `publicExtendedAccessTime`, `endOfLifeTime`);
  `global.amazon.nova-2-lite-v1:0` answered Converse (53 in / 4 out tokens) but
  `CountTokens` rejects every Nova id ("The provided model doesn't support counting
  tokens"); `anthropic.claude-haiku-4-5-20251001-v1:0` counts (30 tokens for a short
  prompt), as do Claude Sonnet 4.5 and 4.6;
  `ListAgents` returned no agents in us-east-1. Historical allowlisting is not
  observable from a list call, and we never call `CreateAgent`.
- Price List API (`pricing`, us-east-1, `AmazonBedrock`): Nova 2.0 Lite input tokens
  per 1K — standard in-Region 0.00033, `cross-region-global` 0.00030, flex 0.000165,
  priority 0.0005775, batch 0.000165 (USD, read 2026-10-05). The 10% in-Region premium
  over global is visible in the data.
- Claude Haiku 4.5 rejects `temperature` and `top_p` together; requests set
  `temperature` only.
- `bedrock-openai`: the preferred Chat Completions model is `openai.gpt-oss-20b-1:0`,
  subject to catalog availability and verified pricing. GPT-OSS does not offer the
  Responses API through `bedrock-runtime`; when the catalog shows the model, that
  live row returns `not_run` / `not_supported_by_model` (a neutral skip);
  `model_unavailable` is reserved for a model that is unexpectedly absent.
- LocalStack: Bedrock runtime supports `Converse`, `InvokeModel`,
  `InvokeModelWithBidirectionalStream` only.

---

## 10. Tests and gates

- `tests/test_<module>.py` per module; offline; no network (socket guard); no
  monkey-patching of our code; inject ports.
- Each demo covers, **where applicable to its rows**: success per supported lane
  (fake emulator/live ports), missing configuration, missing SDK, 403, throttling,
  no response (`attempt_failed`), malformed response, budget refusal, `contract_only`
  under an unsupported lane, redaction on failure text. Pure calculations get no
  invented 403 tests.
- Cleanup: tests for intent-before-request, partial creation, failed delete with
  retry, dry-run default, ownership checks refusing a foreign resource.
- Docs: a test extracts every `awsai-demo ...` command from `README.md`,
  `docs/**/*.md` and `deck/content/*.yaml` and parses it with the real CLI parser.
  Commands inside a fenced block preceded by `<!-- example: offline expect=<status>
  -->` are **executed** in a temporary working directory and must produce that
  status (`ok`, `paused`, `blocked`) with exit code 0, or `expect=exit1` for a
  deliberate failure. Live, emulator and placeholder commands are parsed, not run.
- Coverage: `src/awsai_demo`, `deck/` and `scripts/` at **100% statements and
  branches** (`fail_under = 100`), subprocess coverage enabled (`parallel = true`,
  `COVERAGE_PROCESS_START`), `__main__` paths exercised through subprocess tests, no
  `pragma: no cover` beyond the `if TYPE_CHECKING:` exclusion. `mypy --strict` covers
  the same three trees. The gate environment installs the `deck` group.
- Gates (all must pass): `uv run ruff format --check .`, `uv run ruff check .`,
  `uv run mypy`, `uv run pytest`, `uv lock --check`,
  `uv run --group deck python deck/build_deck.py` (layout, text fit, one-line
  headlines, notes per slide, snippet markers, references).

---

## 11. The deck contract (Codex renders, Claude writes content)

### 11.1 Files

- `deck/content/NN_<section>.yaml` (sorted by name = running order).
- `deck/speaker_notes.md`, sections keyed `<!-- slide: <slide-id> -->`.
- `deck/build_deck.py` → `aws_ai_stack.pptx`; reuse the siblings' `theme.py`,
  `components.py`, `diagrams.py`, `notes.py` and their four checks.
- `deck/facts.json` written only by `deck/collect_facts.py`; offline by default;
  `--execution live` requires explicit flags and the 3.4 policy. Offline and live
  facts are stored under separate top-level keys (`offline`, `live`) with timestamp,
  mode, model/profile id, usage and `estimated`/`measured` labels; regenerating
  offline facts never overwrites recorded live facts. The build never calls AWS and
  refuses facts whose **measurement fingerprint** (hash of `uv.lock`, `src/`,
  `data/aws_ai_stack_notes.md`, `data/lineage.yaml`, `data/pricing_snapshot.json`
  and `deck/collect_facts.py`) is stale. Editing slide YAML or notes does not
  invalidate facts.
- `deck/build_llms.py` → `llms-full.txt` (slides in order with segment, time window,
  visible text and speaker notes, then the README). `llms.txt` is Claude's.

### 11.2 YAML schema (`schema_version: 1`)

Loaded with a `SafeLoader` subclass that **rejects duplicate keys**; unknown fields
are errors; slide ids unique across files.

```yaml
schema_version: 1
section: {id: agentcore, index: "04", title: AgentCore, minutes: [25, 38]}
slides:
  - id: agentcore.harness            # <section>.<name>
    kind: content                    # title | section | content | closing | appendix
    eyebrow: agentcore / harness
    title: The harness is Agents Classic, rebuilt on AgentCore   # one line
    deck: Declare the model, tools and prompt; AWS runs the loop.
    audience: [developers, architects]
    layout: main_rail                # full | main_rail | two_column
    main:   [ <block>, ... ]
    rail:   [ <block>, ... ]
    run: [awsai-demo agentcore-harness]   # validated by the CLI parser
```

Blocks (each may set `h:` in inches; default per type; position stacks top-down):

| `type` | Fields |
|---|---|
| `text` | `text`, `size?`, `bold?`, `tone?` |
| `bullets` | `items: [str \| {text, level}]`, `size?` |
| `table` | `columns`, `rows` **or** `source: {lineage: {filter: {field: value}, fields: [...]}}` \| `source: {facts: path}`; `widths?` |
| `lineage` | `ids: [lineage ids]` (rows drawn before → relation → successors, status chip, dates) |
| `timeline` | `events: [{date, label, lane?}]` **or** `ids: [lineage ids]` + `date_field` |
| `layer_map` | `layers: [{name, items: [{label, status?, ref?}]}]` |
| `diagram` | `nodes: [{id, label, x, y, w, h, kind?, lane?}]`, `edges: [{from, to, label?, style?}]`, `boundaries?: [{label, nodes}]` (inches on a 0..CONTENT grid) |
| `code` | `ref: "src/awsai_demo/<file>.py#<marker>"`, `caption?` — **never inline code** |
| `metrics` | `items: [{value, label}]` (values may be `{facts.*}`) |
| `callout` | `text`, `tone: info \| warning \| success` |
| `compare` | `left: {title, items}`, `right: {title, items}` |
| `matrix` | `source: {facts: path}`, `rows?`, `columns?` |
| `steps` | `items: [str]` |
| `image` | `path` (under `deck/assets/`), `caption?` |
| `source_note` | `text` |

Interpolation: only `{facts.<dotted.path>}` and `{lineage.<id>.<field>}`; no
expressions; an unknown key fails the build.

### 11.2a Schema details fixed by the written content

These were pinned down while `deck/content/*.yaml` was written; the renderer and its
validator implement exactly this.

- **Slide kinds.** `title`: `eyebrow`, `title`, `deck` (tagline); presenter, role
  and date come from `build_deck.py` arguments. `section`: `eyebrow`, `title`,
  `deck`, `bullets` (list of strings, the section's stops); the section `index`
  comes from the file's `section` header. `closing`: `eyebrow`, `title`, `deck`.
  `content` and `appendix` use `layout`, `main`, `rail`.
- **Layouts.** `full` (one column, the content width) and `main_rail` (main column
  plus right rail, the siblings' `COL_MAIN_W` / `COL_RAIL_W`). No other layouts.
- **Slide-level fields.** `audience` (list, notes only), `run` (list of CLI command
  strings drawn as the run strip, each parsed by the real CLI parser), `source_note`
  (string, drawn above the footer). All visible strings, including `deck` and
  `source_note`, support interpolation.
- **Section header.** `section: {id, index, title, minutes: [start, end] | null}`;
  `null` for the appendix. Agenda times and llms-full time windows come from here
  and from the notes headings.
- **Table sources.** `source: {lineage: {filter, fields}}`: `filter` maps a record
  field to a value **or a list of values** (membership); several keys are ANDed;
  rows keep `data/lineage.yaml` order. `fields` may include list fields
  (`successors`, `former_names`), rendered as comma-joined **names** (ids resolved to
  record names). Column headers default to the field names title-cased unless
  `columns` is given. `source: {facts: <lane>.<key>}` reads a `table`-typed fact; a
  `columns` list is then required.
- **`lineage` block.** One row per id: `former_names`/`name` -> relation label ->
  successor names, a status chip coloured by `normalized_status` (active teal;
  preview blue; maintenance and legacy amber; sunset and end_of_life coral;
  renamed and moved grey) and the dates `launched`, `effective`, `end_of_support`
  when present.
- **`timeline` block.** `ids` + `date_field` (default `launched`); events sorted by
  date; reduced-precision dates sort as the first day of the period.
- **`layer_map` block.** `items[].status` colours as above; `ref` is a lineage id
  (validated, not drawn).
- **`diagram` block.** Node `kind`: `default`, `guard` (amber), `store` (cylinder),
  `external` (outlined), `note` (no border, muted text). Edge `style`: `solid`
  (default) or `dashed`. `\n` in a label is a line break. Coordinates are inches
  relative to the block's top-left; the validator rejects nodes outside the block.
- **Value formatting.** Booleans render as `yes` / `no`; integers with thousands
  separators; `null` fails the build (a slide must never show "None").
- **Facts contract.** `deck/facts_contract.yaml` (Claude) lists every key slides may
  quote, its lane, source command, derivation and type. `collect_facts.py` must
  produce exactly those keys; the build fails on a reference outside the contract.

### 11.3 Code markers

In source: `# slide: <marker>` … `# end-slide: <marker>` (balanced, unique per file).
The renderer copies the lines between them verbatim (dedented). The marked code must
be ordinary, tested code — a marker is a viewport, not a separate example.

Markers the slides reference (each region at most **14 lines** of at most **72
characters**, so it fits a code panel at a readable size):

| Marker | Must show |
|---|---|
| `bedrock_runtime_demo.py#converse-request` | the `converse(...)` call with `modelId`, `messages`, `system`, `inferenceConfig` |
| `bedrock_openai_demo.py#openai-client` | `OpenAI(base_url=…/openai/v1, api_key=<short-term Bedrock key>)` and one `chat.completions.create` |
| `guardrails_demo.py#apply-guardrail` | one `apply_guardrail(...)` call and reading `action` |
| `strands_agent_demo.py#agent` | the `@tool price_pilot` function and the `Agent(...)` construction with `structured_output_model` |
| `agentcore_runtime_demo.py#app` | `BedrockAgentCoreApp()`, `@app.entrypoint`, the handler calling the agent |
| `agentcore_harness_demo.py#create-harness` | the `create_harness(...)` request incl. the inline function tool |
| `agentcore_gateway_demo.py#cedar-policy` | the Cedar policy text (a Python string constant is fine) |
| `decision_demo.py#interrupt` | the `accept_proposal` tool raising `tool_context.interrupt(...)` |
| `knowledge_bases_demo.py#untrusted` | how retrieved passages are wrapped as untrusted data before generation |
| `kendra_demo.py#translate` | the Kendra filter → BMKB filter translation function |

---

## 12. Repository layout

```text
.
|-- README.md  CONTRIBUTING.md  llms.txt  llms-full.txt  aws_ai_stack.pptx
|-- .env.example  .gitignore  .gitattributes  pyproject.toml  uv.lock
|-- docker-compose.localstack.yml
|-- data/   aws_ai_stack_notes.md  lineage.yaml  components.yaml  pricing_snapshot.json
|-- docs/   research.md  module_spec.md  glossary.md  lineage.md  localstack.md
|           architect_guide.md  decision_records.md  exercises/README.md  exercises/answers.md
|-- scripts/ snapshot_prices.py
|-- src/awsai_demo/  __init__.py cli.py registry.py contracts.py redact.py runtime.py
|           credentials.py policy.py manifest.py stubs.py offline.py providers.py
|           scenario.py lineage.py doctor.py resume_worker.py <30 demo modules>
|-- tests/  test_*.py  fixtures/
`-- deck/   content/*.yaml  speaker_notes.md  assets/  build_deck.py  collect_facts.py
            facts.py facts.json theme.py components.py diagrams.py notes.py
            snippets.py schema.py build_llms.py
```

Git-ignored: `.env`, `.venv/`, `.awsai_runs/`, `.awsai_checkpoints/`,
`.pytest_tmp/`, caches.

---

## 13. Delivery order

1. Codex: skeleton, `pyproject.toml`, contracts, redact, runtime/credentials, policy,
   manifest, stubs, offline model, scenario, lineage loader, registry, CLI, doctor —
   with tests at 100%.
2. Claude in parallel: `data/lineage.yaml`, `data/components.yaml`, deck YAML,
   speaker notes, docs.
3. Codex: the eight ★ demos, then the rest, each with tests.
4. Codex: deck renderer + validators + `collect_facts.py` + `build_llms.py`.
5. Both: offline facts, a bounded live facts run with the CSV credentials, deck build,
   rendered-slide review (Claude), cross-review (Codex reviews deck/docs facts;
   Claude reviews code clarity and claims).
6. Claude: `llms.txt`; review generated `llms-full.txt`.
