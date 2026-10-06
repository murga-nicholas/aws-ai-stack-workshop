# Decision records

Why this repository is built the way it is. Each record states the context, the
decision, what we gave up, and how to tell if the decision has stopped being right.
They were agreed between the two authors of the repository — Claude (deck and docs)
and Codex (code) — before any code was written, and are binding through
[`module_spec.md`](module_spec.md).

---

## DR-001 — Offline first, and every result says how it was produced

**Context.** A workshop that needs an AWS account loses half its audience in the first
five minutes. A workshop that fakes cloud calls loses its credibility the first time
someone looks closely.

**Decision.** Every demo runs offline by default, with no account, no key and no
Docker. Every result carries a `mode` (`local_execution`, `local_contract`,
`local_emulator`, `live_model`, `live_identity`, `live_service`, `attempt_failed`,
`not_run`) and a separate `status`. `contracts.result()` raises if the evidence
contradicts the mode, and there is no silent fallback between lanes: a live run never
falls back to a replayed fixture. If nothing in the live lane runs, the result is
`not_run`; if a live read runs and the billable call is refused, the result keeps that
live mode with status `blocked`.

**Cost.** More code than a "happy path" demo; every module needs three lanes.

**Revisit if** a reviewer can find a result whose mode overstates what ran.

---

## DR-002 — botocore's `Stubber` is the offline contract for AWS calls

**Context.** Offline AWS demos usually mock the client, which proves nothing about
the request. In-process emulators such as `moto` are excellent but do not cover
AgentCore, Agents Classic or most of Bedrock.

**Decision.** Offline AWS calls use a real `boto3` client with botocore's own
`Stubber`. Every stubbed call declares `expected_params`, so the request is validated
against the **real service model** of the installed botocore. Only the response is
synthetic, and it is named (`fixture_id`). Event-stream responses, which `Stubber`
cannot represent, are validated with `botocore.validate.validate_parameters` and
replayed through the demo's port.

**What it does not prove.** IAM permissions, provider-specific semantics, streaming
transport or service behaviour. The slides say exactly this.

**Revisit if** botocore stops shipping `Stubber`, or if AWS publishes local emulation
for AgentCore.

---

## DR-003 — A scripted model inside the real Strands loop

**Context.** Mocking an agent framework hides exactly the behaviour a workshop should
show: tool calls, callbacks, interrupts, session persistence.

**Decision.** The offline model is `ScriptedModel`, a subclass of Strands' own
`Model` base class that replays Converse-shaped stream events. The real `Agent` loop,
tool execution, hooks, interrupts and session managers run around it. Scripted
inference is labelled `local_contract`; the surrounding protocol work (MCP, A2A, the
local AgentCore app) is recorded separately as `local_execution`.

**Revisit if** Strands changes the `Model` interface (four abstract methods in
1.57.2).

---

## DR-004 — One fact, one place: `data/lineage.yaml`

**Context.** Between June and September 2026 AWS moved three services into
maintenance, renamed an endpoint and retired several models. A date typed into a
slide, a README and a CLI table drifts within weeks.

**Decision.** Every lifecycle fact lives once, with its sources, in
`data/lineage.yaml`. The CLI (`awsai-demo lineage`), the glossary tables, the deck's
lineage, timeline and deprecation slides, and each demo's `lifecycle_refs` all read
it. `data/components.yaml` maps every AWS component to the demo and slide that cover
it, and a test fails if a covered component lacks either.

**Revisit if** AWS publishes a machine-readable lifecycle feed we could read instead.

---

## DR-005 — Live runs are bounded before they are sent

**Context.** The workshop machine has working AWS credentials. A demo that loops, or
that invokes something with invisible downstream charges, turns a workshop into a
bill.

**Decision.** An `ExecutionPolicy` caps input tokens, output tokens, model calls,
agent iterations, retries (one), wall time and estimated spend. Every billable live
operation **reserves** a conservative cost before dispatch, priced from a snapshot of
the AWS Price List API; reservations are shared across processes. An operation with
no price row, or with charges we cannot see (invoking a user's deployed runtime,
payments, batch jobs, ingestion), is never executed live. The wording everywhere is
"application-enforced limits, not an AWS billing ceiling".

**Cost.** Some interesting live paths (invoking a deployed AgentCore Runtime, a live
Gateway call) are shown as request shapes only.

**Revisit if** AWS exposes per-request cost ceilings.

---

## DR-006 — Creating AWS resources is opt-in, recorded first, and cleaned up by evidence

**Context.** Four demos can create real resources (memory, guardrail, S3 vector
bucket, Code Interpreter session). The harness would be a fifth, but its full cost
cannot be bounded per command (session-lifetime charges, CloudWatch telemetry), so
it stays read-only live under DR-005. A crash between "create" and "delete"
leaves things running.

**Decision.** Creation requires `--allow-create`. An intent record is written to a
private manifest **before** each create request is sent, with a deterministic,
service-valid name and client token; it is updated with the exact id afterwards.
Teardown runs in reverse order in `finally`. `awsai-demo cleanup` is dry-run by
default and deletes only what it can prove this run created: exact id, `run-id` tag,
naming scheme, account fingerprint and Region. Untaggable children are deleted only
through their owned parent; sessions are reconciled by exact name. The code never
creates IAM roles, deploys containers or starts ingestion jobs.

**Revisit if** a resource type gains tagging (the rules then tighten, not loosen).

---

## DR-007 — Credentials are chosen once, in a fixed order

**Context.** The workshop machine has three credential sources, one of them invalid.
boto3 stops at the first source it finds, not the first that works.

**Decision.** `--profile` → `AWS_PROFILE` → `AWS_CREDS_FILE_PATH` (an IAM console CSV,
loaded into memory only) → the default chain. Once chosen, a failure is reported; it
never falls through to another identity. Offline and emulator lanes never load real
credentials. Account ids, ARNs, keys and tokens are redacted at every output boundary.

**Revisit if** the team standardises on IAM Identity Center profiles only.

---

## DR-008 — Approval is bound to a version; the effect is keyed by the business action

**Context.** A human-in-the-loop agent can be resumed twice — by a retry, a replay or
two operators — and Strands, like most frameworks, may re-enter the tool on resume.

**Decision.** The approval is bound to the proposal's content hash. The business
effect is written once, keyed by `accept-proposal:<proposal version>`, in a SQLite
transaction (offline and live) or a DynamoDB conditional write (emulator). The
approver's label is audit data, not part of the key, and it is explicitly not an
authenticated identity. Default runs never synthesise an approval; the deck's
evidence comes from `--simulate-approval`, which is offline-only and says so.

**Revisit if** the business action gains a natural idempotency key upstream.

---

## DR-009 — Pin Strands' A2A extra, not the newest A2A SDK

**Context.** `a2a-sdk` 1.2.2 is current, but `strands-agents[a2a]` 1.57.2 pins
`a2a-sdk>=0.3,<0.4`; with 1.x installed, Strands' A2A module fails to import
(`DataPart` was removed). `bedrock-agentcore[strands-agents]` separately pins
`mcp<2.0`, against Strands' `mcp<2.2`.

**Decision.** Install `strands-agents[a2a]` and let it resolve `a2a-sdk` 0.3.x; do not
install the `bedrock-agentcore[strands-agents]` extra. Both facts are taught on the
slides because every team will hit them.

**Revisit when** Strands releases support for `a2a-sdk` 1.x.

---

## DR-010 — The deck is data; code panels are cut from tested source

**Context.** The two sibling workshops generate their decks from Python. Here the
authors split the work: Codex writes all Python, Claude writes the deck.

**Decision.** Slides are declarative YAML (`deck/content/*.yaml`) validated against a
schema; Codex's renderer reuses the siblings' DataArt design system and its layout,
text-fit, one-line-headline and notes checks. A code panel can only reference a marked
region of a real, linted, typed and covered source file. Numbers come only from
`deck/facts.json` keys listed in `deck/facts_contract.yaml`; facts are recorded by
`collect_facts.py`, and the build refuses facts whose measurement inputs changed.

**Revisit if** a slide needs a layout the schema cannot express — extend the schema,
do not hand-edit the `.pptx`.

---

## DR-011 — LocalStack is optional and licensed

**Context.** Since 2026-03-23 LocalStack requires an auth token; Bedrock emulation
needs the paid Ultimate plan; the free Hobby plan is non-commercial.

**Decision.** The emulator lane is optional and never required to follow the
workshop. Missing token → `not_run`; plan lacks a service → `entitlement_missing`;
container down → `attempt_failed`. The `localstack-ai` demo reports advertised,
attempted and passed coverage separately, so an advertised feature is never presented
as a tested one.

**Revisit if** LocalStack's licensing changes again.
