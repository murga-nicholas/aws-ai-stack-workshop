# Architect guide — the AWS AI stack from zero

A self-paced route through the stack for someone who has written Python and used AWS a
little, but has never built an agent. Each part ends with a **gate**: something you
produce, not something you read. The gates are the four promises on slide 2, and the
exercises that test them are in [`exercises/README.md`](exercises/README.md).

Budget about four hours. Everything runs offline unless a step says otherwise.

---

## Part 0 — Set up (15 minutes)

Follow the README quickstart, then run:

```bash
uv run awsai-demo doctor
```

```bash
uv run awsai-demo list
```

`doctor` tells you which credentials it can see (never their values) and which lanes
each demo could use. `list` shows the thirty demos, grouped by lane.

---

## Part 1 — What an agent is, in AWS terms (30 minutes)

An **agent** is four things:

1. a **model** that reads text and decides what to do next;
2. a **system prompt** that tells it its job;
3. **tools** — functions it may ask your code to run;
4. a **loop** that keeps calling the model until it stops asking for tools.

Every AWS agent product is a different answer to one question: *who owns the loop?*

| You write | AWS runs the loop | Product |
|---|---|---|
| Configuration (model, prompt, tools) | Yes | **AgentCore harness** (and, for old accounts, Agents Classic) |
| Python code with a framework | No — your framework does | **Strands** (or another framework), hosted on **AgentCore Runtime** |

Run the smallest real example and read its output field by field:

```bash
uv run awsai-demo strands-agent --format json
```

Find these in the JSON: `mode` (why is it `local_contract`?), `operations` (which ran
locally, which were scripted?), `data` (the typed `PilotSummary`).

**Gate 1 — Explain.** Without notes, draw one request from a user to a Strands agent
on AgentCore Runtime, through a Gateway tool protected by Policy, to a Bedrock model,
and back. Label who owns each arrow: your code, the framework, or AWS.

---

## Part 2 — The model layer (30 minutes)

Read slides 12–16, then run:

```bash
uv run awsai-demo bedrock-runtime
```

```bash
uv run awsai-demo model-lifecycle
```

Learn three things well:

- **Converse versus InvokeModel.** One request shape for every model versus a body
  per provider.
- **The model id form decides where your data goes.** `amazon.…` is one Region,
  `us.…` one geography, `global.…` anywhere.
- **Lifecycle is part of the design.** A Legacy model can lock out a new customer and
  an idle account. Read `modelLifecycle` before you choose.

Then guardrails: `ApplyGuardrail` checks text without a model call, so it protects
any model.

---

## Part 3 — Agents and tools (60 minutes)

Run each and read its headline and `operations`:

```bash
uv run awsai-demo strands-multiagent
```

```bash
uv run awsai-demo agentcore-runtime
```

```bash
uv run awsai-demo agentcore-harness
```

```bash
uv run awsai-demo agentcore-gateway
```

```bash
uv run awsai-demo mcp
```

```bash
uv run awsai-demo a2a
```

Questions to answer for yourself:

- `agentcore-runtime` makes real HTTP calls. To where? Why is its mode
  `local_execution` and not live?
- In `agentcore-gateway`, the Cedar policy denies `accept_proposal`. Where does that
  check run, and why does it matter that it is not inside the agent?
- In `agentcore-harness`, what replaces Agents Classic's "return of control"?

Then the centrepiece:

```bash
uv run awsai-demo decision
```

It stops with status `paused` and prints a run id. Resume it as a named human — the
resume runs in a separate OS process — and then replay it:

```bash
uv run awsai-demo decision --resume RUN_ID --approve --approver "your name"
```

```bash
uv run awsai-demo decision --replay RUN_ID
```

The replay must find the effect already written and do nothing. For a one-command
version of the same proof, `--simulate-approval` pauses, resumes in a separate OS
process as the approver `workshop-simulation`, and replays; it runs offline only:

```bash
uv run awsai-demo decision --simulate-approval
```

Read the responsibility matrix row by row.

**Gate 2 — Choose.** For a use case of your own, fill in the responsibility matrix
for the harness and for Strands on Runtime, and write three sentences defending your
choice. A good answer names at least one thing you give up.

---

## Part 4 — Data (30 minutes)

```bash
uv run awsai-demo knowledge-bases
```

```bash
uv run awsai-demo s3-vectors
```

```bash
uv run awsai-demo kendra
```

The retrieval demo plants a document that contains an instruction. Find the case in
the output: it was retrieved and not obeyed. Retrieved text is **data**. It may be
quoted and cited; it may never change permissions or approve an action.

---

## Part 5 — Lifecycle and migration (30 minutes)

```bash
uv run awsai-demo lineage --status maintenance
```

Read [`lineage.md`](lineage.md) sections 1 and 2. Then answer: a customer built on
Agents Classic in 2024 and wants to open a new AWS account for a second region. What
happens, and what do you recommend?

**Gate 3 — Migrate.** Take one Agents Classic agent definition (the fixture in the
`agents-classic` demo is enough) and write its mapping to the harness, row by row,
including the one feature that does not map.

---

## Part 6 — Running it honestly (45 minutes)

Three lanes, one command each time:

```bash
uv run awsai-demo bedrock-runtime
```

```bash
uv run awsai-demo bedrock-runtime --execution emulator
```

```bash
uv run awsai-demo bedrock-runtime --execution live
```

The emulator lane needs LocalStack with an Ultimate token ([`localstack.md`](localstack.md));
the live lane needs AWS credentials with Bedrock permissions and spends a fraction of
a cent. If you have neither, the second and third commands return mode `not_run`
with error code `missing_configuration` before sending anything; the live one names
the four credential sources it tried. That is the correct result.

Then evaluation and observability:

```bash
uv run awsai-demo evaluation
```

```bash
uv run awsai-demo observability
```

One evaluation case fails on purpose. A gate that cannot fail is not a gate.

**Gate 4 — Run honestly.** Run three demos in every lane available to you and write
one line per run: the mode, the status, and what that combination proves and does not
prove.

---

## Where to go next

- [`module_spec.md`](module_spec.md) — the contract every module follows; read it
  before adding a technology.
- [`decision_records.md`](decision_records.md) — why the repository is built this way.
- [`research.md`](research.md) — every external claim, dated and sourced.
- [`glossary.md`](glossary.md) — when a name confuses you.
