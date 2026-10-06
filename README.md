# AWS AI stack — a runnable workshop

Every technology AWS ships for building agents in Python, in one repository:
**Amazon Bedrock**, **Strands Agents**, **Amazon Bedrock AgentCore**, **MCP** and
**A2A** — beside the **Agents Classic**, **Kendra** and first-generation model code
they replace, and **LocalStack** for running parts of it on a laptop.

**Built by Mykola Murha — Data and AI Engineer, DataArt.**

Thirty technologies, thirty modules, one command each. There is a companion deck,
`aws_ai_stack.pptx`, but you do not need it: this README is self-contained, and every
command below is checked by the test suite against the code in this repository.

## The thing that makes this repo different

**Every demo runs on a laptop with no AWS account, no API key and no Docker — and
none of them pretend.**

Each result carries a `mode` saying exactly how it was produced:

| `mode` | Meaning |
|---|---|
| `local_execution` | Real local computation or protocol execution (Strands, MCP, A2A, a local AgentCore app, Cedar). Nothing simulated. |
| `local_contract` | Real SDK and application code with an explicitly synthetic response. The AWS request was validated against the real service model; the fixture is named. |
| `local_emulator` | LocalStack executed the AWS API on `localhost`. Not AWS. |
| `live_model` | A model actually answered. |
| `live_identity` | AWS STS actually verified the credentials. |
| `live_service` | An AWS service actually answered — including a real HTTP 403. |
| `attempt_failed` | A call was sent and no answer came back. |
| `not_run` | A prerequisite was missing. The result says which. |

`status` is separate: a real 403 is `mode="live_service"`, `status="blocked"` — a
*successful* demonstration of an authorisation boundary, not a broken demo.
`contracts.result()` raises if a result's evidence contradicts its mode, and no demo
ever falls back from one lane to another without saying so.

### Summer 2026 changed what a new AWS account can use

On **2026-07-30** AWS closed **Bedrock Agents** (now "Agents Classic"), **Amazon
Kendra** and **Amazon Q Business** to new customers. Existing users keep them; a new
account gets an HTTP 403. Most tutorials online were written before that date. This
repository starts from the lineage, so you build on what your account can actually
use:

```bash
uv run awsai-demo lineage --status maintenance
```

## What this project proves

| # | Claim | Where to look |
|---|---|---|
| 1 | Agents Classic, the AgentCore harness and Strands run **the same business decision**, judged on identical checks | [`decision_demo.py`](src/awsai_demo/decision_demo.py) |
| 2 | The business action cannot run before a **named human** approves, and runs **at most once** even when two processes resume | [`decision_demo.py`](src/awsai_demo/decision_demo.py), [`resume_worker.py`](src/awsai_demo/resume_worker.py) |
| 3 | A **separate OS process** resumes the paused Strands run from disk alone | [`resume_worker.py`](src/awsai_demo/resume_worker.py) |
| 4 | Offline AWS calls are validated against the **real botocore service model**; only the response is synthetic | [`stubs.py`](src/awsai_demo/stubs.py) |
| 5 | A tool call is **denied by policy outside the agent**, in the same Cedar language AgentCore Policy uses | [`agentcore_gateway_demo.py`](src/awsai_demo/agentcore_gateway_demo.py) |
| 6 | A retrieved document containing an instruction is retrieved and **not obeyed** | [`knowledge_bases_demo.py`](src/awsai_demo/knowledge_bases_demo.py) |
| 7 | **Authentication is not authorisation**: two principals pass STS, only one may call Bedrock | [`aws_identity_demo.py`](src/awsai_demo/aws_identity_demo.py) |
| 8 | Every live call is **priced and reserved before it is sent**; unpriceable calls never run | [`policy.py`](src/awsai_demo/policy.py) |
| 9 | Every AWS resource a demo creates is **recorded before the request** and cleaned up only by evidence | [`manifest.py`](src/awsai_demo/manifest.py) |
| 10 | Every lifecycle fact lives **once**, with its source; the CLI, the glossary and the deck read the same record | [`data/lineage.yaml`](data/lineage.yaml) |
| 11 | Every AWS component maps to a demo **and** a slide, and a test fails if one is missing | [`data/components.yaml`](data/components.yaml) |
| 12 | Slides quote only **recorded** facts, and every code panel is cut from tested source | [`deck/`](deck/) |

---

## Quickstart

About five minutes on a clean machine. Every command runs from the repository root.

### 1. Install `uv`

The Python package and project manager
([official docs](https://docs.astral.sh/uv/getting-started/installation/)):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart the terminal, then confirm:

```bash
uv --version
```

### 2. Get the code and build the environment

```powershell
git clone <this-repository>
cd aws-ai-stack-workshop
Copy-Item .env.example .env
uv sync --frozen
```

```bash
git clone <this-repository>
cd aws-ai-stack-workshop
cp .env.example .env
uv sync --frozen
```

**Copy `.env.example` unchanged and leave every value blank.** You do not need a
single credential to continue. `uv` installs Python 3.12 for you if it is missing.

### 3. See what you have

<!-- example: offline expect=ok -->
```bash
uv run awsai-demo list
```

<!-- example: offline expect=ok -->
```bash
uv run awsai-demo doctor
```

`doctor` prints installed package versions, which credentials it can see (presence,
never a value) and which lanes each demo could use on this machine. It makes no
network call; `doctor --probe` does, read-only.

### 4. Run the centrepiece

<!-- example: offline expect=paused -->
```bash
uv run awsai-demo decision
```

One business task — cost a six-week support pilot against a 25,000 USD budget and
refuse to accept it without a named human — built three ways: **Agents Classic**
(return of control), the **AgentCore harness** (an inline function tool) and
**Strands** code (an interrupt). The Strands lane stops with status `paused` and
prints a run id. Approve it from a different process:

```bash
uv run awsai-demo decision --resume RUN_ID --approve --approver "your name"
```

Or let the repository simulate the approval, offline, and replay the run to prove
nothing happens twice:

<!-- example: offline expect=ok -->
```bash
uv run awsai-demo decision --simulate-approval
```

The output is a **responsibility matrix** — for the loop, tool authorisation, the
approval pause, persistence, crash recovery, deduplication, tracing and cost limits,
who owns it: your application, the framework, or AWS — plus both process ids and the
number of business effects after the resume and after the replay.

<!-- example: offline expect=paused -->
```bash
uv run awsai-demo all
```

`all` reports `paused`: twenty-nine demos finish and `decision` waits, correctly,
for a human.

---

## The technologies, and the command for each

Every one has its own module, its own test module and its own slide.
★ marks the eight demos presented in the talk.

### Models — Amazon Bedrock

| Command | Technology | What it shows |
|---|---|---|
| ★ `bedrock-runtime` | Converse, InvokeModel | One request shape for every model, tool use, typed output, streaming |
| `bedrock-openai` | OpenAI-compatible APIs | The `openai` SDK on Bedrock with a short-term Bedrock API key |
| ★ `model-lifecycle` | Model catalog | `ACTIVE` / `LEGACY`, inference profiles, Titan to Nova 2 |
| `guardrails` | Bedrock Guardrails | `ApplyGuardrail` on input and output, without a model call |
| `bedrock-flows` | Prompt management, Flows | Versioned prompts, a flow graph, and the Agents Classic trap |
| `nova-act` | Amazon Nova Act | Browser-automation workflows, as a service |
| `model-customization` | Fine-tuning, RFT, distillation, import | Five ways to change a model, as request shapes |

### Agents — Strands and AgentCore

| Command | Technology | What it shows |
|---|---|---|
| `agents-classic` | Bedrock Agents Classic | Return of control, and what maintenance mode means |
| ★ `strands-agent` | Strands Agents | The agent loop, a tool, structured output, any model |
| `strands-multiagent` | Graph, Swarm, agents-as-tools | The code-first replacement for Classic collaboration |
| `agentcore-runtime` | AgentCore Runtime | `/invocations` and `/ping` served on localhost |
| ★ `agentcore-harness` | AgentCore harness | Agents Classic rebuilt as configuration |
| `agentcore-memory` | AgentCore Memory | Short-term events, long-term strategies |
| ★ `agentcore-gateway` | Gateway and Policy | A real MCP server, a Cedar rule that denies a tool |
| `agentcore-identity` | AgentCore Identity | Inbound versus outbound authorisation |
| `agentcore-tools` | Code Interpreter, Browser, Web Search, Payments | Managed tools and what they cost |
| `mcp` | Model Context Protocol | Serve a tool and consume it |
| `a2a` | Agent2Agent, AWS Agent Registry | An agent card, a task, a registry search |
| ★ `decision` | All three lanes | The centrepiece and the responsibility matrix |

### Data

| Command | Technology | What it shows |
|---|---|---|
| ★ `knowledge-bases` | Knowledge Bases, Managed Knowledge Base | Cited retrieval, and an injected instruction not obeyed |
| `kendra` | Amazon Kendra (maintenance) | A Kendra query translated to a managed knowledge base |
| `s3-vectors` | Amazon S3 Vectors | A vector bucket, an index, a similarity query |
| `ai-services` | Comprehend, Translate, Polly, Transcribe, Textract, Rekognition, Data Automation | One call each, and which features are closing |

### Platform and operations

| Command | Technology | What it shows |
|---|---|---|
| `sagemaker-ai` | SageMaker AI | Endpoints, and the eight features in maintenance |
| `aws-identity` | IAM and STS | Real authentication, and the authorisation gate behind it |
| `evaluation` | AgentCore Evaluations | A gate that fails on purpose |
| `observability` | OpenTelemetry | The span tree an agent run produces |
| `cost` | Pricing | Cost per successful run across tiers and routing |

### Local — LocalStack

| Command | Technology | What it shows |
|---|---|---|
| ★ `localstack` | LocalStack core | Health, plan, and the approval sink in DynamoDB |
| `localstack-ai` | LocalStack AI services | Advertised versus attempted versus passed coverage |

Every command accepts `--execution offline|emulator|live` and `--format pretty|json`.

---

## Going live

Live runs are opt-in, never a silent fallback, and they stop **before** spending
money if something is missing.

### Credentials

The repository picks credentials once, in this order, and never switches identity
after a failure:

1. `--profile NAME`
2. `AWS_PROFILE`
3. `AWS_CREDS_FILE_PATH` — the path to an IAM console access-key CSV
   (`Access key ID,Secret access key`), loaded into memory only
4. the standard AWS credential chain

```dotenv
AWS_CREDS_FILE_PATH=C:\path\to\access-keys.csv
AWS_REGION=us-east-1
```

```bash
uv run awsai-demo doctor --probe
```

```bash
uv run awsai-demo model-lifecycle --execution live
```

```bash
uv run awsai-demo bedrock-runtime --execution live
```

The tested default model is `global.anthropic.claude-haiku-4-5-20251001-v1:0` (Claude Haiku 4.5). It is the
default because Bedrock's `CountTokens` can count its input exactly before the
call, so the budget is reserved from a real number; no Nova model supports
`CountTokens` today. A `global.` profile is
not a data-residency boundary; pass `--model` with a `us.` or `eu.` profile if that
matters to you.

### Cost

Every live model or service call is priced from a snapshot of the AWS Price List API
and **reserved before it is sent**. By default a command may spend at most 0.05 USD
(0.50 USD for `all`); the estimate is printed to stderr first. These are
application-enforced limits, not an AWS billing ceiling. Calls whose cost cannot be
bounded — invoking a deployed runtime, batch jobs, model customization, ingestion,
payments — never run live; their request shapes are shown offline.

### Creating resources

Four demos can create real, cheap, self-contained resources — a memory, a guardrail,
an S3 vector bucket and a Code Interpreter session — and only with `--allow-create`:

```bash
uv run awsai-demo guardrails --execution live --allow-create
```

Each creation is written to a private manifest under `.awsai_runs/` **before** the
request is sent, and deleted in reverse order afterwards. If a run is interrupted:

```bash
uv run awsai-demo cleanup
```

```bash
uv run awsai-demo cleanup --run-id RUN_ID --execute
```

`cleanup` is a dry run unless you pass `--execute`, and it deletes only what it can
prove this repository created. The code never creates IAM roles.

The AgentCore harness is deliberately **not** created live: AWS bills session
lifetime and CloudWatch telemetry beyond what a per-command budget can bound, so
`agentcore-harness --execution live` lists harnesses and stops there: `CreateHarness`,
`GetHarness` and `InvokeHarness` are not sent (without `--allow-create` they report
`create_not_allowed` or `invoke_not_allowed`; with it, `budget_exceeded`). The offline
lane is where those requests are validated against the service model.

### A different vendor, to show Strands is model-agnostic

```dotenv
OPENAI_API_KEY=sk-...
```

```bash
uv run awsai-demo strands-agent --execution live --provider openai
```

---

## LocalStack

LocalStack needs an auth token since 2026-03-23 (the Hobby plan is free for
non-commercial use), and Bedrock emulation needs the Ultimate plan. AgentCore,
Knowledge Bases, Guardrails and S3 Vectors are not emulated. The full picture is in
[`docs/localstack.md`](docs/localstack.md).

```bash
docker compose -f docker-compose.localstack.yml up -d
```

```bash
uv run awsai-demo localstack --execution emulator
```

```bash
uv run awsai-demo decision --execution emulator
```

---

## Observability

Nothing leaves the machine unless you ask. `--trace memory` prints the span tree of a
run; `--trace otlp` sends the same spans to a collector on `localhost`.

<!-- example: offline expect=ok -->
```bash
uv run awsai-demo observability
```

---

## Build the deck

```bash
uv run --group deck python deck/collect_facts.py
```

```bash
uv run --group deck python deck/build_deck.py
```

The first command runs the demos the deck quotes and writes their payloads to
`deck/facts.json`; live facts are recorded separately and only when you ask. The
second renders the slides from `deck/content/*.yaml`, the facts and the speaker notes.
It refuses to write a deck whose shapes overflow the safe area, whose text will not
fit its box, whose headline wraps, whose slide lacks speaker notes, or whose facts
were recorded from different code.

---

## Tests and quality gates

```bash
uv run ruff format --check .
```

```bash
uv run ruff check .
```

```bash
uv run mypy
```

```bash
uv run pytest
```

```bash
uv lock --check
```

All five must pass before this is shared or presented.

- Ruff formats and lints at 79 columns across 26 rule families.
- `mypy --strict` covers `src/`, `deck/` and `scripts/`.
- Coverage is enforced at **100% statements and branches**, including subprocesses.
- Tests are **offline**: they never call AWS, OpenAI or LocalStack, and a network
  guard fails any test that tries.
- The commands in this README marked as examples are **executed** by the suite.

Know what the suite does and does not prove. It proves application behaviour under
controlled responses, and that every AWS request matches the installed service
model. It does **not** prove an integration works against your account — that is what
`doctor --probe` and a live run are for, and exactly why `mode` exists.

---

## Project map

```text
.
|-- data/
|   |-- aws_ai_stack_notes.md          # the local, sourced retrieval corpus
|   |-- lineage.yaml                   # every name, status, date and source
|   |-- components.yaml                # component -> demo -> slide
|   `-- pricing_snapshot.json          # AWS Price List API snapshot
|-- docs/
|   |-- research.md                    # every external claim, dated and sourced
|   |-- glossary.md                    # every AWS AI name, old and new
|   |-- lineage.md                     # migration guides
|   |-- localstack.md                  # what runs locally, on which plan
|   |-- architect_guide.md             # the from-zero curriculum, with gates
|   |-- decision_records.md            # why the repo is built this way
|   |-- module_spec.md                 # the contract every module follows
|   `-- exercises/                     # the four gates, with answers
|-- src/awsai_demo/
|   |-- cli.py  registry.py            # one entry point, one row per technology
|   |-- contracts.py  redact.py        # the mode/status envelope; redaction
|   |-- credentials.py  runtime.py     # credential order, .env, settings
|   |-- policy.py  manifest.py         # budgets; resource manifest and cleanup
|   |-- stubs.py  offline.py           # Stubber helpers; the scripted model
|   |-- providers.py  network.py       # the model switch; the egress guard
|   |-- scenario.py  lineage.py        # the shared task; the lineage loader
|   |-- doctor.py  resume_worker.py    # readiness; the resuming process
|   `-- <30 technology modules>
|-- deck/
|   |-- content/*.yaml                 # the slides, as data
|   |-- speaker_notes.md               # the presenter script
|   |-- facts_contract.yaml            # what the slides may quote
|   `-- *.py                           # renderer, checks, facts, llms-full
|-- scripts/snapshot_prices.py         # refreshes the price snapshot
|-- tests/                             # offline, 100% statements and branches
|-- docker-compose.localstack.yml
|-- .env.example                       # every variable the code reads
`-- pyproject.toml  uv.lock
```

---

## Safety, cost and generated files

- **No credential value is ever printed, logged, returned or committed.** Account ids,
  ARNs, keys and tokens are redacted at every output boundary.
- **Offline is offline.** The offline and emulator lanes install a network guard; the
  only destinations allowed are registered `localhost` endpoints.
- **Every loop is bounded** by iteration, call, retry, time and cost limits.
- **Irreversible actions need a named approval**, bound to the proposal's version,
  and the effect is deduplicated where it is written.
- **Retrieved text is untrusted.** Documents, tool results and other agents' replies
  are data; they never change permissions or approve an action.
- **Generated and git-ignored**: `.venv/`, `.uv-cache/`, `.awsai_runs/`,
  `.awsai_checkpoints/`, `.pytest_tmp/` and the tool caches.

---

## Troubleshooting

**`uv: command not found`** — restart the terminal after installing `uv`.

**A live demo says `not_run` / `missing_configuration`** — that call was not sent.
Read the result's `next_steps` and each operation's `error_code` for what is missing.

**A live demo says `blocked` / `authorization_denied`** — your credentials
authenticated and the principal lacks permission. That is a correct result; grant the
permission or use other credentials.

**`budget_exceeded`** — the call's reserved cost would pass the limit, or the model is
missing from the price snapshot. Refresh it with
`uv run python scripts/snapshot_prices.py` (live, read-only).

**An emulator demo says `entitlement_missing`** — your LocalStack plan does not include
that service; Bedrock, SageMaker and Textract need Ultimate.

**A value in `.env` seems ignored** — real environment variables win over the file.

**`ImportError: cannot import name 'DataPart'`** — something upgraded `a2a-sdk` to 1.x;
Strands' A2A support needs 0.3.x. Run `uv sync --frozen`.

---

## Further reading

**Bedrock** —
[document history](https://docs.aws.amazon.com/bedrock/latest/userguide/doc-history.html) ·
[model lifecycle](https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle.html) ·
[Agents Classic maintenance mode](https://docs.aws.amazon.com/bedrock/latest/userguide/agents-classic-maintenance-mode.html)

**AgentCore** —
[release notes](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/release-notes.html) ·
[harness](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/harness.html)

**Lifecycle** —
[Services in Maintenance](https://docs.aws.amazon.com/general/latest/gr/maintenance_services.html) ·
[Services in Sunset](https://docs.aws.amazon.com/general/latest/gr/sunset_services.html)

**Strands** — [PyPI](https://pypi.org/project/strands-agents/)

**LocalStack** — [Bedrock](https://docs.localstack.cloud/aws/services/bedrock/) ·
[2026.03.0 release](https://blog.localstack.cloud/localstack-for-aws-release-2026-03-0/)

Everything above was read on **2026-10-05**; see [`docs/research.md`](docs/research.md)
for the claim-by-claim record.

---

## About

Built and maintained by **Mykola Murha**, Data and AI Engineer at **DataArt**.

I build agent systems that survive contact with production: explicit control flow,
typed boundaries, least-privilege tool access, approval as a durable state
transition, deduplication at the effect boundary, budgets enforced before the call,
and observability wired in from the first commit.

This repository is a compact, honest demonstration of that approach across the whole
AWS AI stack — including the services that closed while it was being written.

Questions, corrections and merge requests are welcome.
