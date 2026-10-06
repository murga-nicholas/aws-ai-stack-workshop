# Lineage and migration guides

What replaced what in the AWS AI stack between 2023 and 2026, and what to do if you
are on the old side of the arrow. Read on **2026-10-05**.

Every date here comes from [`data/lineage.yaml`](../data/lineage.yaml), which lists
the source for each record. To see the same data from the command line:

```bash
uv run awsai-demo lineage
```

```bash
uv run awsai-demo lineage --status maintenance
```

The four status words mean different things, and the difference decides how urgent
a migration is:

| Status | Can existing customers keep using it? | Can a new account start? | Is there an end date? |
|---|---|---|---|
| **Maintenance** | Yes, with bug and security fixes | **No** | Not announced |
| **Sunset** | Until the end-of-support date | No | **Yes** |
| **Legacy** (models) | Yes, but idle accounts can lose access after 15 days | No | Yes (the EOL date) |
| **End of life** | No — requests fail | No | Passed |

---

## 1. Bedrock Agents Classic → AgentCore

**What happened.** Agents for Amazon Bedrock (GA 2023-11-28) was renamed Amazon
Bedrock Agents Classic. AWS announced maintenance mode on 2026-06-30; from 2026-07-30
it is closed to new customers. Accounts with no Agents activity in the previous 12
months receive `AccessDeniedException` (HTTP 403) from `CreateAgent` and
`InvokeInlineAgent`. The model catalog available to Classic agents is frozen at
2026-07-30. There is **no end-of-life date**, and Knowledge Bases and Guardrails are
not affected.

**Two destinations.**

| If your agent is… | Move it to | Why |
|---|---|---|
| Model + action groups + a knowledge base | **AgentCore harness** | Closest to Classic: AWS runs the loop; AWS estimates hours of work |
| A custom orchestrator, multi-agent, or stage prompt overrides | **Code on AgentCore Runtime** (Strands or another framework) | The harness does not express these |

**Mapping, piece by piece** (from AWS's maintenance-mode guide):

| Classic | AgentCore |
|---|---|
| Action group (OpenAPI or function schema + Lambda) | A Gateway target exposed as an MCP tool, or a code-level tool |
| Return of control / `AMAZON.UserInput` | Harness **inline function tool**: the stream ends with a `toolUse` block; your client answers with a `toolResult` |
| `AMAZON.CodeInterpreter` | AgentCore Code Interpreter |
| Knowledge base attached to the agent | Knowledge base reached through the Gateway, or a retrieval tool |
| Session and memory settings | AgentCore Memory (short- and long-term) |
| Guardrail on the agent | Guardrail on the model, plus Policy at the Gateway |
| Multi-agent collaboration | Harness: agent-as-tool only. Full patterns: Strands `GraphBuilder` / `Swarm` on Runtime |
| Stage-specific prompt overrides | **Not replicated** |

**Checklist.**

1. Inventory: list agents, action groups, knowledge bases and aliases per Region.
2. Classify each agent with the table above.
3. For harness candidates, the AgentCore CLI can import a Classic configuration, and
   the `amazon-bedrock` skill in the Agent Toolkit for AWS drives the migration
   with approval checkpoints.
4. Re-run your evaluation cases against the new agent before switching traffic
   (`uv run awsai-demo evaluation` shows the shape of such a gate).
5. Keep the Classic agent until the new one passes; nothing forces a cut-over date.

See it in code: `uv run awsai-demo agents-classic`, `uv run awsai-demo
agentcore-harness`, and the side-by-side `uv run awsai-demo decision`.

---

## 2. Amazon Kendra → Bedrock Managed Knowledge Base

**What happened.** Kendra (2020) entered maintenance on 2026-06-30 and closed to new
customers on 2026-07-30. AWS recommends the Bedrock Managed Knowledge Base (GA
2026-06-17).

**What you gain:** native `RetrieveAndGenerate`, agentic retrieval across knowledge
bases, managed parsing and a managed vector store. **What you lose:** 32+ native
connectors become 7 (route the rest through S3), keyword-only and semantic-only search
(managed is always hybrid), query suggestions, facets, custom synonyms, spell checking,
incremental learning and document-enrichment hooks.

**The API change** is mostly mechanical: `kendra.retrieve(QueryText=…)` becomes
`bedrock-agent-runtime.retrieve(retrievalQuery={"text": …})`, and Kendra's
`AttributeFilter` operators are renamed (`EqualsTo` → `equals`, `AndAllFilters` →
`andAll`, …). `startsWith` and `stringContains` have no equivalent on managed
knowledge bases. `uv run awsai-demo kendra` shows the translation as a tested pure
function.

---

## 3. Amazon Q Business → Amazon Quick

Q Business (GA 2024-04-30) entered maintenance on 2026-06-30. The successor is Amazon
Quick. The fastest path is **Bring Your Own Index**: connect the existing Q Business
index to Quick and run both in parallel. Q Apps are recreated as Quick Flows;
guardrails and actions do not transfer through BYOI. This is an end-user product; the
workshop shows it as lineage only.

---

## 4. CodeWhisperer → Q Developer → Kiro

CodeWhisperer became Amazon Q Developer on 2024-04-30. The Q Developer **IDE plugins
and paid subscriptions** stopped taking new sign-ups on 2026-05-15 and reach end of
support on **2027-04-30**; the successor is **Kiro**, AWS's spec-driven agentic IDE
(GA 2025-11-17). Q Developer in the AWS console is not affected.

---

## 5. Models: Titan → Nova → Nova 2, and the lifecycle in general

Amazon's first Bedrock models were **Titan**. **Nova** replaced them on 2024-12-03 and
**Nova 2** followed on 2025-12-02. Titan Text Premier is gone; Nova v1 Canvas and Reel
reached end of life on 2026-09-30 and Nova Sonic v1 on 2026-09-14. Titan Text
Embeddings V2 remains a supported embedding model.

The general rule matters more than any single model:

- Read `modelLifecycle` from `ListFoundationModels` before you choose a model
  (`uv run awsai-demo model-lifecycle`). `status` is `ACTIVE` or `LEGACY`; timestamps
  such as `legacyTime` and `endOfLifeTime` appear when set.
- A Legacy model cannot be fine-tuned again or given new Provisioned Throughput.
- Models launched on or after 2026-09-07 publish an "EOL no sooner than" date on their
  model card.
- Example on the workshop date: Claude Sonnet 4 is Legacy with end of life on
  2026-10-14.

---

## 6. Endpoints: `bedrock-mantle` → `bedrock-runtime` for OpenAI-compatible calls

Bedrock started serving the OpenAI Responses and Chat Completions APIs on 2025-12-03
at `bedrock-mantle.{region}.api.aws`. On 2026-09-15 AWS made
`https://bedrock-runtime.{region}.amazonaws.com/openai/v1` the recommended path for new
integrations. Both work; new code should use `bedrock-runtime`. See `uv run
awsai-demo bedrock-openai`.

---

## 7. Smaller renames and closures

| Was | Now | Note |
|---|---|---|
| Prompt Flows (preview 2024-07) | Amazon Bedrock Flows | GA 2024-11-22; a Flow's agent node depends on Agents Classic |
| Bedrock Studio | Amazon Bedrock in SageMaker Unified Studio | Moved 2025-03-25 |
| Amazon SageMaker (the ML service) | Amazon SageMaker AI | Renamed 2024-12-03 |
| `multi-agent-orchestrator` | `agent-squad` | PyPI package deprecated 2025-05-05 |
| `bedrock-agentcore-starter-toolkit` | AgentCore CLI | Both published; docs lead with the CLI |
| AgentCore Registry (preview) | AWS Agent Registry (`agent-registry` namespace) | GA 2026-08 |
| Comprehend Prompt Safety Classification | Bedrock Guardrails | Maintenance 2026-03-31 |
| Amazon Forecast | SageMaker Canvas | Maintenance 2024-07-29 |
| Fraud Detector, Lookout for Equipment | — | End of support 2026-10-07 |
| LocalStack community image | LocalStack for AWS with an auth token (free Hobby plan) | 2026-03-23 |

---

## 8. A habit worth keeping

Before you design on any AWS AI service, run:

```bash
uv run awsai-demo lineage --status maintenance
```

Then check AWS's own pages, which change faster than any slide:
[Services in Maintenance](https://docs.aws.amazon.com/general/latest/gr/maintenance_services.html)
and [Services in Sunset](https://docs.aws.amazon.com/general/latest/gr/sunset_services.html).
