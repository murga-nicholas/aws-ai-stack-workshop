# AWS AI stack — research notes

Everything below was read from official AWS documentation, the LocalStack
documentation and the PyPI JSON API on **2026-10-05**. Each claim carries a source
(section 10). Package versions are the ones PyPI reported that day; run
`uv run awsai-demo doctor` to print what is actually installed on your machine rather
than trusting this file.

This document is the evidence base for the deck and the README. It is dated on
purpose: between June and September 2026 AWS moved three of its first-generation
generative AI services into maintenance mode, and blog posts written before that
summer are now wrong in ways that cost money.

---

## 1. The one-paragraph history

AWS built its generative AI stack in **two waves**.

The **first wave (2023–2024)** was *managed and opinionated*. Amazon Bedrock gave
API access to foundation models, and AWS wrapped the most common patterns in fully
managed services: **Agents for Amazon Bedrock** ran the reasoning loop for you,
**Knowledge Bases** ran the retrieval pipeline, **Amazon Kendra** was enterprise
search, and **Amazon Q Business** was a finished assistant for employees. You
configured; AWS orchestrated.

The **second wave (2025–2026)** is *modular and framework-agnostic*. AWS
open-sourced its own agent framework, **Strands Agents** (May 2025), and shipped
**Amazon Bedrock AgentCore** (preview July 2025, GA October 2025): runtime,
memory, gateway, identity, tools, policy, evaluations and observability as separate
building blocks that work with *any* framework and *any* model. In June 2026 AWS
added a managed **harness** on AgentCore — a config-defined agent — and six weeks
later closed the first-wave services to new customers.

| When | What happened |
|---|---|
| 2023-04 | Amazon Bedrock announced (preview) |
| 2023-09-28 | **Amazon Bedrock GA** (us-east-1, us-west-2) |
| 2023-11-28 | **Agents for Amazon Bedrock** and **Knowledge Bases** GA |
| 2024-04-30 | Amazon CodeWhisperer becomes **Amazon Q Developer**; Amazon Q Business GA |
| 2024-11-22 | Prompt Flows (preview 2024-07) GA as **Amazon Bedrock Flows** |
| 2024-12-03 | **Amazon Nova** models; SageMaker becomes **SageMaker AI**; Bedrock multi-agent collaboration |
| 2025-03-25 | Bedrock Studio moves into **SageMaker Unified Studio** |
| 2025-05 | **Strands Agents** open-sourced |
| 2025-07-07 | Bedrock **API keys** |
| 2025-07 | **AgentCore** preview; Strands 1.0; Kiro preview (2025-07-14) |
| 2025-10-13 | **AgentCore GA** in nine Regions |
| 2025-11-17 | Kiro GA |
| 2025-12-02 | **Nova 2**; **S3 Vectors** GA |
| 2025-12-03 | Bedrock **OpenAI-compatible** endpoint (Responses, Chat Completions), named `bedrock-mantle` |
| 2026-03-02 | Fully **managed knowledge bases** with agentic retrieval |
| 2026-03-03 / 03-31 | AgentCore **Policy** GA / AgentCore **Evaluations** GA |
| 2026-05-06 | **AWS MCP Server** GA, with the Agent Toolkit for AWS |
| 2026-06-30 | **Bedrock Agents → "Agents Classic"**, **Kendra** and **Q Business** enter maintenance mode |
| 2026-06-17 | AWS Summit New York: AgentCore **harness** and **Bedrock Managed Knowledge Base** GA |
| 2026-07 | AgentCore **Web Search** GA; Policy runs Bedrock Guardrails at the gateway |
| 2026-07-30 | Agents Classic, Kendra and Q Business **closed to new customers** |
| 2026-08 | **AWS Agent Registry** GA under its own `agent-registry` namespace |
| 2026-09-15 | AWS recommends `bedrock-runtime`, not `bedrock-mantle`, for OpenAI-compatible calls |

The pattern: **the managed products of 2023 are now thin defaults on top of
modular parts**. A team that wants "configure, don't code" gets the AgentCore
harness; a team that wants control writes Strands (or LangGraph, or anything) and
deploys it on AgentCore Runtime. Both share the same memory, gateway, identity and
observability.

---

## 2. Package reality check (PyPI, read 2026-10-05)

| Package | Latest | Released | Verdict |
|---|---|---|---|
| `boto3` / `botocore` | 1.43.108 | 2026-10-02 | **Current.** Ships daily; every AWS API, including AgentCore |
| `strands-agents` | **1.57.2** | 2026-10-01 | **Current.** AWS's open-source agent framework |
| `strands-agents-tools` | 0.8.9 | 2026-09-15 | Current — prebuilt tools |
| `strands-agents-evals` | 1.4.0 | 2026-09-22 | Current — evaluation framework |
| `bedrock-agentcore` | **1.24.0** | 2026-09-28 | **Current.** AgentCore Python SDK (runtime app, memory, identity, tools, evaluation, payments) |
| `bedrock-agentcore-starter-toolkit` | 0.3.13 | 2026-09-15 | Still published; the **AgentCore CLI** (GA v0.4.0, 2026-03) is the documented path |
| `agent-squad` | 1.1.4 | 2026-09-23 | Current — community multi-agent orchestrator from AWS Labs |
| `multi-agent-orchestrator` | 0.1.15 | 2025-05-05 | **Deprecated.** PyPI summary: "Use 'agent-squad' instead" |
| `mcp` | 2.3.0 | 2026-10-02 | Current — Model Context Protocol SDK |
| `a2a-sdk` | 1.2.2 | 2026-10-05 | Current — Agent2Agent SDK |
| `openai` | 3.24.0 | 2026-10-02 | Current — also the client for Bedrock's OpenAI-compatible APIs |
| `langchain-aws` | 1.8.0 | 2026-09-30 | Current |
| `opentelemetry-sdk` | 1.45.0 | 2026-09-25 | Current |
| `aws-opentelemetry-distro` | 0.21.0 | 2026-10-01 | Current — ADOT, the path into CloudWatch GenAI observability |
| `sagemaker` | 3.23.0 | 2026-09-24 | Current — SageMaker Python SDK v3 |
| `localstack` | 2026.8.2 | 2026-09-28 | Current — the LocalStack CLI (calendar versioning) |
| `moto` | 5.2.3 | 2026-08-22 | Current — in-process AWS mocks |

The boto3 1.43.108 service list contains these AI clients:
`bedrock`, `bedrock-runtime`, `bedrock-agent`, `bedrock-agent-runtime`,
`bedrock-agentcore`, `bedrock-agentcore-control`, `bedrock-data-automation`,
`bedrock-data-automation-runtime`, `agent-registry`, `agent-registry-control`,
`nova-act`, `s3vectors`, `sagemaker`, `sagemaker-runtime`, `comprehend`, `textract`,
`transcribe`, `rekognition`, `polly`, `translate`, `kendra`, `qbusiness`, `lexv2-*`.

---

## 3. Amazon Bedrock — the model layer

### 3.1 Three ways to call a model

| API | Endpoint | What it is | Use when |
|---|---|---|---|
| `InvokeModel` | `bedrock-runtime` | The original call. The request **body is provider-specific** (Anthropic Messages JSON, Nova JSON, Llama JSON…) | You need a provider feature Converse does not expose |
| `Converse` / `ConverseStream` | `bedrock-runtime` | One request and response shape for every model, with tool use, documents, images and guardrails | Default for new AWS-native code |
| OpenAI-compatible `Responses` / `Chat Completions`, Anthropic `Messages` | `bedrock-runtime` (recommended since 2026-09-15); `bedrock-mantle.{region}.api.aws` (2025-12-03) | The vendors' own wire formats, served by Bedrock. The `openai` and `anthropic` SDKs work with a changed base URL and key | Porting existing OpenAI or Anthropic code |

`bedrock-runtime` in boto3 1.43.108 has eleven operations: `ApplyGuardrail`,
`Converse`, `ConverseStream`, `CountTokens`, `GetAsyncInvoke`,
`InvokeGuardrailChecks`, `InvokeModel`, `InvokeModelWithBidirectionalStream`,
`InvokeModelWithResponseStream`, `ListAsyncInvokes`, `StartAsyncInvoke`.

### 3.2 Access and authentication

- Since **2025-10-15**, access to all serverless foundation models is enabled by
  default given the right IAM permissions; the old per-model "request access" step is
  gone for most models.
- **Bedrock API keys** (2025-07-07) give a bearer token for the Bedrock APIs, so the
  OpenAI SDK can authenticate without SigV4. IAM can restrict them with
  `bedrock:CallWithBearerToken` and the `bedrock:bearerTokenType` condition key.
- Inference **service tiers**: on-demand Standard, plus **Flex** and **Priority**
  (2025-11-18) and **Reserved** (2025-11-26).
- **Cross-Region inference profiles** (`us.`, `eu.`, `global.` prefixes) route a
  request across Regions for capacity. A geographic profile keeps data inside that
  geography; a `global.` profile does not — a data-residency decision, not a
  performance tweak.
- Quotas changed on **2026-09-21**: tokens per day became a **cross-model,
  per-account** quota on `bedrock-runtime`.

### 3.3 Model lifecycle

AWS describes three states: **Active**, **Legacy** and **End-of-Life (EOL)**. The API
exposes them as `modelLifecycle` in `ListFoundationModels` / `GetFoundationModel`:
`status` is `ACTIVE` or `LEGACY` (an EOL model is no longer listed), with optional
timestamps `startOfLifeTime`, `legacyTime`, `publicExtendedAccessTime` and
`endOfLifeTime`.

- Models launched **on or after 2026-09-07** follow the new policy: the model card
  states an "EOL no sooner than" date and a Legacy period of **6 months or 45 days**.
- Older models follow the legacy policy: at least 12 months on Bedrock, at least 6
  months in Legacy, and — for EOL dates after 2026-02-01 — a *public extended access*
  phase in which the provider may raise prices.
- In Legacy, **new customers cannot adopt the model, existing customers may lose
  access after 15 days of inactivity**, and no new Provisioned Throughput can be
  created. Migration is never automatic.

Pending EOL on 2026-10-05 included Claude Sonnet 4 (EOL 2026-10-14), Claude Opus 4.1
(2027-01-08), Jamba 1.5 (2026-11-26) and TwelveLabs Marengo Embed 2.7 (2026-11-30).
Nova v1 Canvas and Reel reached EOL on 2026-09-30, Nova Sonic v1 on 2026-09-14.

### 3.4 Amazon's own models

Titan (2023) → **Nova** (2024-12-03: Micro, Lite, Pro, Premier, Canvas, Reel, Sonic)
→ **Nova 2** (2025-12-02: Lite, Pro, Sonic, multimodal embeddings). Titan Text
Premier is gone (2026-07); Titan Text Embeddings V2 remains an embedding choice for
knowledge bases. **Amazon Nova Act** is a separate service (boto3 `nova-act`) for
reliable browser-automation agents: workflow definitions, runs, sessions and acts.

### 3.5 Safety: Guardrails

Bedrock Guardrails filters content, denied topics, words and PII, checks contextual
grounding, and — since 2025-08-05 — runs **Automated Reasoning checks** that validate
an answer against formal rules. `ApplyGuardrail` evaluates text **without invoking a
model**, so the same guardrail protects a model hosted anywhere. Guardrails can be
shared across an AWS Organization (2025-11-21) and enforced through an IAM condition
key (2025-03-18). Inside AgentCore, Guardrails also run in **Policy** at the gateway
(2026-07), outside the agent's own code.

Comprehend's **Prompt Safety Classification** entered maintenance on 2026-03-31;
Guardrails is the current path.

### 3.6 Low-code: Prompt management and Flows

Prompt management stores versioned prompts; Flows (GA 2024-11-22) chains prompts,
knowledge bases, Lambda functions, conditions, loops (DoWhile, 2025-09-26) and inline
code into a deployable graph. Both remain active. A Flow's **agent node** calls a
Bedrock agent, which is now an Agents Classic resource — so new accounts cannot build
that node type.

---

## 4. Agents — from Agents Classic to AgentCore

### 4.1 Agents Classic: what changed on 2026-07-30

From the Bedrock User Guide (maintenance-mode page):

- Agents for Amazon Bedrock (launched November 2023) is now **Amazon Bedrock Agents
  Classic**. AWS **announced** maintenance mode on **2026-06-30**; the restriction on
  new customers took **effect on 2026-07-30**.
- **Nothing breaks for existing users.** Accounts with Bedrock Agents activity in the
  previous 12 months are allowlisted. All APIs keep working.
- For **every other account**, `CreateAgent` and `InvokeInlineAgent` return
  **`AccessDeniedException` (HTTP 403)**: "Bedrock Agents is in Maintenance Mode. New
  agent creation is not available for accounts without prior service usage." There
  is no exception process.
- The **model catalog is frozen** at 2026-07-30: new models appear only on AgentCore.
- API namespaces (`bedrock-agent`, `bedrock-agent-runtime`), CloudFormation types and
  IAM prefixes are unchanged — "Classic" is a naming update only.
- There is **no end-of-life date**. Knowledge Bases and Guardrails are not affected.

### 4.2 The two successor paths

| Path | What you write | Closest to |
|---|---|---|
| **AgentCore harness** (recommended) | Configuration: model, system prompt, tools. `CreateHarness`, `InvokeHarness` | Agents Classic's managed loop |
| **Code-defined agent on AgentCore Runtime** | A Strands, LangGraph, OpenAI Agents SDK, Claude Agent SDK or custom agent in a container or a zip | A custom orchestrator |

AWS's own capability mapping:

| Agents Classic | AgentCore |
|---|---|
| Managed orchestration loop | Harness |
| Action groups (OpenAPI / function schema + Lambda) | Gateway tools exposed over MCP, or code-level `@tool` |
| Knowledge base on the agent | Gateway-fronted knowledge base or a retrieval tool |
| Return of control / `AMAZON.UserInput` | Harness **inline function tools**: the harness pauses and returns `tool_use` to the client |
| `AMAZON.CodeInterpreter` | AgentCore Code Interpreter |
| Session and memory settings | AgentCore Memory (short- and long-term strategies) |
| Guardrails on the agent | Bedrock Guardrails + Policy at the gateway |
| Multi-agent collaboration | **Limited** in the harness (agent-as-tool); full patterns need framework code on Runtime |
| Stage-specific prompt overrides | **Not replicated**; system prompt only |

Migration tooling: the AgentCore CLI can import Agents Classic configurations, and the
`amazon-bedrock` skill in the **Agent Toolkit for AWS** drives an end-to-end migration
to the harness. AWS estimates hours for "model + action groups + knowledge base"
agents.

### 4.3 Amazon Bedrock AgentCore, component by component

| Component | What it does | Status (2026-10-05) |
|---|---|---|
| **Runtime** | Serverless host for agent or tool code in isolated microVMs; sessions up to 8 hours; HTTP, MCP, A2A and AG-UI protocols; bidirectional streaming; shell commands (`InvokeAgentRuntimeCommand`) | GA 2025-10 |
| **Harness** | Config-defined managed agent: model, prompt, tools, memory, skills; export to Strands code | GA 2026-06-17 |
| **Memory** | Short-term events and long-term records via strategies (semantic, summary, user preference, episodic, self-managed); direct ingestion (2026-08) | GA |
| **Gateway** | Turns Lambda, OpenAPI, Smithy, API Gateway, MCP servers, HTTP passthrough and model providers into MCP tools; inbound auth; rate limits | GA |
| **Identity** | Workload identities for agents; inbound JWT/IAM; outbound OAuth2 / API-key credential providers; on-behalf-of token exchange | GA |
| **Code Interpreter** / **Browser** | Sandboxed code execution (Python, JavaScript, Node.js) and a managed Chrome session | GA |
| **Web Search** | Managed web search exposed as a Gateway connector, data stays in AWS | GA 2026-07 |
| **Observability** | OpenTelemetry traces into CloudWatch; trajectories; cross-account monitoring | GA |
| **Policy** | Cedar policies evaluated at the gateway on every tool call; natural-language authoring; Guardrails integration | GA 2026-03-03 |
| **Evaluations** | Built-in, custom code-based and third-party evaluators; online and batch | GA 2026-03-31 |
| **Optimization** | Recommendations, batch evaluations, A/B tests (GA 2026-07); failure insights (preview) | Mixed |
| **Managed Knowledge Base** | Fully managed RAG reachable through the gateway; six connectors | GA 2026-06-17 |
| **Payments** | Agent-initiated payments with Coinbase and Stripe | Preview 2026-05 |
| **AWS Agent Registry** | Governed catalog of agents, tools, skills and MCP servers; own namespace | GA 2026-08 |

boto3 1.43.108: `bedrock-agentcore` (data plane) has **67** operations, including
`InvokeAgentRuntime`, `InvokeHarness`, `CreateEvent`, `RetrieveMemoryRecords`,
`StartCodeInterpreterSession`, `InvokeCodeInterpreter`, `StartBrowserSession`,
`GetWorkloadAccessToken`, `GetResourceOauth2Token`, `Evaluate` and `GetAgentCard`.
`bedrock-agentcore-control` has **171**, including `CreateAgentRuntime`,
`CreateHarness`, `CreateMemory`, `CreateGateway`, `CreateGatewayTarget`,
`CreatePolicyEngine`, `CreatePolicy`, `CreateWorkloadIdentity`,
`CreateOauth2CredentialProvider`, `CreateEvaluator` and `CreateRegistry`.
`agent-registry` (discovery) has three: `BatchGetDiscoverableRegistryRecord`,
`ListDiscoverableRegistryRecords`, `SearchDiscoverableRegistryRecords`.

### 4.4 Strands Agents

Strands is the framework AWS built internally (it originally powered Amazon Q
Developer) and open-sourced in May 2025. It is **model-driven**: you give the model a
prompt and tools and let it plan, rather than drawing a graph first. Version 1.57.2
exposes, among others:

- `Agent`, the `tool` decorator, `ToolContext`, skills, plugins and sandboxes;
- models: `BedrockModel` (Converse), `OpenAIModel` (with a `BedrockMantleConfig`),
  a `ModelRouter` with fallback and classifier strategies, plus Anthropic, Ollama,
  LiteLLM and others;
- multi-agent: `GraphBuilder`, `Swarm`, and agents-as-tools;
- **interrupts** (`strands.types.interrupt`) for human-in-the-loop pauses;
- session managers: `FileSessionManager`, `S3SessionManager`,
  `RepositorySessionManager`, `SnapshotSessionManager`; AgentCore Memory ships its own;
- hooks (`BeforeToolCallEvent`, `AfterModelCallEvent`, …), conversation managers,
  structured output (`structured_output_model=`), MCP (`strands.tools.mcp.MCPClient`)
  and A2A.

`multi-agent-orchestrator` → `agent-squad` is a separate, community-maintained AWS
Labs framework; it is not Strands.

---

## 5. Retrieval and data

### 5.1 Knowledge Bases, managed knowledge bases and Kendra

- **Bedrock Knowledge Bases** (GA 2023-11-28): you choose the vector store (OpenSearch
  Serverless or managed clusters, Aurora, Neptune for GraphRAG, MongoDB, Pinecone,
  Redis, **S3 Vectors**), the embedding model and the chunking; `Retrieve` and
  `RetrieveAndGenerate` in `bedrock-agent-runtime`.
- **Fully managed knowledge bases** (2026-03-02; `type: "MANAGED"`) remove the vector
  store from your account: built-in connectors (S3, Confluence, SharePoint, Web
  Crawler, Google Drive, OneDrive, custom), smart parsing, **always-hybrid** search,
  1024-dimension float32 embeddings, and **agentic retrieval** across knowledge
  bases. Semantic chunking is not available on the managed type.
- **Amazon Kendra** entered maintenance on 2026-06-30 and closed to new customers on
  2026-07-30. AWS's migration target is the **Bedrock Managed Knowledge Base
  (BMKB)**. AWS's own comparison: Kendra 32+ connectors, BMKB 7; Kendra
  keyword/semantic/hybrid, BMKB hybrid only; BMKB adds native `RetrieveAndGenerate`
  and agentic retrieval. Kendra features without a BMKB equivalent: query
  suggestions, faceted search, custom synonyms, spell checking, incremental learning,
  custom document enrichment hooks.

### 5.2 S3 Vectors

Vector buckets and indexes inside S3 (boto3 `s3vectors`, 21 operations including
`CreateVectorBucket`, `CreateIndex`, `PutVectors`, `QueryVectors`). GA 2025-12-02:
up to two billion vectors per index, 10,000 indexes per bucket, around 100 ms for
frequent queries, AWS's claim of up to 90% lower cost; a supported Knowledge Bases
vector store.

### 5.3 Task-specific AI services and their status

| Service | Status | Detail |
|---|---|---|
| Amazon Textract | Active | Bedrock Data Automation is the generative alternative for documents |
| Amazon Transcribe | Active | |
| Amazon Polly | Active | |
| Amazon Translate | Active | |
| Amazon Comprehend | **Partly in maintenance** | Topic Modeling, Event Detection and Prompt Safety Classification: maintenance 2026-03-31 |
| Amazon Rekognition | **Partly in maintenance** | Streaming Events and Batch Image Content Moderation: maintenance 2026-03-31 |
| Bedrock Data Automation | Active | Multimodal extraction with blueprints; synchronous invocation 2025-11-20 |
| Amazon Kendra | **Maintenance** | 2026-06-30, see 5.1 |
| Amazon Forecast | **Maintenance** | 2024-07-29; SageMaker Canvas recommended |
| Amazon Fraud Detector | **Sunset** | End of support 2026-10-07 |
| Amazon Lookout for Equipment | **Sunset** | End of support 2026-10-07 |
| Amazon Monitron | **Maintenance** | 2024-10-31 |
| Amazon CodeGuru Reviewer | **Maintenance** | 2025-10-07 |

---

## 6. Platform, assistants and tools

- **Amazon SageMaker AI** (renamed from SageMaker on 2024-12-03) remains the place to
  train, tune and host your own models (JumpStart, HyperPod, endpoints). On
  2026-06-30 **eight** features entered maintenance: A2I, Clarify, Debugger, GeoSpatial,
  Ground Truth, Model Monitor, Role Manager and Studio Lab. Two more are in **sunset**:
  the public Mechanical Turk workforce (end of support 2026-09-30) and **Profiler**
  (end of support 2027-06-30).
- **Amazon Q Business** entered maintenance on 2026-06-30; the successor is
  **Amazon Quick** (Quick Sight, Flows, Automate, Research, Spaces), migrated with
  "Bring Your Own Index".
- **Amazon Q Developer IDE plugins**: new sign-ups blocked 2026-05-15, end of support
  **2027-04-30**; successor **Kiro**. Q Developer in the AWS Console is not affected.
- **AWS MCP Server** (GA 2026-05-06) is a managed MCP server that gives coding agents
  audited access to AWS APIs; it is part of the **Agent Toolkit for AWS** (skills,
  plugins). The open-source `awslabs/mcp` repository holds individual servers,
  including one for AgentCore.

---

## 7. Local development with LocalStack

### 7.1 The 2026 licensing change

- On **2026-03-23** (release `2026.03.0`) LocalStack merged its Community and Pro
  images into **one image that requires an auth token** (`LOCALSTACK_AUTH_TOKEN`).
  The services you get are decided by the token's plan.
- Plans: **Hobby** (free, non-commercial use only), **Base**, **Ultimate** (45-day
  trial), **Enterprise**. Versions are calendar-based; the current tag on 2026-10-05
  is `2026.09.0`.
- A new single-binary CLI, **`lstk`**, sits beside the Python `localstack` CLI.

### 7.2 What it emulates for AI (each service page, read 2026-10-05)

| Service | Plan | Coverage | Notes |
|---|---|---|---|
| Bedrock (`bedrock`) | **Ultimate** | 6 of 108 ops | `ListFoundationModels`, `GetFoundationModel`, batch jobs |
| Bedrock Runtime | **Ultimate** | 3 of 11 ops | `Converse`, `InvokeModel`, `InvokeModelWithBidirectionalStream`; served by **Ollama**; text models only; `DEFAULT_BEDROCK_MODEL`, `BEDROCK_PREWARM`, `BEDROCK_PULL_MODELS`; any Ollama model as `ollama.<name>` |
| SageMaker AI | **Ultimate** | 120 of 404 ops | Runtime `InvokeEndpoint` family |
| Textract | **Ultimate** | 5 of 25 ops | |
| Transcribe | **Hobby** | 16 of 43 ops | Offline speech recognition with Vosk models |
| OpenSearch | **Hobby** | 14 of 96 ops | A vector store you can run locally |
| S3, Lambda, DynamoDB | **Hobby** | high | The non-AI building blocks of most agent tools |

**Not emulated**: AgentCore (feature request closed as not planned), Agents Classic,
Knowledge Bases, Guardrails, S3 Vectors, Comprehend, Kendra. An honest local story
for AgentCore is therefore *its own local mode*: `BedrockAgentCoreApp` serves the
same `/invocations` and `/ping` contract on `localhost:8080` that AgentCore Runtime
calls in the cloud.

---

## 8. Measured on the workshop machine (2026-10-05)

- An IAM user in a sandbox account **authenticates**: STS `GetCallerIdentity`
  succeeds.
- The same principal receives **`AccessDeniedException`** from `ListFoundationModels`,
  `ListInferenceProfiles`, `ListGuardrails`, `ListAgents`, `ListKnowledgeBases`,
  `ListAgentRuntimes` and `ListHarnesses`, in both us-east-1 and us-east-2.

Holding credentials proves who you are, not what you may do. The `aws-identity` demo
shows both halves, and it never prints the account id or the ARN.

---

## 9. Found while building

Verified against the installed packages and, where stated, live read-only calls,
2026-10-05 to 2026-10-06.

| Finding | Consequence in this repository |
|---|---|
| `strands-agents[a2a]` 1.57.2 pins `a2a-sdk>=0.3,<0.4`; with `a2a-sdk` 1.2.2 installed, `strands.multiagent.a2a` fails with `ImportError: cannot import name 'DataPart'` | The `a2a` extra resolves `a2a-sdk` 0.3.x; [DR-009](decision_records.md) |
| `bedrock-agentcore[strands-agents]` pins `mcp<2.0`, Strands pins `mcp<2.2` | That extra is not installed; `mcp` 2.1.x resolves |
| `mcp` 2.x exposes the high-level server as `mcp.server.mcpserver.MCPServer` (older samples use `FastMCP`) | The MCP and Gateway demos use `MCPServer` |
| `CreateHarness.harnessName` must match `[a-zA-Z][a-zA-Z0-9_]{0,39}`; `CreateMemory.name` `[a-zA-Z][a-zA-Z0-9_]{0,47}` — hyphens are rejected | Service-specific resource names (module spec 3.6) |
| `CreateHarness` requires `executionRoleArn`, and AWS bills harness session lifetime and CloudWatch telemetry beyond the invocation | The repository never creates a role, and the harness is never created live (module spec 3.4) |
| botocore `retries={"max_attempts": 2}` means **two retries**; `total_max_attempts: 2` means one | The repository uses `total_max_attempts` |
| `strands.models.bedrock.BedrockModel` streams by default (`ConverseStream`), which LocalStack does not implement | Emulator lane passes `streaming=False` |
| `modelLifecycle.status` is only `ACTIVE` or `LEGACY`; end of life is a separate timestamp | The catalog demo counts the two statuses and reads `endOfLifeTime` |
| `StartCodeInterpreterSession` accepts no tags | Sessions are owned by exact name and client token, never by tag |
| On Windows, `asyncio` opens an internal loopback socket pair and botocore queries the OS version through a subprocess | The offline network guard allows exactly those two, nothing else |
| With the workshop's CSV credentials (us-east-1): 120 foundation models listed; `global.amazon.nova-2-lite-v1:0` answered Converse with 53 input and 4 output tokens | Live inference works on the account; the per-request overhead (53 tokens for a 32-character prompt) is why token budgets need exact counts |
| A second IAM principal passes STS `GetCallerIdentity` and receives `AccessDeniedException` from every Bedrock and AgentCore list call | The authentication-versus-authorisation appendix slide |
| AWS Price List API, `AmazonBedrock`, us-east-1: Nova 2.0 Lite input tokens cost 0.00030 USD per 1K through `cross-region-global` and 0.00033 in-Region; Flex and Batch 0.000165; Priority 0.0005775 | The cost demo and the routing advice on the catalog slide |
| `bedrock-runtime.CountTokens` returns `ValidationException: The provided model doesn't support counting tokens` for every Nova model tried (Nova 2 Lite in all three id forms, Nova Micro, Nova 2 Sonic), and for Claude Sonnet 5 and 5.5; it answers for Claude Haiku 4.5, Sonnet 4.5 and Sonnet 4.6 | A live call can only be priced exactly before dispatch for a model that supports counting; this decides the tested default live model |
| AWS Price List API, `AmazonBedrockFoundationModels`, "Claude Haiku 4.5 (Amazon Bedrock Edition)", us-east-1: 1.00 / 5.00 USD per 1M input / output tokens via global routing, 1.10 / 5.50 in-Region, half for batch | The same 10% in-Region premium as Nova 2 Lite |
| `kendra.ListIndices` with the workshop credentials returned `SubscriptionRequiredException`: the account is new to Kendra, which closed to new customers on 2026-07-30 | Maintenance mode observed live, not only read about |
| `Converse` on Claude Haiku 4.5 with fully authorised credentials returned `ResourceNotFoundException`: "Model use case details have not been submitted for this account. Fill out the Anthropic use case details form before using the model." | A third access layer after authentication and IAM: a one-time, per-account provider form. The demos map it to `model_access_required` |
| `translate.TranslateText` with the workshop credentials returned `SubscriptionRequiredException` (HTTP 400) | Some AI services need an account-level subscription before the first call; the demo reports it as `live_service`, `blocked` |
| A live run of all thirty demos with the workshop credentials leaked no account id, ARN or key into any output | Redaction verified on real responses, not only on fixtures |
| `CreateGuardrailVersion` returned `ConflictException` when it reused the `clientRequestToken` of the `CreateGuardrail` call that made its parent; with its own token it succeeded. Version creation is also asynchronous (HTTP 202, status `VERSIONING`) | Every create call gets its own deterministic token; the demo waits for `READY` before versioning and before `ApplyGuardrail` |
| Real boto3 returns `modelLifecycle` timestamps as `datetime` objects; fixtures that used strings hid a serialization crash that only a live run exposed | Fixtures now use the Python types botocore returns |

---

## 10. Sources

**Amazon Bedrock** —
[document history](https://docs.aws.amazon.com/bedrock/latest/userguide/doc-history.html) ·
[Agents Classic maintenance mode](https://docs.aws.amazon.com/bedrock/latest/userguide/agents-classic-maintenance-mode.html) ·
[model lifecycle](https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle.html) ·
[model lifecycle (legacy)](https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle-legacy.html) ·
[GA announcement 2023](https://aws.amazon.com/about-aws/whats-new/2023/09/amazon-bedrock-generally-available) ·
[Responses API 2025](https://aws.amazon.com/about-aws/whats-new/2025/12/amazon-bedrock-responses-api-from-openai) ·
[redesigned console 2026](https://aws.amazon.com/about-aws/whats-new/2026/06/amazon-bedrock-redesigned-console-optimized-openai-anthropic-compatible-apis/)

**AgentCore** —
[release notes](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/release-notes.html) ·
[GA announcement](https://aws.amazon.com/about-aws/whats-new/2025/10/amazon-bedrock-agentcore-available) ·
[Policy and Evaluations preview](https://aws.amazon.com/about-aws/whats-new/2025/12/amazon-bedrock-agentcore-policy-evaluations-preview) ·
[harness](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/harness.html) ·
[harness GA](https://aws.amazon.com/about-aws/whats-new/2026/06/amazon-bedrock-agentcore-harness-generally-available/) ·
[Managed Knowledge Base launch](https://aws.amazon.com/blogs/aws/introducing-amazon-bedrock-managed-knowledge-base-for-faster-more-accurate-enterprise-ai-applications/) ·
[registry](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/registry.html)

**Lifecycle** —
[Services in Maintenance](https://docs.aws.amazon.com/general/latest/gr/maintenance_services.html) ·
[Services in Sunset](https://docs.aws.amazon.com/general/latest/gr/sunset_services.html) ·
[Kendra availability change](https://docs.aws.amazon.com/kendra/latest/dg/kendra-availability-change.html) ·
[Q Business availability change](https://docs.aws.amazon.com/amazonq/latest/qbusiness-ug/qbusiness-availability-change.html) ·
[Q Developer end of support](https://aws.amazon.com/blogs/devops/amazon-q-developer-end-of-support-announcement/)

**Data** —
[S3 Vectors GA](https://aws.amazon.com/about-aws/whats-new/2025/12/amazon-s3-vectors-generally-available/)

**Tools** —
[AWS MCP Server GA](https://aws.amazon.com/about-aws/whats-new/2026/05/aws-mcp-server/) ·
[Agent Toolkit for AWS](https://aws.amazon.com/about-aws/whats-new/2026/05/agent-toolkit/) ·
[Strands Agents on PyPI](https://pypi.org/project/strands-agents/)

**LocalStack** —
[Bedrock](https://docs.localstack.cloud/aws/services/bedrock/) ·
[SageMaker](https://docs.localstack.cloud/aws/services/sagemaker/) ·
[Transcribe](https://docs.localstack.cloud/aws/services/transcribe/) ·
[Textract](https://docs.localstack.cloud/aws/services/textract/) ·
[2026.03.0 release](https://blog.localstack.cloud/localstack-for-aws-release-2026-03-0/) ·
[pricing](https://www.localstack.cloud/pricing) ·
[AgentCore feature request](https://github.com/localstack/localstack/issues/12974)

**Secondary sources used only for dates the primary pages do not state** —
[AgentCore GA article, 2025-10-13](https://aws-news.com/article/2025-10-13-make-agents-a-reality-with-amazon-bedrock-agentcore-now-generally-available) ·
[Kiro GA, re:Invent 2025 summary](https://caylent.com/blog/aws-reinvent-2025-every-ai-announcement-including-amazon-nova-2-and-kiro)
