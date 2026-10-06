# The AWS AI glossary — every name, what it is, what it was called

AWS renamed, merged or closed a third of its generative AI names between 2024 and
2026, and the old names still dominate search results, blog posts and Stack Overflow
answers. This file is the decoder ring.

Read on **2026-10-05**. The status tables in section 6 are rendered from
[`data/lineage.yaml`](../data/lineage.yaml) — the same records that
`uv run awsai-demo lineage` prints and the deck quotes — and a test fails if they
disagree. Sources for every date are in that file and in [`research.md`](research.md).

Legend: **ACTIVE** · **PREVIEW** · **MAINTENANCE** (existing customers only, no new
features) · **SUNSET** (has an end-of-support date) · **LEGACY** (a model: no new
customers, migrate) · **END OF LIFE** (gone) · **RENAMED** · **MOVED**.

---

## 1. Three things people call "Bedrock agents" — learn this first

This is the most common confusion in an AWS conversation about agents in 2026.

| Name | What it actually is | Who runs the reasoning loop | Can a new account use it? |
|---|---|---|---|
| **Amazon Bedrock Agents Classic** (was "Agents for Amazon Bedrock") | The 2023 fully managed agent service: action groups, return of control, knowledge bases attached to the agent | AWS | **No.** Closed to new customers on 2026-07-30; `CreateAgent` returns HTTP 403 |
| **AgentCore harness** | A configuration-defined managed agent on AgentCore (`CreateHarness`, `InvokeHarness`) | AWS | Yes. GA 2026-06-17; AWS's recommended home for Classic workloads |
| **An agent on AgentCore Runtime** | Your own code (Strands, LangGraph, OpenAI Agents SDK, anything) hosted by AgentCore | You | Yes. GA 2025-10-13 |

> When someone says "we built it on Bedrock Agents", ask which of the three. The
> answer decides whether their architecture is still available to you.

And "AgentCore" itself is not one thing: it is a family of separately usable
building blocks — Runtime, harness, Memory, Gateway, Policy, Identity, Code
Interpreter, Browser, Web Search, Observability, Evaluations, Payments (preview) —
plus the AWS Agent Registry, which started inside AgentCore and now has its own API
namespace.

---

## 2. Two endpoints, three wire formats, one service

Bedrock answers model calls on two endpoints:

| Endpoint | Wire formats | Status |
|---|---|---|
| `bedrock-runtime.{region}.amazonaws.com` | Bedrock's own `Converse` / `InvokeModel`; and, under `/openai/v1`, OpenAI Chat Completions and Responses | **Recommended** for new integrations since 2026-09-15 |
| `bedrock-mantle.{region}.api.aws` | OpenAI Chat Completions and Responses, Anthropic Messages | Launched 2025-12-03; still supported |

`InvokeModel` takes a provider-specific body; `Converse` takes one format for every
model. Prefer `Converse` for AWS-native code and the OpenAI-compatible path for code
that already speaks OpenAI.

---

## 3. Three forms of model id

| Form | Example | Means |
|---|---|---|
| Foundation model | `amazon.nova-2-lite-v1:0` | One Region. Use it with `GetFoundationModel` |
| Geographic inference profile | `us.amazon.nova-2-lite-v1:0` | Requests are routed across Regions **inside one geography** |
| Global inference profile | `global.amazon.nova-2-lite-v1:0` | Routed **worldwide**. More capacity, lower price, and **not** a data-residency boundary |

The workshop's tested default live model is `global.anthropic.claude-haiku-4-5-20251001-v1:0`: it is the cheapest
model we found whose input `CountTokens` can count exactly, which the budget rule
needs. Nova models do not support `CountTokens`.

---

## 4. Four things that used to be called "Amazon Q"

| Name | What it is now |
|---|---|
| **Amazon Q Business** | The employee assistant. **MAINTENANCE** since 2026-06-30; successor **Amazon Quick** |
| **Amazon Q Developer** (IDE plugins) | Was **CodeWhisperer** until 2024-04-30. **SUNSET**, end of support 2027-04-30; successor **Kiro** |
| **Amazon Q Developer** (in the AWS console) | Not affected by the sunset |
| **Amazon Quick** (was "Quick Suite") | The successor of Q Business: Quick Sight, Flows, Automate, Research, Spaces |

None of them is a developer API used by this workshop; they appear on the lineage
slides only.

---

## 5. Words that sound alike and are not

| Pair | The difference |
|---|---|
| **Maintenance** vs **sunset** | Maintenance: existing customers keep it, no end date published. Sunset: an end-of-support date is announced |
| **Legacy** (model) vs **maintenance** (service) | Legacy applies to a model version: new customers cannot start, idle accounts can lose access after 15 days |
| **Knowledge Base** vs **Managed Knowledge Base** | Standard: you choose the vector store. Managed (GA 2026-06-17): Bedrock runs it, search is always hybrid |
| **AgentCore Browser** vs **Nova Act** | Browser is a managed Chrome session any model drives. Nova Act is a model and service trained for UI actions |
| **Strands Agents** vs **Agent Squad** | Strands is AWS's own open-source agent framework. Agent Squad (was `multi-agent-orchestrator`) is a separate AWS Labs community project |
| **AgentCore CLI** vs **starter toolkit** | The CLI (Node.js, GA v0.4.0) is the documented path; `bedrock-agentcore-starter-toolkit` (Python) is still published |
| **SageMaker** vs **SageMaker AI** | Since 2024-12-03, "SageMaker" is the unified data and AI platform; the ML service is "SageMaker AI" |
| **AWS MCP Server** vs **`awslabs/mcp`** | The AWS MCP Server is a managed service (GA 2026-05-06). `awslabs/mcp` is an open-source repository of individual MCP servers |
| **LocalStack Community** vs **LocalStack for AWS** | Since 2026-03-23 there is one image and it needs an auth token; the free Hobby plan replaces the old community image for non-commercial use |

Not covered by this workshop: Trainium, Inferentia and the Neuron SDK (AWS's AI
accelerators and their compiler). They matter for training and hosting your own
models at scale, not for building agents.

---

## 6. Every name, by layer

"Date" is the date the current status took effect, or the launch date for an active
name.

### 6.1 Models and inference

| Name | Was called | Status | Date | Successor | Record |
|---|---|---|---|---|---|
| **Amazon Bedrock** | - | ACTIVE | 2023-09-28 | - | `bedrock` |
| **InvokeModel API** | - | ACTIVE | 2023-09-28 | Converse API | `invoke-model-api` |
| **Converse API** | - | ACTIVE | 2024-05-30 | - | `converse-api` |
| **bedrock-mantle endpoint** | - | ACTIVE | 2026-09-15 | OpenAI-compatible APIs on bedrock-runtime | `bedrock-mantle` |
| **OpenAI-compatible APIs on bedrock-runtime** | - | ACTIVE | 2026-09-15 | - | `bedrock-runtime-openai` |
| **Amazon Bedrock API keys** | - | ACTIVE | 2025-07-07 | - | `bedrock-api-keys` |
| **Bedrock model lifecycle (Active, Legacy, EOL)** | - | ACTIVE | 2026-09-07 | - | `bedrock-model-lifecycle` |
| **Cross-Region inference profiles** | - | ACTIVE | 2024-08 | - | `bedrock-inference-profiles` |
| **Bedrock inference service tiers** | - | ACTIVE | 2025-11-18 | - | `bedrock-service-tiers` |
| **Amazon Titan Text models** | - | END OF LIFE | 2026-07 | Amazon Nova (first generation) | `titan-text` |
| **Amazon Titan Text Embeddings V2** | - | ACTIVE | 2024-04 | - | `titan-embeddings-v2` |
| **Amazon Nova (first generation)** | - | LEGACY | 2026-03-13 | Amazon Nova 2 | `nova-v1` |
| **Amazon Nova 2** | - | ACTIVE | 2025-12-02 | - | `nova-2` |
| **Anthropic Claude Sonnet 4 on Bedrock** | - | LEGACY | 2026-04-14 | - | `claude-sonnet-4` |
| **Anthropic Claude Haiku 4.5 on Bedrock** | - | ACTIVE | 2025-10-15 | - | `claude-haiku-4-5` |
| **Bedrock model customization** | - | ACTIVE | 2023-09-28 | - | `model-customization` |
| **Amazon Nova Act** | - | ACTIVE | 2025-12 | - | `nova-act` |

### 6.2 Safety and low-code

| Name | Was called | Status | Date | Successor | Record |
|---|---|---|---|---|---|
| **Amazon Bedrock Guardrails** | - | ACTIVE | 2024-04-23 | - | `guardrails` |
| **Amazon Comprehend Prompt Safety Classification** | - | MAINTENANCE | 2026-03-31 | Amazon Bedrock Guardrails | `comprehend-prompt-safety` |
| **Amazon Bedrock Prompt management** | - | ACTIVE | 2024-11 | - | `prompt-management` |
| **Amazon Bedrock Flows** | Prompt Flows | ACTIVE | 2024-11-22 | - | `bedrock-flows` |
| **Amazon Bedrock Studio** | - | MOVED | 2025-03-25 | Amazon SageMaker Unified Studio | `bedrock-studio` |

### 6.3 Agents

| Name | Was called | Status | Date | Successor | Record |
|---|---|---|---|---|---|
| **Amazon Bedrock Agents Classic** | Agents for Amazon Bedrock, Amazon Bedrock Agents | MAINTENANCE | 2026-07-30 | AgentCore harness, AgentCore Runtime | `agents-classic` |
| **Bedrock Agents multi-agent collaboration** | - | MAINTENANCE | 2026-07-30 | Strands multi-agent patterns | `agents-classic-multi-agent` |
| **Strands Agents** | - | ACTIVE | 2025-05 | - | `strands-agents` |
| **Strands multi-agent patterns** | - | ACTIVE | 2025-07 | - | `strands-multiagent` |
| **multi-agent-orchestrator** | - | RENAMED | 2025-05-05 | Agent Squad | `multi-agent-orchestrator` |
| **Agent Squad** | multi-agent-orchestrator | ACTIVE | 2025 | - | `agent-squad` |
| **Amazon Bedrock AgentCore** | - | ACTIVE | 2025-10-13 | - | `agentcore` |
| **AgentCore Runtime** | - | ACTIVE | 2025-10-13 | - | `agentcore-runtime` |
| **AgentCore harness** | - | ACTIVE | 2026-06-17 | - | `agentcore-harness` |
| **AgentCore Memory** | - | ACTIVE | 2025-10-13 | - | `agentcore-memory` |
| **AgentCore Gateway** | - | ACTIVE | 2025-10-13 | - | `agentcore-gateway` |
| **Policy in AgentCore** | - | ACTIVE | 2026-03-03 | - | `agentcore-policy` |
| **AgentCore Identity** | - | ACTIVE | 2025-10-13 | - | `agentcore-identity` |
| **AgentCore Code Interpreter** | - | ACTIVE | 2025-10-13 | - | `agentcore-code-interpreter` |
| **AgentCore Browser** | - | ACTIVE | 2025-10-13 | - | `agentcore-browser` |
| **AgentCore Web Search** | - | ACTIVE | 2026-07 | - | `agentcore-web-search` |
| **AgentCore payments** | - | PREVIEW | 2026-05 | - | `agentcore-payments` |
| **AgentCore Observability** | - | ACTIVE | 2025-10-13 | - | `agentcore-observability` |
| **AgentCore Evaluations** | - | ACTIVE | 2026-03-31 | - | `agentcore-evaluations` |
| **AgentCore optimization loop** | - | ACTIVE | 2026-07 | - | `agentcore-optimization` |
| **AWS Agent Registry** | AgentCore Registry | ACTIVE | 2026-08 | - | `agent-registry` |
| **bedrock-agentcore-starter-toolkit** | - | ACTIVE | 2025-07 | AgentCore CLI | `agentcore-starter-toolkit` |
| **AgentCore CLI** | - | ACTIVE | 2026-03 | - | `agentcore-cli` |
| **Model Context Protocol** | - | ACTIVE | 2024-11 | - | `mcp` |
| **AWS MCP Server** | - | ACTIVE | 2026-05-06 | - | `aws-mcp-server` |
| **Agent2Agent protocol** | - | ACTIVE | 2025-04 | - | `a2a` |

### 6.4 Retrieval and AI services

| Name | Was called | Status | Date | Successor | Record |
|---|---|---|---|---|---|
| **Amazon Bedrock Knowledge Bases** | Knowledge Bases for Amazon Bedrock | ACTIVE | 2023-11-28 | - | `knowledge-bases` |
| **Amazon Bedrock Managed Knowledge Base** | - | ACTIVE | 2026-06-17 | - | `managed-knowledge-base` |
| **Amazon Kendra** | - | MAINTENANCE | 2026-07-30 | Amazon Bedrock Managed Knowledge Base | `kendra` |
| **Amazon S3 Vectors** | - | ACTIVE | 2025-12-02 | - | `s3-vectors` |
| **Amazon Bedrock Data Automation** | - | ACTIVE | 2025-03-03 | - | `bedrock-data-automation` |
| **Amazon Textract** | - | ACTIVE | 2019-05 | - | `textract` |
| **Amazon Transcribe** | - | ACTIVE | 2017-11 | - | `transcribe` |
| **Amazon Polly** | - | ACTIVE | 2016-11 | - | `polly` |
| **Amazon Translate** | - | ACTIVE | 2017-11 | - | `translate` |
| **Amazon Comprehend** | - | ACTIVE | 2017-11 | - | `comprehend` |
| **Amazon Rekognition Streaming Events and Batch Image Content Moderation** | - | MAINTENANCE | 2026-03-31 | - | `rekognition-maintenance-features` |
| **Amazon Forecast** | - | MAINTENANCE | 2024-07-29 | Amazon SageMaker AI | `forecast` |
| **Amazon Fraud Detector** | - | SUNSET | 2025-10-07 | - | `fraud-detector` |
| **Amazon Lookout for Equipment** | - | SUNSET | 2025-10-07 | - | `lookout-for-equipment` |
| **Amazon Monitron** | - | MAINTENANCE | 2024-10-31 | - | `monitron` |
| **Amazon CodeGuru Reviewer** | - | MAINTENANCE | 2025-10-07 | - | `codeguru-reviewer` |

### 6.5 Platform, assistants and identity

| Name | Was called | Status | Date | Successor | Record |
|---|---|---|---|---|---|
| **Amazon SageMaker AI** | Amazon SageMaker | ACTIVE | 2024-12-03 | - | `sagemaker-ai` |
| **SageMaker AI features in maintenance** | - | MAINTENANCE | 2026-06-30 | - | `sagemaker-ai-maintenance-features` |
| **SageMaker AI Profiler** | - | SUNSET | 2026-07-30 | - | `sagemaker-profiler` |
| **Amazon SageMaker Unified Studio** | - | ACTIVE | 2025-03 | - | `sagemaker-unified-studio` |
| **Amazon Q Business** | - | MAINTENANCE | 2026-07-30 | Amazon Quick | `q-business` |
| **Amazon Quick** | Amazon Quick Suite | ACTIVE | 2025-10 | - | `amazon-quick` |
| **Amazon CodeWhisperer** | - | RENAMED | 2024-04-30 | Amazon Q Developer IDE plugins | `codewhisperer` |
| **Amazon Q Developer IDE plugins** | Amazon CodeWhisperer | SUNSET | 2026-05-15 | Kiro | `q-developer-ide` |
| **Kiro** | - | ACTIVE | 2025-11-17 | - | `kiro` |
| **AWS IAM and STS** | - | ACTIVE | 2011 | - | `iam-sts` |

### 6.6 Local development

| Name | Was called | Status | Date | Successor | Record |
|---|---|---|---|---|---|
| **LocalStack Community image** | - | MOVED | 2026-03-23 | LocalStack for AWS (authenticated image) | `localstack-community-image` |
| **LocalStack for AWS (authenticated image)** | LocalStack Pro | ACTIVE | 2026-03-23 | - | `localstack-auth-token` |
| **LocalStack Bedrock emulation** | - | ACTIVE | 2024-11 | - | `localstack-bedrock` |
| **LocalStack AI service emulation** | - | ACTIVE | - | - | `localstack-ai-services` |
