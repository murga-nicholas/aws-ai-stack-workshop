# AWS AI stack — grounding corpus

A small, local, sourced corpus. The retrieval and knowledge-base demos index **this
file** so that a grounded answer can be checked against a source you can read in a
minute, and so that a citation can be verified rather than trusted.

Each section has a stable `id`. Citations in demo output refer to these ids.
Everything here was verified on **2026-10-05**; see `docs/research.md` for sources.

---

## id: bedrock-what-it-is
### Amazon Bedrock is the model layer

Amazon Bedrock is a managed service that serves foundation models from Amazon,
Anthropic, OpenAI, Meta, Mistral AI, Cohere, Qwen, DeepSeek and others through one
AWS API, with AWS identity, logging and network controls. It became generally
available on 2023-09-28. Bedrock also hosts the surrounding building blocks:
Guardrails, Knowledge Bases, Prompt management, Flows, batch inference and model
customization.

## id: bedrock-converse
### Converse is the default way to call a model

`Converse` and `ConverseStream` in the `bedrock-runtime` client accept one request
shape for every model: messages, a system prompt, inference settings, tool
definitions, documents and images. `InvokeModel` is the older call; its request body
is provider-specific, so moving from one model family to another means rewriting the
body. Use Converse unless you need a provider feature it does not expose.

## id: bedrock-openai-compatible
### Bedrock speaks the OpenAI and Anthropic wire formats

Since 2025-12-03 Bedrock serves the OpenAI Responses and Chat Completions APIs, and
the Anthropic Messages API. The endpoint was first named `bedrock-mantle`; since
2026-09-15 AWS recommends `bedrock-runtime` for new integrations. Existing OpenAI SDK
code works by changing the base URL and the key. A Bedrock API key (2025-07-07) is a
bearer token that IAM can restrict with the `bedrock:CallWithBearerToken` action.

## id: bedrock-lifecycle
### Every model is Active, Legacy or End-of-Life

`ListFoundationModels` returns a `modelLifecycle` status for every model. In the
Legacy state new customers cannot adopt the model, existing customers may lose access
after 15 days of inactivity, and no new Provisioned Throughput can be created. After
the End-of-Life date requests fail. Migration is never automatic. Models launched on
or after 2026-09-07 publish an "EOL no sooner than" date and a Legacy period of six
months or 45 days.

## id: bedrock-inference-profiles
### Cross-Region inference is a residency decision

An inference profile such as `us.amazon.nova-2-lite-v1:0` routes requests across
Regions in one geography for capacity. A `global.` profile routes worldwide and does
not keep data inside one geography. Choose the profile from your data-residency
requirement first and your throughput need second.

## id: models-titan-to-nova
### Titan became Nova, then Nova 2

Amazon's first Bedrock models were Titan (2023). Amazon Nova replaced them on
2024-12-03, and Nova 2 (Lite, Pro, Sonic, multimodal embeddings) followed on
2025-12-02. Titan Text Premier is no longer available; Nova v1 Canvas and Reel reached
End-of-Life on 2026-09-30. Titan Text Embeddings V2 remains a supported embedding
model for knowledge bases.

## id: guardrails
### Guardrails work without a model call

Bedrock Guardrails filter harmful content, denied topics, words and personal data,
check whether an answer is grounded in its sources, and run Automated Reasoning
checks against formal rules. `ApplyGuardrail` evaluates any text without invoking a
model, so one guardrail can protect a model hosted anywhere. Comprehend's Prompt
Safety Classification entered maintenance on 2026-03-31; Guardrails is the current
path.

## id: flows-prompts
### Prompt management and Flows are the low-code layer

Prompt management stores versioned prompts. Bedrock Flows (generally available
2024-11-22, previously called Prompt Flows) connects prompts, knowledge bases, Lambda
functions, conditions, loops and inline code into a deployable graph. A Flow's agent
node calls an Agents Classic agent, so accounts without prior Agents usage cannot use
that node type.

---

## id: agents-classic-status
### Bedrock Agents is now Agents Classic, in maintenance

Agents for Amazon Bedrock launched in November 2023. AWS announced on 2026-06-30 that
it is now Amazon Bedrock Agents Classic, and from 2026-07-30 it is closed to new
customers. Existing agents keep working and there is no end-of-life date. Accounts
without Bedrock Agents activity in the previous 12 months receive an
AccessDeniedException (HTTP 403) from CreateAgent and InvokeInlineAgent. The model
catalog for Agents Classic is frozen at 2026-07-30.

## id: agents-classic-migration
### Where Agents Classic workloads go

AWS recommends the AgentCore harness for configuration-defined agents and code-defined
agents on AgentCore Runtime for custom orchestration. Action groups become Gateway
tools over MCP; return of control becomes an inline function tool that pauses the
harness; the code interpreter action group becomes AgentCore Code Interpreter; session
memory becomes AgentCore Memory. Stage-specific prompt overrides are not replicated,
and full multi-agent collaboration needs framework code.

## id: strands-overview
### Strands Agents is AWS's open-source agent framework

Strands Agents was open-sourced by AWS in May 2025 and reached 1.0 in July 2025. It
is model-driven: you give an `Agent` a model, a system prompt and tools, and the model
plans the steps. It supports Bedrock, OpenAI, Anthropic, Ollama, LiteLLM and other
providers, MCP tools, A2A, multi-agent graphs and swarms, hooks, session managers and
interrupts for human approval. Version 1.57.2 was current on 2026-10-05.

## id: strands-interrupts
### Human approval in Strands is an interrupt

A Strands tool or hook can raise an interrupt. The agent stops, returns the interrupt
to the caller, and resumes when the caller answers it. With a session manager the
paused state survives a process restart, so a different process can resume the run.
Resuming can re-enter the tool, so an irreversible action must be deduplicated where
its effect is written, not only by the checkpoint.

## id: agentcore-overview
### AgentCore is the agent platform, framework-agnostic

Amazon Bedrock AgentCore entered preview in July 2025 and became generally available
on 2025-10-13. It provides separate building blocks — Runtime, Memory, Gateway,
Identity, Code Interpreter, Browser, Observability, Policy and Evaluations — that work
with any agent framework and any model, including models outside Bedrock.

## id: agentcore-runtime
### AgentCore Runtime hosts agent code

AgentCore Runtime runs an agent or tool in an isolated microVM per session, for up to
eight hours, and speaks HTTP, MCP, A2A and AG-UI. A Python agent becomes deployable by
wrapping it in `BedrockAgentCoreApp` with an entrypoint; locally the same app serves
`POST /invocations` and `GET /ping` on port 8080, which is the contract the managed
runtime calls.

## id: agentcore-harness
### The harness is a managed agent defined by configuration

The AgentCore harness became generally available on 2026-06-17. You declare the
model, system prompt, tools and memory with `CreateHarness` and run it with
`InvokeHarness`; there is no orchestration code and no container. It can export the
agent as Strands code when you outgrow configuration. It is AWS's recommended
destination for Agents Classic workloads.

## id: agentcore-memory
### Memory has a short-term and a long-term half

AgentCore Memory stores conversation events as short-term memory and extracts
long-term records with strategies: semantic facts, summaries, user preferences,
episodic memory or a self-managed strategy. Records are scoped by actor and session
namespaces. Strands can use it as its session manager.

## id: agentcore-gateway
### Gateway turns APIs into MCP tools

AgentCore Gateway exposes Lambda functions, OpenAPI and Smithy APIs, API Gateway
stages, existing MCP servers, HTTP endpoints and model providers as tools over the
Model Context Protocol, with inbound authentication, rate limits and outbound
credentials handled by the gateway rather than the agent.

## id: agentcore-policy
### Policy is enforced outside the agent

Policy in AgentCore became generally available on 2026-03-03. Policies written in
Cedar, or authored in natural language and compiled to Cedar, are evaluated at the
gateway on every tool call before the call is allowed. Because the check runs outside
the agent's code, a model cannot reason its way around it. Since mid-2026 Policy can
also run Bedrock Guardrails on tool inputs and outputs.

## id: agentcore-identity
### Agents need their own identity

AgentCore Identity gives each agent a workload identity, validates inbound callers
(IAM or JWT), and manages outbound credentials through OAuth2 and API-key credential
providers stored in a token vault. On-behalf-of token exchange lets an agent act for
an authenticated user without repeated consent prompts.

## id: agentcore-tools
### Built-in tools: code, browser, web search

AgentCore Code Interpreter runs Python, JavaScript and Node.js in a sandbox. AgentCore
Browser gives an agent a managed Chrome session. Web Search, generally available in
mid-2026, is a managed search tool exposed as a Gateway connector so that queries
stay inside AWS. Payments, in preview since 2026-05, lets an agent pay for APIs and
content within spending rules.

## id: agentcore-evaluations
### Evaluations measure agents, not just answers

AgentCore Evaluations became generally available on 2026-03-31. It scores agent
sessions and traces with built-in, custom code-based and third-party evaluators, both
online on production traffic and in batch. The optimization loop adds recommendations
for prompts and tool descriptions, batch evaluation and A/B tests.

## id: agentcore-observability
### Observability is OpenTelemetry

Strands and the AgentCore SDK emit OpenTelemetry spans for agent invocations, model
calls and tool calls. AgentCore Observability shows them in CloudWatch as traces and
trajectories. Nothing is exported unless an exporter is configured.

## id: agent-registry
### The Agent Registry is a governed catalog

AWS Agent Registry became generally available in August 2026 under its own
`agent-registry` API namespace. It catalogs agents, tools, skills, MCP servers and
custom resources, can auto-discover AgentCore runtimes and gateways across an AWS
Organization, and exposes search through APIs and an MCP endpoint.

## id: protocols
### MCP and A2A are the two protocols

The Model Context Protocol connects an agent to tools and data. The Agent2Agent
protocol connects an agent to another agent: an agent card describes it, and tasks
carry the work. AgentCore Runtime hosts both; Strands consumes MCP tools and can serve
or call A2A agents.

---

## id: knowledge-bases
### Knowledge Bases run retrieval for you

Bedrock Knowledge Bases ingest documents, chunk and embed them, store the vectors and
answer `Retrieve` or `RetrieveAndGenerate` requests with citations. With a standard
knowledge base you choose the vector store; with a fully managed knowledge base
(2026-03-02; Managed Knowledge Base generally available 2026-06-17) Bedrock operates
the store, search is always hybrid, and agentic retrieval can plan multi-step queries
across knowledge bases.

## id: kendra-status
### Kendra is in maintenance

Amazon Kendra entered maintenance on 2026-06-30 and closed to new customers on
2026-07-30. AWS recommends the Bedrock Managed Knowledge Base for new search and RAG
applications. Query suggestions, faceted search, custom synonyms, spell checking and
incremental learning have no direct equivalent and need workarounds.

## id: s3-vectors
### S3 Vectors stores embeddings in S3

Amazon S3 Vectors became generally available on 2025-12-02. A vector bucket holds
indexes of up to two billion vectors; `PutVectors` writes and `QueryVectors` searches
by similarity with metadata filters. It is a supported vector store for Bedrock
Knowledge Bases and suits large, infrequently queried corpora.

## id: retrieved-text-untrusted
### Retrieved text is data, not instructions

A document, a tool result or another agent's reply re-enters the prompt as text the
model may obey. A retrieval pipeline must treat retrieved text as untrusted data:
quote it, cite it, never let it change permissions or tool policy, and keep
irreversible actions behind an approval that retrieved text cannot grant.

## id: ai-services-status
### Task-specific AI services: mostly active, partly in maintenance

Textract, Transcribe, Polly and Translate are active. Comprehend Topic Modeling, Event
Detection and Prompt Safety Classification, and Rekognition Streaming Events and Batch
Image Content Moderation, entered maintenance on 2026-03-31. Amazon Forecast has been
in maintenance since 2024-07-29. Amazon Fraud Detector and Lookout for Equipment reach
end of support on 2026-10-07.

---

## id: sagemaker-ai
### SageMaker AI is for your own models

Amazon SageMaker was renamed SageMaker AI in December 2024. It trains, tunes and hosts
models you own, with JumpStart, HyperPod and real-time endpoints. Eight features
entered maintenance on 2026-06-30: A2I, Clarify, Debugger, GeoSpatial, Ground Truth,
Model Monitor, Role Manager and Studio Lab. Profiler reaches end of support on
2027-06-30.

## id: assistants-lineage
### Assistants: Q Business became Quick, Q Developer became Kiro

Amazon Q Business entered maintenance on 2026-06-30; its successor is Amazon Quick.
Amazon CodeWhisperer became Amazon Q Developer in April 2024; Q Developer IDE plugins
stopped taking new sign-ups on 2026-05-15 and reach end of support on 2027-04-30,
with Kiro as the successor. Q Developer in the AWS console is not affected.

---

## id: localstack-licensing
### LocalStack needs an auth token since March 2026

On 2026-03-23 LocalStack merged its community and pro images into one image that
requires `LOCALSTACK_AUTH_TOKEN`. The Hobby plan is free for non-commercial use;
Base, Ultimate and Enterprise are paid. The token's plan decides which services
start.

## id: localstack-ai
### What LocalStack emulates for AI

LocalStack emulates Bedrock `Converse` and `InvokeModel` by serving Ollama models
(Ultimate plan), SageMaker AI endpoints and Textract (Ultimate), and Transcribe with
offline Vosk models (Hobby). S3, Lambda, DynamoDB and OpenSearch run on Hobby. It does
not emulate AgentCore, Agents Classic, Knowledge Bases, Guardrails or S3 Vectors.

---

## id: scenario-pilot
### The workshop scenario

A team must cost a six-week customer-support pilot against a budget of 25,000 USD and
must not accept the proposal without a named human approval. The approval is bound to
one version of the proposal; a changed proposal needs a new approval, and accepting
the same proposal twice must not create a second commitment.
