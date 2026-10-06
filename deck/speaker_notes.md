# Speaker notes - AWS AI stack workshop

The presenter script for `aws_ai_stack.pptx`. The deck build copies each section into
that slide's presenter notes, so the script and the slides cannot drift apart: a slide
without a section, or a section without a slide, fails the build.

How to read it:

- Times follow the agenda on slide 3: 44 presented slides in 60 minutes, the last
  five of them for questions. The appendix (slides 45-55) is not presented; its notes
  are for questions and for readers.
- "Say" is the script. "Show" tells you where to point. "Demo" is an optional live
  command; skip it if you are behind. "If asked" holds facts for questions. "Next" is
  the bridge to the following slide.
- The audience is mixed: sales, managers, architects and Python developers. Every
  slide works at two levels: the headline for everyone, the table, diagram or code
  for the technical half.
- Every demo runs offline by default and prints its mode. Say the mode out loud when
  you run one. It is what stops a local demonstration from sounding like a cloud call.
- Numbers on the slides come from `deck/facts.json`, recorded by real runs. If the
  slide and this script disagree, the slide is right; fix the script.

<!-- slide: open.title -->
### 1. Cover - 0:00-0:45

Say:
I am Mykola Murha, a Data and AI Engineer at DataArt. For the next hour we look at the
AWS AI stack as it stands in October 2026: Amazon Bedrock for models, Strands for
writing agents, Bedrock AgentCore for running and governing them, MCP and A2A for
connecting them, and LocalStack for running some of it on a laptop.

One promise, and the repository keeps it for me: every command I run prints how its
result was produced. Offline, emulator or live. Nothing simulated is presented as a
cloud call.

Next:
Here is what you should be able to do at the end.

<!-- slide: open.promise -->
### 2. Four things you can do after this hour - 0:45-2:00

Say:
Four skills, and each one has a command behind it.

Explain: trace one request from Bedrock through a Strands agent and AgentCore, and
name who owns each step. Choose: decide between the AgentCore harness, a code-defined
agent and the older Agents Classic, and defend the choice. Migrate: notice that a
service is in maintenance mode before you build on it. Run honestly: run any demo
offline, against LocalStack or live, and say which one you did.

Show:
The right-hand column. All four commands work tonight with no AWS account.

If asked:
- "Do I need an AWS account?" Not for any of the offline runs. Live runs need
  credentials and spend a few cents at most; the repo refuses a call it cannot
  bound.

Next:
The route through the hour.

<!-- slide: open.agenda -->
### 3. Agenda - 2:00-3:00

Say:
Eight stops. Eight minutes of lineage first, because the names changed a lot. Then
Bedrock for models, Strands for agent code, and the longest stop, AgentCore. Then one
business decision built three ways. Then data, LocalStack, and how to choose. Five
minutes for questions at the end.

Show:
The command column: one command per segment.

Next:
Section one: where today's names came from.

<!-- slide: lineage.section -->
### 4. Section 01 - Lineage - 3:00-3:20

Say:
If you copy a tutorial from 2024 you may be copying a service your account can no
longer use. So we read the lineage first.

<!-- slide: lineage.stack_today -->
### 5. The stack today - 3:20-4:40

Say:
This is the whole stack on one slide, top to bottom. You write agents in code with
Strands or any other framework. You run and govern them with AgentCore: runtime,
harness, memory, gateway and policy, identity, tools, observability and evaluation.
Underneath, Bedrock serves the models and the guardrails. On the right of the data
layer, managed knowledge bases and S3 Vectors. At the bottom, local development.

Two boxes are amber. Agents Classic and Kendra are in maintenance mode. Keep them in
mind; we come back to them in two minutes.

Show:
Point at the two amber boxes.

Demo:
`uv run awsai-demo list` - thirty demos, one per technology on this map.

Next:
How we got here.

<!-- slide: lineage.timeline -->
### 6. Two waves - 4:40-6:00

Say:
AWS built this in two waves. In 2023 the services were finished products: Bedrock
Agents ran the reasoning loop, Knowledge Bases ran retrieval, and you configured them.
From 2025 AWS shipped parts instead: Strands as an open-source framework, then
AgentCore as separate building blocks that work with any framework and any model.

In June 2026 the harness arrived: a configuration-defined agent on top of AgentCore.
Six weeks later, on 30 July, Agents Classic, Kendra and Q Business closed to new
customers.

Show:
The amber callout. Most tutorials online predate that date.

If asked:
- Dates: Bedrock GA 2023-09-28; Agents and Knowledge Bases GA 2023-11-28; AgentCore GA
  2025-10-13; harness GA 2026-06-17 at AWS Summit New York.

Next:
The agent lineage in detail.

<!-- slide: lineage.agents -->
### 7. Agents lineage - 6:00-7:20

Say:
Read each row left to right: the old name, the relationship, the successor, the status.

Agents for Amazon Bedrock is now called Agents Classic. AWS recommends two successors:
the AgentCore harness if you want configuration, or a code-defined agent on AgentCore
Runtime if you want control. Classic's multi-agent collaboration moves to framework
code, such as Strands graphs and swarms. Two package renames for the record: the AWS
Labs multi-agent orchestrator is now Agent Squad, and the AgentCore starter toolkit
has been overtaken by the AgentCore CLI.

Demo:
`uv run awsai-demo agents-classic` - the return-of-control contract offline.

If asked:
- "Is Agent Squad the same as Strands?" No. Agent Squad is a community framework from
  AWS Labs; Strands is the framework AWS uses itself.

Next:
Search and assistants.

<!-- slide: lineage.retrieval_assistants -->
### 8. Retrieval and assistants lineage - 7:20-8:30

Say:
Kendra, AWS's enterprise search, is in maintenance; AWS points Kendra customers at the
Bedrock Managed Knowledge Base. Amazon Q Business, the employee assistant, is in
maintenance too; its successor is Amazon Quick. On the developer side, CodeWhisperer
became Q Developer in 2024, and the Q Developer IDE plugins reach end of support on
30 April 2027, with Kiro as the successor. Bedrock Studio moved into SageMaker Unified
Studio in 2025.

Show:
The status chips: maintenance, sunset, renamed, moved. Four different words, four
different consequences.

If asked:
- Q Developer in the AWS console is not affected; only the IDE plugins and paid
  subscriptions are.

Next:
Models have a lifecycle too.

<!-- slide: lineage.models -->
### 9. Models and endpoints - 8:30-9:40

Say:
Every Bedrock model is Active, Legacy or End-of-Life. Legacy is the one people miss: a
new customer cannot start using the model, and an existing one can lose access after
fifteen idle days. At End-of-Life the calls simply fail.

Amazon's own Titan text models gave way to Nova and then Nova 2. And endpoints change
too: Bedrock's OpenAI-compatible endpoint launched as bedrock-mantle, and since
September AWS recommends bedrock-runtime instead.

Show:
The warning: Claude Sonnet 4 reaches end-of-life on 14 October.

Demo:
`uv run awsai-demo model-lifecycle` - reads the lifecycle field.

Next:
What maintenance mode really means.

<!-- slide: lineage.maintenance_mode -->
### 10. Maintenance mode - 9:40-11:00

Say:
Maintenance mode is a closed door with the lights on. If you already use the service,
everything keeps working: APIs, templates, bug fixes. If your account is new, you get
an HTTP 403. For Agents Classic the message literally says "Bedrock Agents is in
Maintenance Mode", and AWS has no exception process.

The table is generated from the same data file the CLI reads, so the slide and the
command cannot disagree. And the footnote is not theory: on the account this workshop
was built with, a plain Kendra ListIndices call came back with
SubscriptionRequiredException. That is the closed door, observed.

Demo:
`uv run awsai-demo lineage --status maintenance`

If asked:
- "When will Agents Classic be switched off?" AWS has published no end-of-life date.
  Maintenance is not sunset; sunset has a date.

Next:
Section two: Bedrock.

<!-- slide: bedrock.section -->
### 11. Section 02 - Bedrock - 11:00-11:15

Say:
Calling a model is the easy part. Choosing the API, the route and a model that will
still exist next year is the work.

<!-- slide: bedrock.converse -->
### 12. Converse - 11:15-12:45

Say:
Bedrock has two native ways to call a model. InvokeModel is the original: its request
body is whatever the provider defined, so switching from Nova to Claude means
rewriting the body. Converse is one message format for every model, with tools,
documents and guardrails built in. Use Converse for new code.

The code panel is cut from the demo module at build time, so what you see is the
tested code, not a slide-only example.

Show:
The live token counts in the rail: a real call to Claude Haiku 4.5, the tested
default. It is the default because CountTokens can count its input before the call,
so the budget is reserved from a real number; Nova does not support CountTokens yet.

Demo:
`uv run awsai-demo bedrock-runtime` - offline it uses botocore's Stubber, which
checks our request against the real API model and returns a named synthetic answer.

If asked:
- Batch and asynchronous invocation use the same request shape; the repo shows their
  requests only, because a batch job cannot be bounded inside one command.

Next:
The other door into Bedrock.

<!-- slide: bedrock.openai -->
### 13. OpenAI-compatible APIs - 12:45-14:00

Say:
If your team already has OpenAI SDK code, Bedrock speaks that wire format: Chat
Completions and Responses. You change the base URL and the key. The key is a
short-term Bedrock API key minted from your AWS session, not an OpenAI key.

Show:
The warning. On most developer laptops OPENAI_API_KEY is a real OpenAI key. Pass the
Bedrock key explicitly.

If asked:
- bedrock-mantle versus bedrock-runtime: mantle came first, in December 2025; since
  15 September 2026 AWS recommends bedrock-runtime for new integrations.

Next:
How to pick the model.

<!-- slide: bedrock.catalog -->
### 14. The catalog - 14:00-15:15

Say:
The model catalog is an API, and these numbers come from a live call. Count how many
are already Legacy.

Then the model id. A plain id lives in one Region. A geographic profile, us-dot,
routes inside one geography. A global-dot profile routes anywhere, which is cheaper
and has more capacity, but it is not a residency boundary.

Show:
The three id forms in the table.

If asked:
- In the Price List data, Nova 2 Lite tokens cost 10% more in-Region than through
  global routing.

Next:
Safety.

<!-- slide: bedrock.guardrails -->
### 15. Guardrails - 15:15-16:40

Say:
Guardrails filter content, topics, words and personal data, check grounding, and run
Automated Reasoning checks against formal rules. The important design fact is
ApplyGuardrail: it checks text without calling any model. So one guardrail can screen
input and output for a model hosted anywhere.

Show:
The two results in the rail: an input with an email address, and a safe output.
These came from a live run that created a guardrail and deleted it afterwards.

If asked:
- Comprehend's prompt-safety classifier entered maintenance on 31 March 2026;
  Guardrails is where that capability lives now.

Next:
The low-code layer.

<!-- slide: bedrock.flows -->
### 16. Prompts and Flows - 16:40-18:00

Say:
Prompt management versions your prompts like code. Flows wires prompts, knowledge
bases, Lambda functions, conditions and loops into a graph you can deploy without an
agent framework. Both are active.

One trap: a Flow's agent node calls an Agents Classic agent, so a new account cannot
use that node. Call AgentCore through a Lambda node instead.

Next:
Section three: Strands.

<!-- slide: strands.section -->
### 17. Section 03 - Strands - 18:00-18:15

Say:
Strands is the framework AWS built for itself and open-sourced in May 2025. It is the
code-first road onto AgentCore.

<!-- slide: strands.agent -->
### 18. The agent loop - 18:15-20:00

Say:
A Strands agent is a model, a system prompt and tools. A tool is a Python function with
a decorator; its docstring is what the model reads. The model decides when to call
the tool. Your code decides what the tool is allowed to do.

The steps on the right are one turn: the model reads the tool specs, asks for
price_pilot, Strands runs it, and the model writes the answer as a typed
PilotSummary object.

Offline, the whole Strands loop runs for real around a scripted model. Only the
model's words are fixed.

Demo:
`uv run awsai-demo strands-agent` - mode local_contract, because the model is
scripted.

If asked:
- Strands supports Bedrock, OpenAI, Anthropic, Ollama, LiteLLM and more. The demo can
  run the same agent on OpenAI with `--provider openai --execution live`.

Next:
More than one agent.

<!-- slide: strands.multiagent -->
### 19. Multi-agent - 20:00-21:30

Say:
Three patterns, chosen by one question: who decides the next step? In a graph, you do;
the edges are code. In a swarm, the agents hand off to each other, inside a hard
limit. With an agent as a tool, a supervisor decides. That last one is the old Agents
Classic supervisor pattern, now in code.

Demo:
`uv run awsai-demo strands-multiagent`

Next:
How agents connect to the outside world.

<!-- slide: strands.protocols -->
### 20. MCP and A2A - 21:30-23:00

Say:
Two protocols, two jobs. MCP connects an agent to tools and data. A2A connects an
agent to another agent: the other agent publishes a card, and you send it tasks. Both
run on this laptop in the demo, MCP over standard input and output, A2A over
localhost HTTP.

Show:
The warning. Strands' A2A support needs a2a-sdk below 0.4; the current 1.x SDK does
not import with it. Pin the Strands extra, not the newest SDK.

Next:
Section four: AgentCore.

<!-- slide: agentcore.section -->
### 21. Section 04 - AgentCore - 23:00-23:15

Say:
AgentCore runs agents you did not have to host. Separate building blocks; use one or
all of them.

<!-- slide: agentcore.map -->
### 22. The map - 23:15-24:45

Say:
Follow a request from the left. Identity checks who is calling. The agent runs either
as your code on Runtime or as configuration in the harness. It reaches tools through
the Gateway, where Policy checks every call. Memory keeps context. Observability and
Evaluations watch it all, and the Agent Registry lets other teams find it.

Show:
The dashed arrow from Policy to Gateway: every tool call.

Next:
Runtime first.

<!-- slide: agentcore.runtime -->
### 23. Runtime - 24:45-26:15

Say:
Runtime hosts your agent in an isolated microVM per session, for up to eight hours.
The contract is two HTTP routes: POST /invocations and GET /ping. The SDK serves the
same two routes on your laptop, so the demo starts the app on localhost and calls it.

Show:
The source note. That run is the Runtime contract on localhost, not AWS. Deploying
needs a container or a zip and an IAM role, which this repo does not create for you.

Demo:
`uv run awsai-demo agentcore-runtime`

Next:
The configuration-only alternative.

<!-- slide: agentcore.harness -->
### 24. The harness - 26:15-28:00

Say:
The harness is Agents Classic rebuilt on AgentCore. You declare the model, prompt and
tools; AWS runs the loop. The left column is what Classic customers have; the right is
where each piece lands. Return of control becomes an inline function tool: the stream
stops with a toolUse block, your code answers with a toolResult.

Two honest gaps from AWS's own guide: stage-specific prompt overrides are not
replicated, and multi-agent collaboration needs code.

Demo:
`uv run awsai-demo agentcore-harness` - offline it validates the CreateHarness
request against the real API model.

If asked:
- The repo never creates a harness live. AWS bills harness session lifetime and
  CloudWatch telemetry, which a per-command budget cannot bound, so the live lane
  lists harnesses and stops there. That refusal is the budget rule working.

Next:
Memory.

<!-- slide: agentcore.memory -->
### 25. Memory - 28:00-29:15

Say:
Two halves. Short-term memory is events you write: the turns of a conversation.
Long-term memory is records that AWS extracts from those events with a strategy:
facts, summaries, user preferences, episodes. Extraction is asynchronous, so a fact
you just wrote is not readable a second later. Scope everything by actor, so one
user's memory never reaches another.

Demo:
`uv run awsai-demo agentcore-memory`

Next:
Tools and the rules around them.

<!-- slide: agentcore.gateway -->
### 26. Gateway and Policy - 29:15-31:00

Say:
The Gateway turns your APIs and Lambda functions into MCP tools. Policy sits in front
of them and evaluates a Cedar rule on every call. Here the rule says
accept_proposal is denied unless a human approved this exact version of the proposal.

The point of the green callout: the check runs outside the agent's code, so a model
cannot argue its way past it.

Demo:
`uv run awsai-demo agentcore-gateway` - a real MCP server and a real Cedar
evaluation, both local.

Next:
Identity.

<!-- slide: agentcore.identity -->
### 27. Identity - 31:00-32:00

Say:
Two directions. Inbound: who may call the agent, checked by IAM or a JWT before your
code runs. Outbound: what the agent may call, using credential providers in a token
vault, so the agent never stores a secret. On-behalf-of exchange lets it act for a
signed-in user.

Next:
Built-in tools.

<!-- slide: agentcore.tools -->
### 28. Built-in tools - 32:00-33:00

Say:
Four managed tools. Code Interpreter runs code in a sandbox, Browser gives the agent a
Chrome session, Web Search is a gateway connector whose queries stay in AWS, and
Payments, in preview, lets an agent pay within limits. Each one costs money while it
runs, so the repo runs only the Code Interpreter live, and only when you pass
`--allow-create`.

Next:
How you know it works.

<!-- slide: agentcore.observe -->
### 29. Observability and evaluation - 33:00-34:45

Say:
Observability is OpenTelemetry: every agent run, model call and tool call is a span.
The names on the slide were recorded from a real Strands run. Evaluations score runs
with built-in, code-based or third-party evaluators, online or in batch.

Show:
One case fails on purpose. A gate that never fails is not a gate.

Demo:
`uv run awsai-demo evaluation`

Next:
Finding agents.

<!-- slide: agentcore.registry -->
### 30. The registry - 34:45-36:00

Say:
The Agent Registry is a governed catalog of agents, tools, skills and MCP servers. It
went GA in August under its own API namespace and can discover AgentCore resources
across an AWS Organization. An A2A agent card is what a record points to.

Next:
Section five: one decision, three ways.

<!-- slide: decision.section -->
### 31. Section 05 - The decision - 36:00-36:15

Say:
One business task. Cost a six-week support pilot, and refuse to accept it until a
named human approves.

<!-- slide: decision.three_ways -->
### 32. Three implementations - 36:15-38:00

Say:
The pilot costs just under twenty thousand dollars against a twenty-five thousand
budget, so the numbers are not the hard part. The hard part is the pause. Lane A uses
Agents Classic's return of control. Lane B uses the harness's inline tool. Lane C is
Strands code with an interrupt.

Show:
The mode column. A and B are contracts offline; C is real framework execution.

Demo:
`uv run awsai-demo decision` - lane C stops with status paused and prints a run id.

Next:
Who owns what.

<!-- slide: decision.matrix -->
### 33. Responsibility matrix - 38:00-40:00

Say:
Read across one row at a time. Moving from Classic to the harness to code moves
responsibility from AWS to you: the loop, the persistence, the deduplication. That is
more work, and it is also the only way to prove what happened.

If asked:
- "So code is always better?" No. If nobody will ever need that proof, the harness is
  less to own.

Next:
The proof.

<!-- slide: decision.recovery -->
### 34. Recovery - 40:00-42:00

Say:
Two process ids: the one that paused and a different one that resumed from disk. One
effect after the resume, still one after the replay. The approval is bound to the
proposal's version hash, and the effect is keyed by the business action, so a
second approval cannot commit twice.

Show:
The approval source. In this recorded run it is simulated, and the output says so. A
real run takes `--approve --approver NAME`.

Next:
Section six: data.

<!-- slide: data.section -->
### 35. Section 06 - Data - 42:00-42:15

Say:
Grounding an agent in your own data: managed retrieval, vectors, and the AI services.

<!-- slide: data.knowledge_bases -->
### 36. Knowledge bases - 42:15-44:00

Say:
A standard knowledge base lets you choose the vector store and the embedding model. A
managed knowledge base runs the whole pipeline: store, parsing, hybrid search, and
agentic retrieval across several knowledge bases. This is where AWS sends Kendra
customers.

Show:
The two values in the rail. The retrieval demo plants a document containing an
instruction. It is retrieved, because it matches, and it is not obeyed, because
retrieved text is data.

Demo:
`uv run awsai-demo knowledge-bases`

Next:
Where the vectors live.

<!-- slide: data.vectors -->
### 37. Vector stores - 44:00-45:30

Say:
Choose by query rate. S3 Vectors stores billions of vectors cheaply and suits large,
infrequently queried corpora. OpenSearch is faster under constant traffic. A managed
knowledge base hides the choice entirely.

Next:
The classic AI services.

<!-- slide: data.ai_services -->
### 38. AI services - 45:30-47:00

Say:
Textract, Transcribe, Translate and Polly are active. Comprehend and Rekognition are
mostly active, with specific features in maintenance: check the feature, not the
service name. Forecast is in maintenance; Fraud Detector and Lookout for Equipment
end support on 7 October 2026.

Next:
Section seven: LocalStack.

<!-- slide: localstack.section -->
### 39. Section 07 - LocalStack - 47:00-47:15

Say:
Running AWS AI locally. An emulator answers, not AWS, and the mode says so.

<!-- slide: localstack.coverage -->
### 40. Coverage by plan - 47:15-49:30

Say:
Since March 2026 LocalStack ships one image and needs an auth token. The plan on the
token decides what starts. On the free Hobby plan you get S3, Lambda, DynamoDB,
Transcribe and OpenSearch. Bedrock, answered by local Ollama models, SageMaker and
Textract need Ultimate.

Show:
The amber line. AgentCore, Knowledge Bases, Guardrails and S3 Vectors are not
emulated at all. For AgentCore the honest local story is the SDK's own localhost
server, which we saw earlier.

Next:
Running it.

<!-- slide: localstack.demo -->
### 41. Running against the emulator - 49:30-52:00

Say:
Three steps: a token in .env, docker compose up, and the same demo command with
`--execution emulator`. The mode changes to local_emulator. In emulator mode the
decision demo writes its approval to DynamoDB with a conditional put, so the emulator
itself refuses the second write.

If asked:
- Without a token the demo returns not_run and names the missing variable; it never
  falls back to offline silently.

Next:
How to choose.

<!-- slide: close.choose -->
### 42. Choosing - 52:00-54:00

Say:
Six lines that settle most designs. Configuration and speed: the harness. Custom
orchestration or a proof of exactly-once: Strands on Runtime. Documents: a managed
knowledge base. Existing OpenAI code: the compatible APIs. Tools for many agents: the
gateway with policy. And if you run Agents Classic today: keep it, and migrate when
you next change it.

Next:
A first step for each of you.

<!-- slide: close.first_step -->
### 43. First steps - 54:00-55:00

Say:
One thing per role, each under an hour. Sales: check what the prospect runs against
the maintenance list. Managers: make the acceptance criteria name the mode.
Architects: fill in the responsibility matrix for your own case. Developers: run
everything offline, then one demo live, and compare the evidence.

Next:
Questions.

<!-- slide: close.questions -->
### 44. Questions - 55:00-60:00

Say:
Thank you. Everything I showed runs offline from the repository; `uv run awsai-demo
list` is the place to start. Questions?

<!-- slide: appendix.section -->
### 45. Appendix

Not presented. Reference slides for questions and for readers.

<!-- slide: appendix.identity -->
### 46. Authentication versus authorisation

Use when asked about credentials. Two principals both pass STS; only one may list
Bedrock models. STS GetCallerIdentity needs no permission, so its success proves
identity only. A real 403 is reported as live_service with status blocked. The repo
never prints an account id or an ARN.

<!-- slide: appendix.cost -->
### 47. Cost per successful run

Use when asked about cost. Prices come from the AWS Price List API, recorded in
`data/pricing_snapshot.json` with the date read. Retries are counted because a failed
call that consumed tokens is billed. The limits in this repo are application-enforced,
not an AWS billing ceiling.

<!-- slide: appendix.sagemaker -->
### 48. SageMaker AI

Use when asked about training your own models. SageMaker AI is still the place to
train and host them. Eight features entered maintenance in June 2026, and Profiler is
in sunset until 30 June 2027.

<!-- slide: appendix.nova_act -->
### 49. Nova Act

Use when asked about browser automation. Nova Act is a model and service trained for
UI actions, with workflow definitions and runs. AgentCore Browser is a managed Chrome
session that any model can drive.

<!-- slide: appendix.customization -->
### 50. Model customization

Use when asked about fine-tuning. Five methods, all long-running and billed by
training, so the repo shows their requests only. A model in Legacy cannot be
fine-tuned again.

<!-- slide: appendix.kendra -->
### 51. Kendra to a managed knowledge base

Use with Kendra customers. The query moves into retrievalQuery and the filter
operators are renamed. Features without a direct equivalent: query suggestions,
facets, synonyms, spell checking, incremental learning.

<!-- slide: appendix.deprecations -->
### 52. Every deprecation

The full table from `data/lineage.yaml`. Run `uv run awsai-demo lineage` for the same
records with sources and dates.

<!-- slide: appendix.repository -->
### 53. The repository

Where everything lives. The slides themselves are data in `deck/content`; the code
panels are cut from tested source at build time.

<!-- slide: appendix.readiness -->
### 54. Modes

The vocabulary every result uses. Status says what happened; mode says how it was
produced. Read the mode before you believe the result.

<!-- slide: appendix.references -->
### 55. References

All sources were read on 2026-10-05. Every title on the three cards is a live
link: Ctrl+click in PowerPoint, or click in the PDF. `docs/research.md` holds the
claim-by-claim record.
