# Worked answers

Open this only after writing your own. These are good answers, not the only ones.

---

## 1.1 Read a result before you read the code

- **Ran for real:** the Strands `Agent` loop, the `price_pilot` tool, the callback
  handler and the validation of the structured output. Their operations are recorded
  with `mode: local_execution`.
- **Synthetic:** every model response. The operations for model calls have
  `mode: local_contract` and a `fixture_id` beginning `strands-script-`.
- **Why the top level is `local_contract`:** the result takes the *weakest* mode of
  its operations. A real framework running around a scripted model proves the
  framework and your tools, not the model's behaviour, so the honest summary is the
  contract.

## 1.2 Draw the request path

User → **Identity** (inbound IAM or JWT, `aws_service`) → **Runtime** hosting your
**Strands** agent (loop: `framework`; your prompt and tools: `application`) →
**Gateway** (`aws_service`) → **Policy** evaluates Cedar (`aws_service`, rule written
by the `application`) → tool → back to the agent → **Bedrock** model through
`global.anthropic.claude-haiku-4-5-20251001-v1:0` (`aws_service`) → answer to the user.

The sentence about data: a `global.` profile may process the request in any Region
AWS routes it to; a regulated workload uses a geographic (`us.`, `eu.`) profile or a
single-Region model id instead.

## 2.1 Read the responsibility matrix

Moving *state persistence* and *deduplication* to `application` means you write and
test them — and it is the only way to *prove* the business action ran once, which an
auditor may require. Moving *orchestration loop* to `application` means you can add
custom steps the harness cannot express. Moving *model choice* is neutral: all three
lanes let you choose.

## 2.2 Make a recommendation

A reasonable answer: "Use the AgentCore harness. Expose the transactions REST API as
an OpenAPI target on AgentCore Gateway, with a Cedar policy that permits read-only
calls. Model the send step as an inline function tool, so the harness stops and the
supervisor approves in our application before anything is sent. Memory stays
short-term only, scoped per complaint. We give up custom orchestration and
stage-specific prompts; if the bank needs a provable exactly-once send, we move that
step to Strands on Runtime with an idempotent sink."

## 3.1 Check before you build

- **Kendra**: closed to new customers since 2026-07-30; creating an index fails for a
  new account. Successor: **Bedrock Managed Knowledge Base**.
- **Flow agent node**: the node calls an **Agents Classic** agent; `CreateAgent`
  returns `AccessDeniedException` (HTTP 403, "Bedrock Agents is in Maintenance Mode").
  Successor: call an AgentCore harness or Runtime agent from a Lambda node.

## 3.2 Map an agent

| Classic | Harness |
|---|---|
| Action group `price_pilot` (Lambda) | Gateway target exposed as an MCP tool, or a code-level tool |
| Return of control for `accept_proposal` | Inline function tool: stream ends with `toolUse`; client sends `toolResult` |
| Knowledge base on the agent | Knowledge base behind the Gateway |
| Session memory | AgentCore Memory, short-term, scoped by actor |
| Guardrail on the agent | Guardrail on the model + Policy at the Gateway |

Not replicated: **stage-specific prompt overrides** (pre-processing, orchestration,
knowledge-base response, post-processing). The harness has one system prompt.

## 4.1 Three lanes, one command

| Lane | Typical result | Proves | Does not prove |
|---|---|---|---|
| offline | `local_contract`, `ok` | The request is valid for the installed botocore service model; the application logic works | Permissions, the model's answer, latency |
| emulator, no token | `not_run`, `blocked`, `missing_configuration` | The prerequisite is missing, before any call | Anything about the emulator |
| emulator, Ultimate token | `local_emulator`, `ok` | The code works against an AWS-compatible API locally | That AWS behaves the same |
| live, authorised | `live_model`, `ok` | AWS answered; token usage is real | That every Region and model behaves the same |
| live, unauthorised | `live_service`, `blocked`, `authorization_denied` | Authentication worked and authorisation did not | — |

`not_run` is correct because the demo refused to pretend: the alternative would be a
silent fallback that prints a green result for something that did not happen.

## 4.2 Break the gate on purpose

The deliberately failing case is named in the output (`failed_case`). It fails the
evaluator that checks *no action before approval*: the scripted run tries to accept
the proposal in the same turn it prices it. The fix is in the agent, not the
evaluator: make `accept_proposal` raise an interrupt that only an approval can answer,
as the `decision` demo's Strands lane does, so the action cannot run in the pricing
turn.
