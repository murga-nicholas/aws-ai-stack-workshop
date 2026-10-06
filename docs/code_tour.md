# Code tour — how to read this repository

The source is about 26,000 lines, but you never need to read most of it. Every demo
module has the same anatomy, and in each one only the first screen or two teaches the
AWS technology; the rest is plumbing that keeps the three execution lanes honest. This
page tells you what to read first and what you can skip.

---

## 1. The anatomy of a demo module

Open any `src/awsai_demo/<name>_demo.py` and you will find four layers, usually in this
order:

| Layer | How to recognise it | Read it? |
|---|---|---|
| **Teaching functions** | Public, near the top: `price_pilot`, `build_agent`, `translate_filter`, `evaluate_accept_proposal`, `wrap_untrusted_passages` | **Yes — this is the technology** |
| **Request builders** | Public functions named `*_params` or `*_request` that return a plain `dict` | **Yes — this is the exact AWS request**, the same dictionary the offline lane validates against botocore's service model and the live lane sends |
| **The entry point** | One `run_<name>_demo(...)` function | Skim: it picks the lane and assembles the result |
| **Lane plumbing** | Private helpers (`_run_offline`, `_run_live_…`, `_emulator_…`, error mapping) | Only when you need to know *how* a lane is made honest |

Two facts make this work:

1. **A request builder is pure.** `converse_text_params(...)` returns the dictionary a
   `bedrock-runtime` client would send. Offline, botocore's `Stubber` checks that
   dictionary against the real API model; live, the same dictionary is sent. So reading
   a builder tells you the real AWS call, with no AWS account.
2. **Lanes never fall back to a fixture.** If nothing in the requested lane runs, the
   mode is `not_run` and the operations say why. If a live read such as `CountTokens`
   runs and the billable call is then refused, the mode stays `live_service` and the
   status is `blocked`. `contracts.py` rejects any mode that overstates what ran.

The code panels on the slides are cut from exactly these teaching functions and
builders, between `# slide: <marker>` comments.

---

## 2. The shared core, in reading order

Read these once; every demo uses them.

| Order | File | What to take from it |
|---|---|---|
| 1 | `scenario.py` | The business task and its numbers: `price_pilot()`, `proposal_version()`, `action_key()`. About 100 lines. |
| 2 | `contracts.py` | The result envelope: `mode`, `status`, `OperationOutcome`, the error codes, and the checks that raise when a result overstates what ran. |
| 3 | `registry.py` | The thirty demos in one table. |
| 4 | `cli.py` | How a command becomes a `run_*_demo` call; lane flags; safe error output. |
| 5 | `stubs.py` and `offline.py` | How the offline lane works: botocore `Stubber` with `expected_params`, and `ScriptedModel`, a real Strands `Model` with a fixed script. |
| 6 | `credentials.py` and `runtime.py` | The credential order (`--profile`, `AWS_PROFILE`, `AWS_CREDS_FILE_PATH`, default chain) and `.env` loading. |
| 7 | `policy.py` and `pricing.py` | Budgets: count tokens, reserve a price **before** the call, refuse what cannot be priced. |
| 8 | `manifest.py` | Creating AWS resources safely: intent recorded before the request, cleanup only by evidence. |
| 9 | `redact.py` and `network.py` | No secrets in output; no network in the offline lane. |

You can stop after step 5 and understand every offline demo.

---

## 3. What to read first in each demo

★ marks the demos presented in the talk.

### Models

| Demo | Read first |
|---|---|
| ★ `bedrock_runtime_demo.py` | `converse_text_params`, `converse_tool_params`, `invoke_model_params` — the same request, two API shapes |
| `bedrock_openai_demo.py` | `bedrock_openai_base_url`, `generate_bedrock_api_key`, `chat_completion_request` |
| ★ `model_lifecycle_demo.py` | `get_foundation_model_params`, then how `run_model_lifecycle_demo` counts `ACTIVE` and `LEGACY` |
| `guardrails_demo.py` | `create_guardrail_params`, `apply_guardrail_params` |
| `bedrock_flows_demo.py` | `create_flow_params`, `invoke_flow_params` |
| `nova_act_demo.py` | `create_workflow_definition_params` |
| `model_customization_demo.py` | `customization_job_params` (three customization types) |

### Agents

| Demo | Read first |
|---|---|
| `agents_classic_demo.py` | `invoke_agent_params`, `return_control_result_params`, `classify_access_denied` |
| ★ `strands_agent_demo.py` | `price_pilot` (a tool), `build_agent` (the agent) — twenty lines that are the whole idea |
| `strands_multiagent_demo.py` | `run_local_workflows` (graph and swarm) |
| `agentcore_runtime_demo.py` | `build_app` — `BedrockAgentCoreApp` with an entrypoint |
| ★ `agentcore_harness_demo.py` | `create_harness_params`, `inline_function_tool`, `build_harness_resume_request` |
| `agentcore_memory_demo.py` | `create_memory_params`, `create_event_params` |
| ★ `agentcore_gateway_demo.py` | `evaluate_accept_proposal` (the Cedar decision), `run_mcp_round_trip` |
| `agentcore_identity_demo.py` | `create_oauth2_credential_provider_request`, `get_resource_oauth2_token_request` |
| `agentcore_tools_demo.py` | `interpreter_code`, `start_request`, `invoke_request` |
| `mcp_demo.py` | the whole file (57 lines) |
| `a2a_demo.py` | `build_a2a_server`, `exchange` |
| ★ `decision_demo.py` | `acceptance_tool` (the interrupt), `apply_approval` (the idempotent sink), `pause_decision`, `resume_from_disk`; then `resume_worker.py`, the separate process |

### Data

| Demo | Read first |
|---|---|
| ★ `knowledge_bases_demo.py` | `build_local_index`, `retrieve`, `wrap_untrusted_passages`, `answer_with_retrieval` |
| `kendra_demo.py` | `translate_filter` |
| `s3_vectors_demo.py` | `cosine_search` |
| `ai_services_demo.py` | `service_rows` |

### Platform, operations, local

| Demo | Read first |
|---|---|
| `sagemaker_ai_demo.py` | the whole file (about 100 lines) |
| `aws_identity_demo.py` | `principal_type`, then `run_aws_identity_demo` |
| `evaluation_demo.py` | `evaluate_case` |
| `observability_demo.py` | `capture_spans`, `span_tree` |
| `cost_demo.py` | `run_cost_demo` |
| ★ `localstack_demo.py` | `approval_put_item_params` (the conditional write), `create_function_params` |
| `localstack_ai_demo.py` | `advertised_coverage` |

---

## 4. Following one request end to end

A good first exercise: trace `uv run awsai-demo strands-agent --execution live`.

1. `cli.py` parses the command and calls `run_strands_agent_demo(execution="live", …)`.
2. `credentials.py` picks one credential source and never switches.
3. `providers.py` builds a `BedrockModel` for the tested default model.
4. Before each model call, `policy.py` counts the input with `CountTokens`, prices it
   from `data/pricing_snapshot.json` and reserves the cost; a call that would pass the
   budget is refused.
5. Strands runs the loop: the model asks for `price_pilot`, the tool runs locally, the
   model answers as a `PilotSummary`.
6. Each call becomes an `OperationOutcome`; `contracts.result()` checks that the
   evidence matches the claimed mode; `redact.py` cleans the output.

Then run the same command without `--execution live` and compare the two results'
`operations`: the framework and tool rows (`Agent.__call__`, `price_pilot`) are
identical; the model rows change from `live_model` to `local_contract`, and the live
run adds one `CountTokens` row per model call — the budget check.

---

## 5. Tests

Every module has `tests/test_<module>.py`. Tests replace an external call by injecting
a port (a small fake for the one call a demo makes) or by using botocore's `Stubber`;
`monkeypatch` is reserved for the standard library and SDK entry points (clocks, package
metadata, an SDK client class), never for the repository's own functions. To see how a demo behaves on a 403, a timeout or a missing price,
read its test file: each failure the demo handles has a named test.
