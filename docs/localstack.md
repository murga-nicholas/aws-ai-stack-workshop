# LocalStack — running AWS AI locally, honestly

LocalStack emulates AWS APIs in a Docker container on your machine. This page says
what it emulates for AI work, on which plan, how to run the workshop against it, and —
just as important — what it does **not** emulate. Read on **2026-10-05**.

When a demo runs against LocalStack its result says `mode: local_emulator`. That
means the emulator answered on `localhost`. It never means AWS answered.

---

## 1. What changed in March 2026

On **2026-03-23** (release `2026.03.0`) LocalStack merged its Community and Pro images
into **one image that requires an auth token**. The token's plan decides which
services start.

| Plan | Price | Use |
|---|---|---|
| **Hobby** | Free | Non-commercial use only |
| **Base** | Paid | Teams, simple applications |
| **Ultimate** | Paid (45-day trial) | Includes the AI services below |
| **Enterprise** | Custom | |

Versions are calendar-based (`2026.09.0` was current on the workshop date). A new
single-binary CLI, `lstk`, sits beside the Python `localstack` CLI; you do not need
either for this workshop — Docker Compose is enough.

If you use LocalStack at work, the Hobby plan's terms do not cover you. Use a paid
plan or the offline lane of this repository.

---

## 2. What is emulated for AI, and on which plan

| Service | Plan | Coverage | What the workshop uses |
|---|---|---|---|
| Bedrock (`bedrock`) | Ultimate | 6 of 108 operations | `ListFoundationModels` |
| Bedrock Runtime | Ultimate | 3 of 11: `Converse`, `InvokeModel`, `InvokeModelWithBidirectionalStream` | `Converse`, `InvokeModel` |
| SageMaker AI | Ultimate | 120 of 404 | `ListEndpoints` |
| Textract | Ultimate | 5 of 25 | `DetectDocumentText` |
| Transcribe | **Hobby** | 16 of 43, with offline Vosk speech models | `StartTranscriptionJob` |
| OpenSearch | **Hobby** | 14 of 96 | — |
| S3, Lambda, DynamoDB | **Hobby** | most operations | the decision demo's approval sink and tool back end |

Bedrock emulation is served by **Ollama**: a request for any Ollama model is written
as `ollama.<name>` (for example `ollama.qwen2.5:0.5b`). Only text models are supported,
and LocalStack does not implement `ConverseStream` or `CountTokens`, so the
`bedrock-runtime` demo reports those two operations as `not_supported_by_emulator`.

Relevant settings (in `docker-compose.localstack.yml`):

| Variable | Meaning |
|---|---|
| `LOCALSTACK_AUTH_TOKEN` | Required. Read from your `.env` |
| `DEFAULT_BEDROCK_MODEL` | The Ollama model used when a request names a Bedrock model id |
| `BEDROCK_PREWARM` | Start the model at container start instead of on the first request |
| `BEDROCK_PULL_MODELS` | Extra Ollama models to download at start |

---

## 3. What is **not** emulated

**AgentCore** (LocalStack closed the feature request as not planned), **Agents
Classic**, **Knowledge Bases**, **Guardrails**, **S3 Vectors**, **Comprehend** and
**Kendra**.

The honest local story for AgentCore is the SDK itself: `BedrockAgentCoreApp` serves
the same `POST /invocations` and `GET /ping` contract on `localhost` that AgentCore
Runtime calls in the cloud. `uv run awsai-demo agentcore-runtime` does exactly that,
and labels it `local_execution`, not `local_emulator` and not live.

---

## 4. Running the workshop against LocalStack

1. Create a free account at localstack.cloud and copy your auth token into `.env`:

   ```dotenv
   LOCALSTACK_AUTH_TOKEN=ls-...
   ```

2. Start the emulator:

   ```bash
   docker compose -f docker-compose.localstack.yml up -d
   ```

3. Check what your plan started:

   ```bash
   uv run awsai-demo localstack --execution emulator
   ```

   ```bash
   uv run awsai-demo localstack-ai --execution emulator
   ```

   `localstack-ai` prints a coverage table with three columns that are deliberately
   different: what LocalStack **advertises**, what the demo **attempted**, and what
   **passed**. A service your plan lacks shows `entitlement_missing`.

4. Run the centrepiece with its approval sink in emulated DynamoDB:

   ```bash
   uv run awsai-demo decision --execution emulator
   ```

   The second conditional `PutItem` for the same business action is refused by the
   emulator itself.

5. Stop it when you are done:

   ```bash
   docker compose -f docker-compose.localstack.yml down
   ```

Without a token, emulator runs stop before any call with mode `not_run` and error
code `missing_configuration`; `localstack` and `strands-agent` name
`LOCALSTACK_AUTH_TOKEN` in their message. If the container is not
running, they report `attempt_failed` / `emulator_unavailable`. Neither ever falls back
to the offline lane silently.

---

## 5. Inside the emulator lane

- Credentials are the fixed values `test` / `test`; your real AWS credentials are
  **never** loaded in this lane.
- The only network destination allowed is `localhost:4566`.
- Writes into LocalStack are recorded as `effect: write` with
  `execution_target: emulator`. LocalStack state is disposable.

---

## 6. Alternatives you may hear about

`moto` mocks AWS inside the Python process and is excellent for unit tests; this
repository uses botocore's own `Stubber` instead, because it validates each request
against the real service model. Newer open-source emulators exist; check their
coverage of the AI APIs before you rely on them, exactly as this page does for
LocalStack.
