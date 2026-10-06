# Exercises — the four competence gates

Reading does not pass a gate. Producing something does.

Each exercise has a **task**, a **done when** you can check yourself, and a worked
answer in [`answers.md`](answers.md). Write yours before you open it. Work them in
order; later ones assume earlier ones. Everything runs offline unless the exercise
says otherwise.

---

## Gate 1 — Explain

*You can trace one request through Bedrock, a Strands agent and AgentCore, and name
who owns each step.*

### 1.1 Read a result before you read the code

```bash
uv run awsai-demo strands-agent --format json
```

**Task.** From the JSON alone, answer: which operations ran for real on your machine?
Which response was synthetic, and what is its fixture id? Why is the top-level mode
`local_contract` even though the Strands loop really ran?

**Done when** each answer points at a field in the JSON.

### 1.2 Draw the request path

**Task.** Draw a user request that reaches a Strands agent on AgentCore Runtime,
calls one tool through AgentCore Gateway with a Cedar policy, calls a Bedrock model
through a `global.` inference profile, and returns. Label every arrow `application`,
`framework` or `aws_service`.

**Done when** your drawing has an Identity check on the way in, a Policy check before
the tool, and you can say in one sentence where the model's data may be processed.

---

## Gate 2 — Choose

*You can pick between the harness, a code-defined agent and Agents Classic, and
defend it.*

### 2.1 Read the responsibility matrix

```bash
uv run awsai-demo decision --simulate-approval
```

**Task.** For each row of the matrix, write one sentence on what changes for your team
when the cell moves from `aws_service` to `application`.

**Done when** you have named at least one row where owning it is an advantage, not a
cost.

### 2.2 Make a recommendation

**Task.** A bank wants an internal agent that drafts replies to customer complaints,
looks up the customer's last three transactions through an existing REST API, and must
never send anything without a supervisor's approval. Recommend the harness or Strands
on Runtime, in five sentences or fewer.

**Done when** your recommendation says how the approval is enforced, where the REST
API is exposed to the agent, and one thing you give up.

---

## Gate 3 — Migrate

*You can spot a service in maintenance mode and plan the move.*

### 3.1 Check before you build

```bash
uv run awsai-demo lineage --status maintenance
```

**Task.** A colleague's design uses Amazon Kendra for search and a Bedrock Flow with
an agent node. Their AWS account is new. List what will fail on day one, with the
error they will see, and the successor for each.

**Done when** you have named both problems and both successors.

### 3.2 Map an agent

```bash
uv run awsai-demo agents-classic --format json
```

**Task.** Using the Agents Classic fixture in that output, write its harness
equivalent as a table: action group, return of control, knowledge base, memory,
guardrail.

**Done when** every row has a destination and you have flagged the one Classic feature
the harness does not replicate.

---

## Gate 4 — Run honestly

*You can run a demo offline, on LocalStack or live, and say which one you did.*

### 4.1 Three lanes, one command

```bash
uv run awsai-demo bedrock-runtime
```

```bash
uv run awsai-demo bedrock-runtime --execution emulator
```

```bash
uv run awsai-demo bedrock-runtime --execution live
```

**Task.** For each run, record `mode`, `status` and the `error.code` if any. Write
what each run proves and what it does not.

**Done when** you can explain why a `not_run` result is a correct answer and not a
failure of the demo.

### 4.2 Break the gate on purpose

```bash
uv run awsai-demo evaluation
```

**Task.** One case fails on purpose. Find it, explain which evaluator caught it, and
describe the change to the agent that would make it pass without weakening the
evaluator.

**Done when** your change keeps the evaluator exactly as strict as it is.
