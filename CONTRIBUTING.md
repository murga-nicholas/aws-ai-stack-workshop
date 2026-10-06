# Contributing and quality gates

This repository favours small, explicit modules over framework magic. Keep every
example easy to explain from a stage, and hold every change to the same automated
checks as production code.

## Set up

```powershell
Copy-Item .env.example .env
uv sync --frozen
```

```bash
cp .env.example .env
uv sync --frozen
```

Run every command from the repository root. `.env.example` documents every variable
the code reads and is the only one of the two files that is tracked. `uv.lock` is
authoritative: change dependencies with `uv`, and commit `pyproject.toml` and the
lockfile together.

## Required checks

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

```bash
uv run --group deck python deck/build_deck.py
```

`uv run ruff format .` and `uv run ruff check --fix .` make safe mechanical fixes.
Review the diff; automated fixes do not replace judgment.

## Who owns what

The repository was written by two authors with a fixed split, and changes keep it:

| Area | Files |
|---|---|
| Code | `src/`, `tests/`, `deck/*.py`, `scripts/`, `pyproject.toml`, `uv.lock` |
| Words and facts | `README.md`, `docs/`, `data/*.yaml`, `data/*.md`, `deck/content/`, `deck/speaker_notes.md`, `deck/facts_contract.yaml`, `deck/llms_header.md`, `llms.txt` |

**Facts and words live in data; code loads and validates them.** If you need a new
date on a slide, add it to `data/lineage.yaml` with a source, not to the YAML of the
slide.

## Adding a technology

Read [`docs/module_spec.md`](docs/module_spec.md) first — it is the contract, not a
suggestion. In short:

1. Add a lineage record to `data/lineage.yaml` (with an `https://` source) and a
   component to `data/components.yaml`.
2. Add one module `src/awsai_demo/<name>_demo.py` with one public
   `def run_<name>_demo(...)` returning a `DemoResult` (the CLI awaits the return
   value only when it is awaitable). Pick the `mode` that is
   **true**, not the one that looks best.
3. Put every external call behind a `Protocol` port and inject it. Offline AWS calls
   use botocore's `Stubber` with `expected_params`; never mock our own code.
4. Add one row to the registry and its operations to `module_spec.md` section 7,
   with lanes, phase and effect.
5. Add `tests/test_<name>_demo.py` and take the module to 100% statements and
   branches.
6. Add the slide to `deck/content/`, its notes section to `deck/speaker_notes.md`,
   and the README row.

## Coding conventions

- Python within 79 columns, docstrings within 72. Google-style docstrings on every
  public API.
- Type public inputs and outputs. Keep `Any` at SDK boundaries and convert it to a
  local `TypedDict`, `Protocol` or dataclass immediately.
- Import optional SDKs **inside** the function, so a missing package is a reported
  result, not a crash.
- Console output stays ASCII: Windows terminals turn an em dash into a replacement
  character.
- Never print a credential, an account id or an ARN. Never commit a `.env`.
- Keep tests offline. Live runs belong in `collect_facts.py --execution live`, with
  explicit flags.

## Honesty rules

These matter most, because the repository's whole argument is that its output can be
trusted.

- **Never widen a claim to make a demo look better.** A scripted response is
  `local_contract` and names its fixture. LocalStack is `local_emulator`, never live.
- **A blocked call is a result, not a failure.** A real 403 is `live_service`,
  `blocked`. Do not catch it and return `ok`.
- **Never invent a zero.** Missing token usage is `None`.
- **Never fall back silently.** Never replay a fixture for a live run. If nothing in
  the live lane runs, return `not_run` and the reason. If a live read such as
  `CountTokens` runs and the billable call is refused, keep that live mode with status
  `blocked`.
- **Do not compare against a straw man.** The Agents Classic and harness lanes of
  `decision` are written as well as the Strands lane.
- If you add a claim about an AWS product, add it to
  [`docs/research.md`](docs/research.md) with a dated source.

## The deck

```bash
uv run --group deck python deck/collect_facts.py
```

```bash
uv run --group deck python deck/build_deck.py
```

Slide numbers come from the order of `deck/content/*.yaml`; never type one. Figures
come only from keys listed in `deck/facts_contract.yaml`. Code panels reference
`# slide: <marker>` regions of tested source. The build fails if a shape overflows the
safe area, a text box is too small for its text, a headline wraps, a slide has no
speaker notes, or the facts were recorded from different code.
