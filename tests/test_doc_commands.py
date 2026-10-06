from __future__ import annotations

import io
import re
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    import pytest

from awsai_demo.cli import build_parser, dispatch
from awsai_demo.lineage import data_text, parse_components, parse_lineage
from awsai_demo.registry import DEMOS
from awsai_demo.runtime import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMMAND_STARTS = ("awsai-demo ", "uv run awsai-demo ")
UTILITY_COMMANDS = {"cleanup", "doctor", "lineage", "list"}


@dataclass(frozen=True)
class CommandExample:
    source: str
    command: str


def test_documented_commands_parse_with_real_cli_parser() -> None:
    parser = build_parser()
    errors: list[str] = []

    for example in _documented_commands():
        try:
            parser.parse_args(_cli_argv(example.command))
        except (SystemExit, ValueError) as exc:
            errors.append(f"{example.source}: {example.command!r}: {exc}")

    assert errors == []


def test_offline_utility_examples_execute_without_aws(tmp_path: Path) -> None:
    parser = build_parser()
    seen: set[tuple[str, ...]] = set()

    for example in _documented_commands():
        argv = tuple(_cli_argv(example.command))
        if argv in seen or not _is_offline_utility(argv):
            continue
        seen.add(argv)
        result = dispatch(
            parser.parse_args(list(argv)),
            Settings(),
            run_directory=tmp_path,
        )
        assert result["status"] in {"ok", "blocked"}
        assert result["evidence"]["aws_executed"] is False

    assert seen


def test_component_slide_references_match_deck_content() -> None:
    data_dir = PROJECT_ROOT / "data"
    lineage = parse_lineage(data_text("lineage.yaml", data_dir))
    slide_ids = _slide_ids()
    components = parse_components(
        data_text("components.yaml", data_dir),
        lineage_ids={record.id for record in lineage},
        demo_ids={demo.name for demo in DEMOS},
        slide_ids=slide_ids,
    )

    assert components


def _documented_commands() -> tuple[CommandExample, ...]:
    examples = [
        *_markdown_commands(PROJECT_ROOT / "README.md"),
        *(
            example
            for path in sorted((PROJECT_ROOT / "docs").rglob("*.md"))
            for example in _markdown_commands(path)
        ),
        *(
            example
            for path in sorted(
                (PROJECT_ROOT / "deck" / "content").glob("*.yaml")
            )
            for example in _yaml_commands(path)
        ),
    ]
    unique: dict[tuple[str, str], CommandExample] = {}
    for example in examples:
        unique[(example.source, example.command)] = example
    return tuple(unique.values())


def _markdown_commands(path: Path) -> tuple[CommandExample, ...]:
    relative = path.relative_to(PROJECT_ROOT).as_posix()
    examples: list[CommandExample] = []
    in_fence = False
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            command = _line_command(stripped)
            if command is not None:
                examples.append(
                    CommandExample(f"{relative}:{line_number}", command),
                )
        for inline in _inline_code(stripped):
            command = _line_command(inline)
            if command is not None:
                examples.append(
                    CommandExample(f"{relative}:{line_number}", command),
                )
    return tuple(examples)


def _inline_code(line: str) -> tuple[str, ...]:
    parts = line.split("`")
    return tuple(parts[index].strip() for index in range(1, len(parts), 2))


def _yaml_commands(path: Path) -> tuple[CommandExample, ...]:
    relative = path.relative_to(PROJECT_ROOT).as_posix()
    document = yaml.safe_load(path.read_text())
    examples = [
        CommandExample(f"{relative}:{location}", command)
        for location, command in _walk_yaml_commands(document)
    ]
    return tuple(examples)


def _walk_yaml_commands(
    value: Any, path: str = "$"
) -> tuple[tuple[str, str], ...]:
    found: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child_path = f"{path}.{key}"
            if key == "run":
                found.extend(_commands_from_run_field(item, child_path))
            else:
                found.extend(_walk_yaml_commands(item, child_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_walk_yaml_commands(item, f"{path}[{index}]"))
    elif isinstance(value, str):
        command = _line_command(value.strip())
        if command is not None:
            found.append((path, command))
    return tuple(found)


def _commands_from_run_field(
    value: Any, path: str
) -> tuple[tuple[str, str], ...]:
    if isinstance(value, str):
        command = _line_command(value.strip())
        return () if command is None else ((path, command),)
    if isinstance(value, list):
        return tuple(
            (f"{path}[{index}]", command)
            for index, item in enumerate(value)
            if isinstance(item, str)
            for command in [_line_command(item.strip())]
            if command is not None
        )
    return ()


def _line_command(text: str) -> str | None:
    candidate = text.removeprefix("$ ").removeprefix("> ").strip()
    if "[" in candidate or "]" in candidate or "<" in candidate:
        return None
    if "..." in candidate or "?" in candidate:
        return None
    if candidate.startswith(COMMAND_STARTS):
        return candidate
    return None


def _cli_argv(command: str) -> list[str]:
    tokens = shlex.split(command, comments=True, posix=True)
    if tokens[:2] == ["uv", "run"]:
        tokens = tokens[2:]
    assert tokens and tokens[0] == "awsai-demo"
    return tokens[1:]


def _is_offline_utility(argv: tuple[str, ...]) -> bool:
    if not argv or argv[0] not in UTILITY_COMMANDS:
        return False
    if "--probe" in argv or "--execute" in argv:
        return False
    if "--execution" in argv:
        index = argv.index("--execution")
        return index + 1 < len(argv) and argv[index + 1] == "offline"
    return True


_EXPECT_FENCE = re.compile(
    r"<!-- example: (?P<lane>\w+) expect=(?P<status>[\w]+) -->[ \t]*\n"
    r"```[^\n]*\n(?P<body>.*?)```",
    re.DOTALL,
)


def _expect_fences(path: Path) -> tuple[tuple[str, str, str, str], ...]:
    text = path.read_text(encoding="utf-8")
    return tuple(
        (
            str(path),
            match.group("lane"),
            match.group("status"),
            match.group("body"),
        )
        for match in _EXPECT_FENCE.finditer(text)
    )


def _documented_fences() -> tuple[tuple[str, str, str, str], ...]:
    paths = [PROJECT_ROOT / "README.md"]
    paths.extend(sorted((PROJECT_ROOT / "docs").rglob("*.md")))
    paths.extend(sorted((PROJECT_ROOT / "deck" / "content").glob("*.yaml")))
    return tuple(
        item
        for path in paths
        for item in _expect_fences(path)
        if path.is_file()
    )


def _first_command(body: str) -> str | None:
    for line in body.splitlines():
        command = _line_command(line.strip())
        if command is not None:
            return command
    return None


def test_expect_fences_execute_in_a_temporary_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from awsai_demo.cli import main

    work = tmp_path / "work"
    shutil.copytree(PROJECT_ROOT / "data", work / "data")
    monkeypatch.chdir(work)
    synthetic = tmp_path / "synthetic.md"
    synthetic.write_text(
        "<!-- example: offline expect=exit1 -->\n"
        "```bash\n"
        "uv run awsai-demo not-a-command\n"
        "```\n"
        "<!-- example: live expect=ok -->\n"
        "```bash\n"
        "uv run awsai-demo doctor --execution live\n"
        "```\n"
        "<!-- example: emulator expect=ok -->\n"
        "```bash\n"
        "uv run awsai-demo localstack --execution emulator\n"
        "```\n"
        "<!-- example: offline expect=ok -->\n"
        "```bash\n"
        "uv run awsai-demo list --run-id <not-a-real-id>\n"
        "```\n",
        encoding="utf-8",
    )
    ran: list[str] = []
    for source, lane, expect, body in (
        *_documented_fences(),
        *_expect_fences(synthetic),
    ):
        command = _first_command(body)
        if lane != "offline" or command is None:
            continue
        stdout, stderr = io.StringIO(), io.StringIO()
        code = main(_cli_argv(command), stdout=stdout, stderr=stderr)
        if expect == "exit1":
            assert code == 1, f"{source}: {stderr.getvalue()}"
        else:
            assert code == 0, f"{source}: {stderr.getvalue()}"
            first = stdout.getvalue().splitlines()[0]
            status = first.split(":", 1)[1].strip().split(" ", 1)[0]
            assert status == expect, first
        ran.append(expect)
    assert {"ok", "paused", "exit1"} <= set(ran)


def _slide_ids() -> set[str]:
    ids: set[str] = set()
    for path in sorted((PROJECT_ROOT / "deck" / "content").glob("*.yaml")):
        document = yaml.safe_load(path.read_text())
        if not isinstance(document, dict):
            continue
        slides = document.get("slides", [])
        if not isinstance(slides, list):
            continue
        ids.update(
            slide["id"]
            for slide in slides
            if isinstance(slide, dict) and isinstance(slide.get("id"), str)
        )
    return ids
