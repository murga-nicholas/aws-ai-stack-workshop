from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

import awsai_demo.network as network
from awsai_demo.network import (
    ENV_GUARD_BOOTSTRAP,
    ENV_GUARD_CONFIG,
    AllowedEndpoint,
    NetworkDenied,
    NetworkGuardConfigurationError,
    NetworkPolicy,
    active_policy,
    assert_stdio_transport,
    bootstrap_python_args,
    bootstrap_python_snippet,
    environment_for_subprocess,
    install_from_environment,
    localstack_endpoint,
    network_guard,
    registered_loopback_endpoint,
)


def test_endpoint_and_policy_validation() -> None:
    endpoint = registered_loopback_endpoint(
        "agentcore-runtime",
        "127.0.0.1",
        8181,
    )
    policy = NetworkPolicy().with_endpoint(endpoint)
    policy.assert_connect_allowed("localhost", 8181)

    with pytest.raises(NetworkGuardConfigurationError):
        AllowedEndpoint("", "127.0.0.1", 1)
    with pytest.raises(NetworkGuardConfigurationError):
        AllowedEndpoint("remote", "example.com", 443)
    with pytest.raises(NetworkGuardConfigurationError):
        AllowedEndpoint("bad-port", "127.0.0.1", 0)
    with pytest.raises(NetworkGuardConfigurationError):
        NetworkPolicy(allowed_endpoints=frozenset({localstack_endpoint()}))

    emulator = NetworkPolicy(
        mode="emulator",
        allowed_endpoints=frozenset({localstack_endpoint()}),
    )
    emulator.assert_connect_allowed("127.0.0.1", 4566)


def test_policy_serialization_and_invalid_json() -> None:
    policy = NetworkPolicy(
        mode="emulator",
        allowed_endpoints=frozenset({localstack_endpoint()}),
        allow_loopback_bind=False,
        allow_stdio_none=False,
        allow_native_subprocesses=True,
    )
    loaded = NetworkPolicy.from_json(policy.to_json())
    assert loaded == policy
    with pytest.raises(NetworkGuardConfigurationError):
        NetworkPolicy.from_json("{")


def test_connect_bind_and_stdio_decisions() -> None:
    policy = NetworkPolicy(
        allowed_endpoints=frozenset(
            {registered_loopback_endpoint("a2a", "127.0.0.1", 9000)},
        ),
    )
    with pytest.raises(NetworkDenied):
        policy.assert_connect_allowed("203.0.113.1", 443)
    with pytest.raises(NetworkDenied):
        policy.assert_connect_allowed("127.0.0.1", 9001)

    policy.assert_bind_allowed("127.0.0.1", 0)
    policy.assert_bind_allowed("", 0)
    with pytest.raises(NetworkDenied):
        policy.assert_bind_allowed("192.0.2.1", 0)
    with pytest.raises(NetworkDenied):
        policy.assert_bind_allowed("127.0.0.1", 70_000)
    with pytest.raises(NetworkDenied):
        NetworkPolicy(allow_loopback_bind=False).assert_bind_allowed(
            "127.0.0.1",
            0,
        )

    assert_stdio_transport("tcp")
    with network_guard(policy):
        assert active_policy() == policy
        assert_stdio_transport("none")
        with pytest.raises(NetworkDenied):
            assert_stdio_transport("tcp")
    assert active_policy() is None


def test_audit_hook_refuses_unregistered_socket_events() -> None:
    policy = NetworkPolicy(
        allowed_endpoints=frozenset(
            {registered_loopback_endpoint("otlp", "127.0.0.1", 4318)},
        ),
    )
    with network_guard(policy):
        sys.audit("socket.connect", None, ("127.0.0.1", 4318))
        sys.audit("socket.bind", None, None)
        sys.audit("socket.bind", None, ("127.0.0.1", 0))
        sys.audit("socket.getaddrinfo", "127.0.0.1", 4318)
        sys.audit("socket.getaddrinfo", "127.0.0.1", None)
        sys.audit("socket.getaddrinfo", None, None)
        left, right = socket.socketpair()
        left.close()
        right.close()
        with pytest.raises(NetworkDenied):
            sys.audit("socket.connect", None, ("203.0.113.1", 443))
        with pytest.raises(NetworkDenied):
            sys.audit("socket.connect", None, "pipe")
        with pytest.raises(NetworkDenied):
            sys.audit("socket.bind", None, "pipe")
        with pytest.raises(NetworkDenied):
            sys.audit("socket.getaddrinfo", "example.com", 443)
        sys.audit("awsai_demo.unrelated")
    sys.audit("socket.connect", None, ("203.0.113.1", 443))


def test_network_helper_edge_branches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv(ENV_GUARD_CONFIG, raising=False)
    assert not network._subprocess_has_guard_env(None)
    monkeypatch.setenv(ENV_GUARD_CONFIG, "present")
    assert network._subprocess_has_guard_env(None)
    assert not network._subprocess_has_guard_env(object())

    assert not network._subprocess_has_bootstrap({}, [])
    assert not network._subprocess_has_bootstrap(
        {ENV_GUARD_BOOTSTRAP: "unexpected"},
        [],
    )
    sitecustomize = tmp_path / "sitecustomize.py"
    assert not network._subprocess_has_bootstrap(
        {ENV_GUARD_BOOTSTRAP: f"sitecustomize:{sitecustomize}"},
        [],
    )
    assert network._command_has_bootstrap(
        ["python", "-c", bootstrap_python_snippet()],
    )
    assert network._looks_like_python_child("runner", "python -c pass")
    assert not network._looks_like_python_child("runner", object())
    assert not network._is_platform_ver_child("powershell.exe", [])
    assert network._command_parts("cmd.exe /c ver") == [
        "cmd.exe",
        "/c",
        "ver",
    ]
    assert network._command_parts(object()) == []

    monkeypatch.setenv("AWSAI_SAMPLE", "value")
    assert network._env_value(object(), "AWSAI_SAMPLE") == "value"
    assert not network._pythonpath_has_parent({}, tmp_path)
    assert not network._is_relative_to(tmp_path / "child", tmp_path / "other")


def test_context_sets_and_restores_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_GUARD_CONFIG, "previous")
    monkeypatch.setenv(ENV_GUARD_BOOTSTRAP, "previous-bootstrap")
    policy = NetworkPolicy()
    with network_guard(policy):
        assert active_policy() == policy
        assert ENV_GUARD_CONFIG in environment_for_subprocess(policy)
    assert active_policy() is None
    assert ENV_GUARD_CONFIG in os.environ
    monkeypatch.delenv(ENV_GUARD_CONFIG)
    monkeypatch.delenv(ENV_GUARD_BOOTSTRAP)
    assert install_from_environment() is False
    assert bootstrap_python_snippet().startswith("import awsai_demo.network")
    with pytest.raises(NetworkGuardConfigurationError):
        bootstrap_python_args(["-m", "module"])


def test_subprocess_audit_requires_bootstrap(tmp_path: Path) -> None:
    policy = NetworkPolicy()
    env = environment_for_subprocess(
        policy,
        src_path=Path("src").resolve(),
    )
    with network_guard(policy), pytest.raises(NetworkDenied):
        subprocess.run(
            [sys.executable, "-c", "pass"],
            env=env,
            check=False,
        )

    code = (
        "import sys\n"
        "from awsai_demo.network import NetworkDenied\n"
        "try:\n"
        "    sys.audit('socket.connect', None, ('203.0.113.1', 443))\n"
        "except NetworkDenied:\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(5)\n"
    )
    args = [sys.executable, *bootstrap_python_args(["-c", code])]
    with network_guard(policy):
        result = subprocess.run(  # noqa: S603
            args,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
    assert result.returncode == 0

    site_env = environment_for_subprocess(
        policy,
        src_path=Path("src").resolve(),
        bootstrap_dir=tmp_path / "boot",
    )
    with network_guard(policy):
        site_result = subprocess.run(  # noqa: S603
            [sys.executable, "-c", code],
            env=site_env,
            check=False,
            capture_output=True,
            text=True,
        )
    assert site_result.returncode == 0


def test_subprocess_audit_refuses_native_by_default() -> None:
    policy = NetworkPolicy()
    with network_guard(policy), pytest.raises(NetworkDenied):
        sys.audit("subprocess.Popen", "cmd.exe", ["cmd.exe"], None, None)
    with network_guard(policy):
        sys.audit(
            "subprocess.Popen",
            "cmd.exe",
            ["cmd.exe", "/c", "ver"],
            None,
            None,
        )
    with network_guard(policy), pytest.raises(NetworkDenied):
        sys.audit(
            "subprocess.Popen",
            "cmd.exe",
            ["cmd.exe", "/c", "dir"],
            None,
            None,
        )

    permissive = NetworkPolicy(allow_native_subprocesses=True)
    with network_guard(permissive):
        sys.audit("subprocess.Popen", "cmd.exe", ["cmd.exe"], None, None)
