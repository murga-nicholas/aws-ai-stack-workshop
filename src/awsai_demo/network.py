"""Process-wide offline and emulator network guard."""

from __future__ import annotations

import json
import os
import platform
import sys
import sysconfig
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from inspect import currentframe
from ipaddress import ip_address
from pathlib import Path
from typing import Any, Final, Literal, cast

GuardMode = Literal["offline", "emulator"]
ENV_GUARD_CONFIG: Final = "AWSAI_NETWORK_GUARD"
ENV_GUARD_BOOTSTRAP: Final = "AWSAI_NETWORK_BOOTSTRAP"
_BOOTSTRAP_TARGET: Final = "awsai_demo.network:install_from_environment"


class NetworkGuardError(Exception):
    """Base class for network guard failures."""


class NetworkDeniedError(NetworkGuardError):
    """Raised when guarded code attempts a forbidden transport."""


class NetworkGuardConfigurationError(NetworkGuardError):
    """Raised when a guard policy cannot be loaded."""


NetworkDenied = NetworkDeniedError


@dataclass(frozen=True, slots=True)
class AllowedEndpoint:
    """A registered loopback endpoint allowed by the guard."""

    name: str
    host: str
    port: int

    def __post_init__(self) -> None:
        """Validate endpoint registration."""
        if not self.name:
            message = "endpoint name must not be empty"
            raise NetworkGuardConfigurationError(message)
        if not _is_loopback_host(self.host):
            message = "registered endpoints must be loopback"
            raise NetworkGuardConfigurationError(message)
        if self.port <= 0 or self.port > 65_535:
            message = "endpoint port must be in 1..65535"
            raise NetworkGuardConfigurationError(message)

    def to_json(self) -> dict[str, Any]:
        """Serialize the endpoint for child-process propagation."""
        return {"name": self.name, "host": self.host, "port": self.port}

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> AllowedEndpoint:
        """Deserialize one endpoint from environment JSON."""
        return cls(
            name=str(raw["name"]),
            host=str(raw["host"]),
            port=int(raw["port"]),
        )


@dataclass(frozen=True, slots=True)
class NetworkPolicy:
    """Offline or emulator network policy for one guarded scope."""

    mode: GuardMode = "offline"
    allowed_endpoints: frozenset[AllowedEndpoint] = frozenset()
    allow_loopback_bind: bool = True
    allow_stdio_none: bool = True
    allow_native_subprocesses: bool = False

    def __post_init__(self) -> None:
        """Validate and freeze endpoint inputs."""
        object.__setattr__(
            self,
            "allowed_endpoints",
            frozenset(self.allowed_endpoints),
        )
        for endpoint in self.allowed_endpoints:
            if endpoint.name == "localstack" and self.mode != "emulator":
                message = "LocalStack endpoint is emulator-only"
                raise NetworkGuardConfigurationError(message)

    def with_endpoint(self, endpoint: AllowedEndpoint) -> NetworkPolicy:
        """Return a copy with one more loopback endpoint."""
        return NetworkPolicy(
            mode=self.mode,
            allowed_endpoints=self.allowed_endpoints | frozenset({endpoint}),
            allow_loopback_bind=self.allow_loopback_bind,
            allow_stdio_none=self.allow_stdio_none,
            allow_native_subprocesses=self.allow_native_subprocesses,
        )

    def assert_connect_allowed(self, host: str, port: int) -> None:
        """Raise when a connect target is not registered loopback."""
        if not _is_loopback_host(host):
            message = "offline guard refused non-loopback connection"
            raise NetworkDenied(message)
        if not any(
            endpoint.port == port and _same_loopback_host(endpoint.host, host)
            for endpoint in self.allowed_endpoints
        ):
            message = "offline guard refused unregistered loopback endpoint"
            raise NetworkDenied(message)

    def assert_bind_allowed(self, host: str, port: int) -> None:
        """Raise when a bind target is not permitted."""
        if not self.allow_loopback_bind:
            message = "offline guard refused socket bind"
            raise NetworkDenied(message)
        if host and not _is_loopback_host(host):
            message = "offline guard refused non-loopback bind"
            raise NetworkDenied(message)
        if port < 0 or port > 65_535:
            message = "offline guard refused invalid bind port"
            raise NetworkDenied(message)

    def assert_stdio_transport(self, transport: str) -> None:
        """Allow MCP's recorded no-socket transport only."""
        if self.allow_stdio_none and transport == "none":
            return
        message = "offline guard refused socket-based MCP transport"
        raise NetworkDenied(message)

    def to_json(self) -> str:
        """Serialize the policy for subprocess inheritance."""
        data = {
            "mode": self.mode,
            "allowed_endpoints": [
                endpoint.to_json() for endpoint in self.allowed_endpoints
            ],
            "allow_loopback_bind": self.allow_loopback_bind,
            "allow_stdio_none": self.allow_stdio_none,
            "allow_native_subprocesses": self.allow_native_subprocesses,
        }
        return json.dumps(data, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> NetworkPolicy:
        """Load a policy from environment JSON."""
        try:
            data = cast("dict[str, Any]", json.loads(raw))
            raw_endpoints = cast(
                "list[Mapping[str, Any]]",
                data["allowed_endpoints"],
            )
            endpoints = frozenset(
                AllowedEndpoint.from_json(item) for item in raw_endpoints
            )
            return cls(
                mode=cast("GuardMode", data["mode"]),
                allowed_endpoints=endpoints,
                allow_loopback_bind=bool(data["allow_loopback_bind"]),
                allow_stdio_none=bool(data["allow_stdio_none"]),
                allow_native_subprocesses=bool(
                    data.get("allow_native_subprocesses", False),
                ),
            )
        except (
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            message = "invalid network guard configuration"
            raise NetworkGuardConfigurationError(message) from error


@dataclass(slots=True)
class _GuardState:
    hook_installed: bool = False
    active_policy: NetworkPolicy | None = None


_STATE = _GuardState()


def registered_loopback_endpoint(
    name: str,
    host: str,
    port: int,
) -> AllowedEndpoint:
    """Create a validated loopback endpoint registration."""
    return AllowedEndpoint(name=name, host=host, port=port)


def localstack_endpoint() -> AllowedEndpoint:
    """Return the standard LocalStack endpoint registration."""
    return AllowedEndpoint(name="localstack", host="localhost", port=4566)


def prime_platform_uname() -> None:
    """Prime platform OS detection before subprocess auditing starts."""
    platform.uname()


def install_network_guard(policy: NetworkPolicy | None = None) -> None:
    """Install the audit hook and maybe activate a policy."""
    if not _STATE.hook_installed:
        prime_platform_uname()
        cast("Any", _audit_hook).__cantrace__ = True
        sys.addaudithook(_audit_hook)
        _STATE.hook_installed = True
    if policy is not None:
        _STATE.active_policy = policy


@contextmanager
def network_guard(policy: NetworkPolicy) -> Iterator[None]:
    """Activate a guard policy for the current process scope."""
    install_network_guard()
    previous_policy = _STATE.active_policy
    previous_config = os.environ.get(ENV_GUARD_CONFIG)
    previous_bootstrap = os.environ.get(ENV_GUARD_BOOTSTRAP)
    _STATE.active_policy = policy
    os.environ[ENV_GUARD_CONFIG] = policy.to_json()
    os.environ[ENV_GUARD_BOOTSTRAP] = _BOOTSTRAP_TARGET
    try:
        yield
    finally:
        _STATE.active_policy = previous_policy
        _restore_env(ENV_GUARD_CONFIG, previous_config)
        _restore_env(ENV_GUARD_BOOTSTRAP, previous_bootstrap)


def install_from_environment() -> bool:
    """Install an inherited guard policy."""
    raw = os.environ.get(ENV_GUARD_CONFIG)
    if raw is None:
        return False
    install_network_guard(NetworkPolicy.from_json(raw))
    return True


def environment_for_subprocess(
    policy: NetworkPolicy,
    env: Mapping[str, str] | None = None,
    *,
    src_path: Path | None = None,
    bootstrap_dir: Path | None = None,
) -> dict[str, str]:
    """Return env that carries guard config to children."""
    merged = dict(os.environ if env is None else env)
    merged[ENV_GUARD_CONFIG] = policy.to_json()
    source = (
        Path(__file__).resolve().parents[1] if src_path is None else src_path
    )
    path_parts = [str(source)]
    if bootstrap_dir is not None:
        bootstrap_path = write_sitecustomize_bootstrap(bootstrap_dir)
        path_parts.insert(0, str(bootstrap_path.parent))
        merged[ENV_GUARD_BOOTSTRAP] = f"sitecustomize:{bootstrap_path}"
    else:
        merged[ENV_GUARD_BOOTSTRAP] = _BOOTSTRAP_TARGET
    existing = merged.get("PYTHONPATH")
    prefix = os.pathsep.join(path_parts)
    merged["PYTHONPATH"] = (
        prefix if not existing else f"{prefix}{os.pathsep}{existing}"
    )
    return merged


def bootstrap_python_snippet() -> str:
    """Return code that activates guard in a child."""
    return (
        "import awsai_demo.network as _awsai_network; "
        "_awsai_network.install_from_environment()"
    )


def bootstrap_python_args(args: Sequence[str]) -> list[str]:
    """Wrap a Python ``-c`` command with guard bootstrap code."""
    if len(args) < 2 or args[0] != "-c":
        message = "only python -c commands can be bootstrap-wrapped"
        raise NetworkGuardConfigurationError(message)
    return ["-c", f"{bootstrap_python_snippet()}\n{args[1]}", *args[2:]]


def write_sitecustomize_bootstrap(bootstrap_dir: Path) -> Path:
    """Write a sitecustomize bootstrap file for Python children."""
    bootstrap_dir.mkdir(parents=True, exist_ok=True)
    target = bootstrap_dir / "sitecustomize.py"
    target.write_text(f"{bootstrap_python_snippet()}\n", encoding="utf-8")
    return target


def assert_stdio_transport(transport: str) -> None:
    """Validate an MCP transport against the active guard."""
    if _STATE.active_policy is None:
        return
    _STATE.active_policy.assert_stdio_transport(transport)


def active_policy() -> NetworkPolicy | None:
    """Return the active process policy, if any."""
    return _STATE.active_policy


def _audit_hook(event: str, args: tuple[Any, ...]) -> None:
    policy = _STATE.active_policy
    if policy is None:
        return
    if event == "socket.connect":
        _audit_connect(policy, args)
    elif event == "socket.bind":
        _audit_bind(policy, args)
    elif event == "socket.getaddrinfo":
        _audit_getaddrinfo(policy, args)
    elif event == "subprocess.Popen":
        _audit_subprocess(policy, args)


def _audit_connect(policy: NetworkPolicy, args: tuple[Any, ...]) -> None:
    address = args[1] if len(args) > 1 else None
    if not isinstance(address, tuple) or len(address) < 2:
        message = "offline guard refused non-TCP socket connect"
        raise NetworkDenied(message)
    host = str(address[0])
    if _is_loopback_host(host) and _is_stdlib_socketpair_call():
        return
    policy.assert_connect_allowed(host, int(address[1]))


def _audit_bind(policy: NetworkPolicy, args: tuple[Any, ...]) -> None:
    address = args[1] if len(args) > 1 else None
    if address is None:
        return
    if not isinstance(address, tuple) or len(address) < 2:
        message = "offline guard refused non-TCP socket bind"
        raise NetworkDenied(message)
    policy.assert_bind_allowed(str(address[0]), int(address[1]))


def _audit_getaddrinfo(policy: NetworkPolicy, args: tuple[Any, ...]) -> None:
    host = args[0] if args else None
    if host is None:
        return
    if not _is_loopback_host(str(host)):
        message = "offline guard refused non-loopback name lookup"
        raise NetworkDenied(message)
    port = int(args[1]) if len(args) > 1 and args[1] is not None else 0
    if port > 0:
        policy.assert_connect_allowed(str(host), port)


def _audit_subprocess(policy: NetworkPolicy, args: tuple[Any, ...]) -> None:
    executable = str(args[0]) if args else ""
    command = args[1] if len(args) > 1 else ()
    env = args[3] if len(args) > 3 else None
    if not _looks_like_python_child(executable, command):
        if _is_platform_ver_child(executable, command):
            return
        if policy.allow_native_subprocesses:
            return
        message = "offline guard refused native subprocess"
        raise NetworkDenied(message)
    if _subprocess_has_guard_env(env) and _subprocess_has_bootstrap(
        env, command
    ):
        return
    message = "Python subprocess missing inherited guard bootstrap"
    raise NetworkDenied(message)


def _subprocess_has_guard_env(env: Any) -> bool:
    if env is None:
        return ENV_GUARD_CONFIG in os.environ
    if isinstance(env, Mapping):
        return ENV_GUARD_CONFIG in env
    return False


def _subprocess_has_bootstrap(env: Any, command: Any) -> bool:
    bootstrap = _env_value(env, ENV_GUARD_BOOTSTRAP)
    if bootstrap is None:
        return False
    if bootstrap.startswith("sitecustomize:"):
        path = bootstrap.partition(":")[2]
        return bool(path) and _pythonpath_has_parent(env, Path(path).parent)
    if bootstrap == _BOOTSTRAP_TARGET:
        return _command_has_bootstrap(command)
    return False


def _env_value(env: Any, name: str) -> str | None:
    if isinstance(env, Mapping):
        value = env.get(name)
        return None if value is None else str(value)
    return os.environ.get(name)


def _pythonpath_has_parent(env: Any, parent: Path) -> bool:
    value = _env_value(env, "PYTHONPATH")
    if value is None:
        return False
    parents = {Path(item) for item in value.split(os.pathsep) if item}
    return parent in parents


def _command_has_bootstrap(command: Any) -> bool:
    if isinstance(command, Sequence) and not isinstance(command, str):
        joined = "\n".join(str(item) for item in command)
    else:
        joined = str(command)
    return (
        "awsai_demo.network" in joined and "install_from_environment" in joined
    )


def _looks_like_python_child(executable: str, command: Any) -> bool:
    candidates = [executable]
    if isinstance(command, Sequence) and not isinstance(command, str):
        candidates.extend(str(item) for item in command[:1])
    elif isinstance(command, str):
        candidates.append(command.split(" ", maxsplit=1)[0])
    return any(
        "python" in Path(candidate).name.lower() for candidate in candidates
    )


def _is_platform_ver_child(executable: str, command: Any) -> bool:
    executable_name = Path(executable).name.lower()
    if executable_name not in {"cmd", "cmd.exe"}:
        return False
    lowered = [part.lower() for part in _command_parts(command)]
    return lowered in (["cmd.exe", "/c", "ver"], ["cmd", "/c", "ver"])


def _command_parts(command: Any) -> list[str]:
    if isinstance(command, Sequence) and not isinstance(command, str):
        return [str(item) for item in command]
    if isinstance(command, str):
        return command.split()
    return []


def _is_stdlib_socketpair_call() -> bool:
    stdlib = Path(sysconfig.get_paths()["stdlib"]).resolve()
    frame = currentframe()
    try:
        while frame is not None:
            code = frame.f_code
            filename = Path(code.co_filename)
            if (
                code.co_name in {"socketpair", "_fallback_socketpair"}
                and filename.name == "socket.py"
                and _is_relative_to(filename.resolve(), stdlib)
            ):
                return True
            frame = frame.f_back
        return False
    finally:
        del frame


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def _is_loopback_host(host: str) -> bool:
    normalized = host.strip("[]").lower()
    if normalized == "localhost":
        return True
    try:
        return ip_address(normalized).is_loopback
    except ValueError:
        return False


def _same_loopback_host(left: str, right: str) -> bool:
    return _is_loopback_host(left) and _is_loopback_host(right)
