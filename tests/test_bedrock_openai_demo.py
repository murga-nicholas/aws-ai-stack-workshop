from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import httpx
import pytest
from botocore.credentials import Credentials

import awsai_demo.bedrock_openai_demo as demo
from awsai_demo.bedrock_openai_demo import (
    OpenAISdkPort,
    OpenAITextResult,
    StaticTokenCounter,
    _chat_usage,
    _credentials_from_session,
    _list_openai_catalog,
    _reserve_and_call_model,
    _responses_usage,
    _token_skip_code,
    bedrock_openai_base_url,
    chat_completion_request,
    default_openai_client,
    generate_bedrock_api_key,
    responses_request,
    run_bedrock_openai_demo,
)
from awsai_demo.credentials import SelectedSession
from awsai_demo.policy import (
    ExecutionPolicy,
    Reservation,
    ReservationUnavailable,
    TokenCount,
)
from awsai_demo.runtime import Settings

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


_DEFAULT_MODEL = "openai.gpt-oss-20b-1:0"


def _data(result: Mapping[str, Any]) -> Mapping[str, Any]:
    return cast("Mapping[str, Any]", result["data"])


class FakeBudget:
    def __init__(self, *, fail_on: str | None = None) -> None:
        self.fail_on = fail_on
        self.calls: list[tuple[str, str, bytes]] = []

    def reserve_model(
        self,
        reservation_id: str,
        *,
        model_id: str,
        serialized_request: bytes,
        exact_counter: object | None = None,
        tier: str = "standard",
        routing: str = "in-region",
    ) -> tuple[Reservation, TokenCount]:
        del exact_counter, tier, routing
        self.calls.append((reservation_id, model_id, serialized_request))
        if self.fail_on in {"*", reservation_id}:
            message = "missing model price"
            raise ReservationUnavailable(message)
        amount = Decimal("0.000001")
        total = amount * Decimal(len(self.calls))
        return (
            Reservation(reservation_id, amount, "model", total, Decimal("0")),
            TokenCount(model_id, 12, "exact", "test"),
        )


class FakeClient:
    def __init__(self, *, fail_chat: BaseException | None = None) -> None:
        self.fail_chat = fail_chat
        self.calls: list[tuple[str, str]] = []

    def chat_completion(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, str]],
        max_tokens: int,
    ) -> OpenAITextResult:
        del messages, max_tokens
        self.calls.append(("chat", model))
        if self.fail_chat is not None:
            raise self.fail_chat
        return OpenAITextResult(
            "chat ok",
            {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
        )

    def response(
        self,
        *,
        model: str,
        input_text: str,
        max_output_tokens: int,
    ) -> OpenAITextResult:
        del input_text, max_output_tokens
        self.calls.append(("responses", model))
        return OpenAITextResult(
            "response ok",
            {"input_tokens": 4, "output_tokens": 5, "total_tokens": 9},
        )


class FakeFactory:
    def __init__(self, *, fail_chat: BaseException | None = None) -> None:
        self.fail_chat = fail_chat
        self.clients: list[FakeClient] = []
        self.calls: list[tuple[str, str]] = []

    def __call__(
        self,
        *,
        api_key: str,
        base_url: str,
        policy: ExecutionPolicy,
    ) -> FakeClient:
        del policy
        self.calls.append((api_key, base_url))
        client = FakeClient(fail_chat=self.fail_chat)
        self.clients.append(client)
        return client


class FakeCatalogPort:
    endpoint_url = "https://bedrock.us-east-2.amazonaws.com"

    def __init__(self, summaries: Sequence[Mapping[str, object]]) -> None:
        self.summaries = summaries
        self.calls: list[tuple[str, str, Mapping[str, Any]]] = []

    def call(
        self,
        service: str,
        operation_name: str,
        params: Mapping[str, Any],
    ) -> Mapping[str, object]:
        self.calls.append((service, operation_name, params))
        return {"modelSummaries": list(self.summaries)}


class FakeSession:
    def __init__(self) -> None:
        self.credentials = Credentials("AKIDEXAMPLE", "secret")

    def get_credentials(self) -> Credentials:
        return self.credentials


class FakeBotoSession:
    def __init__(self) -> None:
        self.client_instance = FakeBotoClient()
        self.clients: list[tuple[str, Mapping[str, object]]] = []

    def client(self, service_name: str, **kwargs: object) -> FakeBotoClient:
        self.clients.append((service_name, kwargs))
        return self.client_instance


class FakeBotoClient:
    def __init__(self) -> None:
        self.meta = FakeBotoMeta(
            "https://bedrock.us-east-1.amazonaws.com",
        )
        self.calls: list[Mapping[str, object]] = []

    def list_foundation_models(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        return dict(
            _catalog(_DEFAULT_MODEL).call(
                "bedrock",
                "ListFoundationModels",
                kwargs,
            )
        )


class FakeBotoMeta:
    def __init__(self, endpoint_url: str) -> None:
        self.endpoint_url = endpoint_url


class InternalOnlySession:
    def __init__(self) -> None:
        self._session = FakeSession()


class ResponseStatusError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.response = FakeHttpResponse(status_code)


class FakeHttpResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class UnsupportedModelError(Exception):
    pass


def test_bedrock_openai_offline_uses_mock_transport() -> None:
    result = run_bedrock_openai_demo(
        execution="offline",
        settings=Settings(),
        policy=ExecutionPolicy(),
    )

    assert result["mode"] == "local_contract"
    assert result["status"] == "ok"
    assert [item["operation"] for item in result["operations"]] == [
        "provide_token",
        "chat.completions.create",
        "responses.create",
    ]
    assert _data(result)["chat_text"].startswith("Bedrock accepted")
    assert _data(result)["responses_model"] == "us.openai.gpt-5.6-sol/global"
    assert result["operations"][1]["usage"] == {
        "input_tokens": 12,
        "output_tokens": 6,
        "total_tokens": 18,
    }


def test_bedrock_openai_emulator_is_not_supported() -> None:
    result = run_bedrock_openai_demo(
        execution="emulator",
        settings=Settings(),
        policy=ExecutionPolicy(),
    )

    assert result["mode"] == "not_run"
    assert {item["error_code"] for item in result["operations"]} == {
        "not_supported_by_emulator",
    }


def test_bedrock_openai_live_reserves_chat_after_catalog_and_uses_key() -> (
    None
):
    budget = FakeBudget()
    factory = FakeFactory()
    catalog = _catalog(_DEFAULT_MODEL)
    token_calls: list[tuple[object, str]] = []

    def token_provider(selected: object, region: str) -> str:
        token_calls.append((selected, region))
        return "bedrock-live-token"

    result = run_bedrock_openai_demo(
        execution="live",
        settings=Settings(region="us-east-2"),
        policy=ExecutionPolicy(),
        budget=cast("Any", budget),
        catalog_port=catalog,
        client_factory=factory,
        token_provider=token_provider,
        exact_counter=StaticTokenCounter(12),
        session_factory=lambda **_kwargs: FakeSession(),
    )

    assert result["mode"] == "live_model"
    assert result["status"] == "ok"
    assert result["evidence"]["credential_source"] == "default_chain"
    assert result["evidence"]["observed_model"] == _DEFAULT_MODEL
    assert result["evidence"]["estimated_cost_usd"] == 0.000001
    assert [item["operation"] for item in result["operations"]] == [
        "ListFoundationModels",
        "provide_token",
        "chat.completions.create",
        "responses.create",
    ]
    assert result["operations"][3]["mode"] == "not_run"
    assert result["operations"][3]["error_code"] == "not_supported_by_model"
    assert catalog.calls == [
        ("bedrock", "ListFoundationModels", {"byProvider": "OpenAI"}),
    ]
    assert [call[0] for call in budget.calls] == ["chat.completions.create"]
    assert len(token_calls) == 1
    assert factory.calls == [
        (
            "bedrock-live-token",
            "https://bedrock-runtime.us-east-2.amazonaws.com/openai/v1",
        ),
    ]
    assert _data(result)["chat_text"] == "chat ok"
    assert "responses_text" not in _data(result)


def test_bedrock_openai_live_prefers_tested_default_chat_model() -> None:
    budget = FakeBudget()
    result = run_bedrock_openai_demo(
        execution="live",
        settings=Settings(),
        policy=ExecutionPolicy(),
        budget=cast("Any", budget),
        catalog_port=FakeCatalogPort(
            [
                {
                    "providerName": "OpenAI",
                    "modelId": "openai.gpt-oss-120b-1:0",
                },
                {
                    "providerName": "OpenAI",
                    "modelId": _DEFAULT_MODEL,
                },
            ],
        ),
        client_factory=FakeFactory(),
        token_provider=lambda _selected, _region: "bedrock-live-token",
        exact_counter=StaticTokenCounter(12),
        session_factory=lambda **_kwargs: FakeSession(),
    )

    assert result["evidence"]["observed_model"] == _DEFAULT_MODEL
    assert budget.calls[0][1] == _DEFAULT_MODEL


def test_bedrock_openai_live_uses_sorted_supported_fallback() -> None:
    budget = FakeBudget()
    result = run_bedrock_openai_demo(
        execution="live",
        settings=Settings(),
        policy=ExecutionPolicy(),
        budget=cast("Any", budget),
        catalog_port=_catalog("openai.gpt-oss-120b-1:0"),
        client_factory=FakeFactory(),
        token_provider=lambda _selected, _region: "bedrock-live-token",
        exact_counter=StaticTokenCounter(12),
        session_factory=lambda **_kwargs: FakeSession(),
    )

    assert result["evidence"]["observed_model"] == "openai.gpt-oss-120b-1:0"
    assert budget.calls[0][1] == "openai.gpt-oss-120b-1:0"


def test_bedrock_openai_live_default_port_reuses_selected_session() -> None:
    budget = FakeBudget()
    session = FakeBotoSession()
    selected_sessions: list[SelectedSession] = []

    def token_provider(selected: SelectedSession, region: str) -> str:
        assert region == "us-east-1"
        selected_sessions.append(selected)
        return "bedrock-live-token"

    result = run_bedrock_openai_demo(
        execution="live",
        settings=Settings(),
        policy=ExecutionPolicy(),
        budget=cast("Any", budget),
        client_factory=FakeFactory(),
        token_provider=token_provider,
        exact_counter=StaticTokenCounter(12),
        session_factory=lambda **_kwargs: session,
    )

    assert result["mode"] == "live_model"
    assert result["evidence"]["credential_source"] == "default_chain"
    assert session.clients[0][0] == "bedrock"
    assert session.client_instance.calls == [{"byProvider": "OpenAI"}]
    assert selected_sessions[0].session is session


def test_bedrock_openai_live_caps_request_before_reservation() -> None:
    budget = FakeBudget()
    result = run_bedrock_openai_demo(
        execution="live",
        settings=Settings(),
        policy=ExecutionPolicy(max_output_tokens=8),
        budget=cast("Any", budget),
        catalog_port=_catalog(_DEFAULT_MODEL),
        client_factory=FakeFactory(),
        token_provider=lambda _selected, _region: "bedrock-live-token",
        exact_counter=StaticTokenCounter(12),
        session_factory=lambda **_kwargs: FakeSession(),
    )

    reserved_request = json.loads(budget.calls[0][2])
    assert result["mode"] == "live_model"
    assert reserved_request["max_tokens"] == 8


def test_bedrock_openai_live_budget_failure_preserves_local_signing() -> None:
    budget = FakeBudget(fail_on="*")
    factory = FakeFactory()

    result = run_bedrock_openai_demo(
        execution="live",
        settings=Settings(),
        policy=ExecutionPolicy(),
        budget=cast("Any", budget),
        catalog_port=_catalog(_DEFAULT_MODEL),
        client_factory=factory,
        token_provider=lambda _selected, _region: "local-signing-token",
        exact_counter=StaticTokenCounter(12),
        session_factory=lambda **_kwargs: FakeSession(),
    )

    assert result["mode"] == "live_service"
    assert result["status"] == "blocked"
    assert [item["operation"] for item in result["operations"]] == [
        "ListFoundationModels",
        "provide_token",
        "chat.completions.create",
        "responses.create",
    ]
    signing = result["operations"][1]
    assert signing["mode"] == "local_execution"
    assert signing["effect"] == "none"
    assert signing["reserved_usd"] is None
    assert result["operations"][2]["error_code"] == "budget_exceeded"
    assert result["operations"][3]["error_code"] == "not_supported_by_model"
    assert factory.clients == []
    assert (
        "missing model price"
        in _data(result)["pricing_errors"]["chat.completions.create"]
    )


def test_bedrock_openai_live_token_failure_blocks_chat_only() -> None:
    result = run_bedrock_openai_demo(
        execution="live",
        settings=Settings(),
        policy=ExecutionPolicy(),
        budget=cast("Any", FakeBudget()),
        catalog_port=_catalog(_DEFAULT_MODEL),
        client_factory=FakeFactory(),
        token_provider=lambda _selected, _region: (_ for _ in ()).throw(
            RuntimeError("no credentials"),
        ),
        exact_counter=StaticTokenCounter(12),
        session_factory=lambda **_kwargs: FakeSession(),
    )

    assert result["mode"] == "live_service"
    assert [item["operation"] for item in result["operations"]] == [
        "ListFoundationModels",
        "provide_token",
        "chat.completions.create",
        "responses.create",
    ]
    assert [item["error_code"] for item in result["operations"]] == [
        None,
        "missing_configuration",
        "missing_configuration",
        "not_supported_by_model",
    ]


def test_local_signing_survives_absent_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    factory = FakeFactory()
    result = run_bedrock_openai_demo(
        execution="live",
        settings=Settings(),
        policy=ExecutionPolicy(),
        catalog_port=_catalog(_DEFAULT_MODEL),
        client_factory=factory,
        token_provider=lambda _selected, _region: "local-signing-token",
        session_factory=lambda **_kwargs: FakeSession(),
    )
    assert result["operations"][1]["mode"] == "local_execution"
    assert result["operations"][2]["mode"] == "not_run"
    assert (
        "snapshot is unavailable"
        in _data(result)["pricing_errors"]["chat.completions.create"]
    )
    assert factory.clients == []


def test_one_local_signature_is_reused_when_a_second_api_is_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Inject a synthetic capability to exercise reuse, not AWS support.
    monkeypatch.setitem(
        demo._OPENAI_API_SUPPORT,
        _DEFAULT_MODEL,
        frozenset({"CHAT_COMPLETIONS", "RESPONSES"}),
    )
    token_calls: list[str] = []

    def sign(_selected: SelectedSession, _region: str) -> str:
        token_calls.append("signed")
        return "local-signing-token"

    result = run_bedrock_openai_demo(
        execution="live",
        settings=Settings(),
        policy=ExecutionPolicy(),
        budget=cast("Any", FakeBudget(fail_on="responses.create")),
        catalog_port=_catalog(_DEFAULT_MODEL),
        client_factory=FakeFactory(),
        token_provider=sign,
        session_factory=lambda **_kwargs: FakeSession(),
    )
    assert token_calls == ["signed"]
    assert [item["operation"] for item in result["operations"]].count(
        "provide_token"
    ) == 1
    assert result["operations"][-1]["error_code"] == "budget_exceeded"
    assert (
        "missing model price"
        in _data(result)["pricing_errors"]["responses.create"]
    )


def test_bedrock_openai_live_uses_documented_capability_map() -> None:
    budget = FakeBudget()
    result = run_bedrock_openai_demo(
        execution="live",
        settings=Settings(model=_DEFAULT_MODEL),
        policy=ExecutionPolicy(),
        budget=cast("Any", budget),
        catalog_port=FakeCatalogPort(
            [
                {
                    "providerName": "OpenAI",
                    "modelId": _DEFAULT_MODEL,
                    "supportedOpenAiApis": ["RESPONSES"],
                },
            ],
        ),
        client_factory=FakeFactory(),
        token_provider=lambda _selected, _region: "bedrock-live-token",
        exact_counter=StaticTokenCounter(12),
        session_factory=lambda **_kwargs: FakeSession(),
    )

    assert result["operations"][2]["operation"] == "chat.completions.create"
    assert result["operations"][2]["mode"] == "live_model"
    assert result["operations"][3]["operation"] == "responses.create"
    assert result["operations"][3]["error_code"] == "not_supported_by_model"
    assert [call[0] for call in budget.calls] == ["chat.completions.create"]


def test_bedrock_openai_live_requested_model_must_be_in_catalog() -> None:
    result = run_bedrock_openai_demo(
        execution="live",
        settings=Settings(model="openai.missing"),
        policy=ExecutionPolicy(),
        budget=cast("Any", FakeBudget()),
        catalog_port=_catalog(_DEFAULT_MODEL),
        client_factory=FakeFactory(),
        token_provider=lambda _selected, _region: pytest.fail("token used"),
        exact_counter=StaticTokenCounter(12),
        session_factory=lambda **_kwargs: pytest.fail("session used"),
    )

    assert [item["error_code"] for item in result["operations"]] == [
        "model_unavailable",
        None,
        "model_unavailable",
        "model_unavailable",
    ]


def test_bedrock_openai_live_rejects_unverified_catalog_model() -> None:
    result = run_bedrock_openai_demo(
        execution="live",
        settings=Settings(),
        policy=ExecutionPolicy(),
        budget=cast("Any", FakeBudget()),
        catalog_port=_catalog("openai.demo"),
        client_factory=FakeFactory(),
        token_provider=lambda _selected, _region: pytest.fail("token used"),
        exact_counter=StaticTokenCounter(12),
        session_factory=lambda **_kwargs: pytest.fail("session used"),
    )

    assert result["mode"] == "live_service"
    assert [item["error_code"] for item in result["operations"]] == [
        "model_unavailable",
        None,
        "model_unavailable",
        "model_unavailable",
    ]


@pytest.mark.parametrize(
    ("http_status", "message", "error_code", "row_status"),
    [
        (400, "model is not supported", "model_unavailable", "blocked"),
        (403, "access denied", "authorization_denied", "blocked"),
        (429, "slow down", "throttled", "blocked"),
        (500, "internal error", "validation_failed", "error"),
    ],
)
def test_bedrock_openai_real_sdk_http_errors_are_recorded(
    http_status: int,
    message: str,
    error_code: str,
    row_status: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            http_status,
            request=request,
            json={"error": {"message": message}},
        )

    result = run_bedrock_openai_demo(
        execution="live",
        settings=Settings(),
        policy=_no_retry_policy(),
        budget=cast("Any", FakeBudget()),
        catalog_port=_catalog(_DEFAULT_MODEL),
        client_factory=_sdk_factory(httpx.MockTransport(handler)),
        token_provider=lambda _selected, _region: "bedrock-live-token",
        exact_counter=StaticTokenCounter(12),
        session_factory=lambda **_kwargs: FakeSession(),
    )

    blocked = result["operations"][2]
    assert result["mode"] == "live_service"
    assert blocked["mode"] == "live_service"
    assert blocked["status"] == row_status
    assert blocked["http_status"] == http_status
    assert blocked["error_code"] == error_code


def test_bedrock_openai_real_sdk_connection_error_is_attempt_failed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        message = "connection refused"
        raise httpx.ConnectError(message, request=request)

    result = run_bedrock_openai_demo(
        execution="live",
        settings=Settings(),
        policy=_no_retry_policy(),
        budget=cast("Any", FakeBudget()),
        catalog_port=_catalog(_DEFAULT_MODEL),
        client_factory=_sdk_factory(httpx.MockTransport(handler)),
        token_provider=lambda _selected, _region: "bedrock-live-token",
        exact_counter=StaticTokenCounter(12),
        session_factory=lambda **_kwargs: FakeSession(),
    )

    failed = result["operations"][2]
    assert result["mode"] == "live_service"
    assert failed["mode"] == "attempt_failed"
    assert failed["response_received"] is False
    assert failed["error_code"] == "timeout"


def test_bedrock_openai_os_error_is_attempt_failed() -> None:
    result = run_bedrock_openai_demo(
        execution="live",
        settings=Settings(),
        policy=ExecutionPolicy(),
        budget=cast("Any", FakeBudget()),
        catalog_port=_catalog(_DEFAULT_MODEL),
        client_factory=FakeFactory(fail_chat=OSError("socket closed")),
        token_provider=lambda _selected, _region: "bedrock-live-token",
        exact_counter=StaticTokenCounter(12),
        session_factory=lambda **_kwargs: FakeSession(),
    )

    failed = result["operations"][2]
    assert failed["mode"] == "attempt_failed"
    assert failed["response_received"] is False


def test_bedrock_openai_live_unsupported_error_is_model_unavailable() -> None:
    factory = FakeFactory(fail_chat=UnsupportedModelError("nope"))
    result = run_bedrock_openai_demo(
        execution="live",
        settings=Settings(),
        policy=ExecutionPolicy(),
        budget=cast("Any", FakeBudget()),
        catalog_port=_catalog(_DEFAULT_MODEL),
        client_factory=factory,
        token_provider=lambda _selected, _region: "bedrock-live-token",
        exact_counter=StaticTokenCounter(12),
        session_factory=lambda **_kwargs: FakeSession(),
    )

    assert result["operations"][2]["error_code"] == "model_unavailable"
    assert result["operations"][2]["http_status"] is None

    response_status = FakeFactory(
        fail_chat=ResponseStatusError(404, "missing model"),
    )
    response_result = run_bedrock_openai_demo(
        execution="live",
        settings=Settings(),
        policy=ExecutionPolicy(),
        budget=cast("Any", FakeBudget()),
        catalog_port=_catalog(_DEFAULT_MODEL),
        client_factory=response_status,
        token_provider=lambda _selected, _region: "bedrock-live-token",
        exact_counter=StaticTokenCounter(12),
        session_factory=lambda **_kwargs: FakeSession(),
    )
    assert response_result["operations"][2]["http_status"] == 404


def test_bedrock_openai_live_missing_catalog_sets_token_reason() -> None:
    outcome, payload = _list_openai_catalog(
        settings=Settings(),
        catalog_port=None,
    )

    assert payload == {}
    assert outcome["error_code"] == "missing_configuration"
    assert _token_skip_code([outcome]) == "missing_configuration"


def test_bedrock_openai_response_live_call_branch_and_unknown_error() -> None:
    selected = SelectedSession(
        session=FakeSession(),
        source="default_chain",
        region="us-east-1",
    )
    outcome, state = _reserve_and_call_model(
        operation_name="responses.create",
        fixture_id="unused",
        model="us.openai.gpt-5.6-sol/global",
        request_body=responses_request("us.openai.gpt-5.6-sol/global"),
        settings=Settings(),
        policy=ExecutionPolicy(),
        budget=cast("Any", FakeBudget()),
        exact_counter=StaticTokenCounter(12),
        client_factory=FakeFactory(),
        token_provider=lambda _selected, _region: pytest.fail("token used"),
        session_factory=None,
        csv_loader=None,
        selected=selected,
        api_key="existing-key",
        base_url=bedrock_openai_base_url("us-east-1"),
    )

    assert outcome["mode"] == "live_model"
    assert state.token_operation == ()
    with pytest.raises(ValueError, match="boom"):
        _reserve_and_call_model(
            operation_name="chat.completions.create",
            fixture_id="unused",
            model=_DEFAULT_MODEL,
            request_body=chat_completion_request(_DEFAULT_MODEL),
            settings=Settings(),
            policy=ExecutionPolicy(),
            budget=cast("Any", FakeBudget()),
            exact_counter=StaticTokenCounter(12),
            client_factory=FakeFactory(fail_chat=ValueError("boom")),
            token_provider=lambda _selected, _region: pytest.fail(
                "token used"
            ),
            session_factory=None,
            csv_loader=None,
            selected=selected,
            api_key="existing-key",
            base_url=bedrock_openai_base_url("us-east-1"),
        )


def test_openai_sdk_port_uses_injected_chat_callback() -> None:
    calls: list[dict[str, object]] = []

    def chat_create(**kwargs: object) -> object:
        calls.append(dict(kwargs))
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="callback ok"),
                ),
            ],
            usage=SimpleNamespace(
                prompt_tokens=1,
                completion_tokens=2,
                total_tokens=3,
            ),
        )

    port = OpenAISdkPort(object(), chat_create=chat_create)
    response = port.chat_completion(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=3,
    )

    assert calls == [
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 3,
        },
    ]
    assert response.text == "callback ok"
    assert response.usage == {
        "input_tokens": 1,
        "output_tokens": 2,
        "total_tokens": 3,
    }


def test_bedrock_openai_helpers_and_marker() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "chatcmpl-marker",
                "object": "chat.completion",
                "created": 1,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "marker chat",
                        },
                        "finish_reason": "stop",
                    },
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    base_url = bedrock_openai_base_url("eu-west-1")
    client = default_openai_client(
        api_key="bedrock-key",
        base_url=base_url,
        policy=ExecutionPolicy(),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    chat = client.chat_completion(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=2,
    )
    selected = SelectedSession(
        session=FakeSession(),
        source="default_chain",
        region="us-east-1",
    )
    token = generate_bedrock_api_key(selected, "us-east-1")

    assert client is not None
    assert chat.text == "marker chat"
    assert token.startswith("bedrock-api-key-")
    assert StaticTokenCounter(7).count_tokens("m", b"{}") == 7
    assert chat_completion_request("m")["model"] == "m"
    assert responses_request("m")["model"] == "m"
    assert _credentials_from_session(InternalOnlySession()) is not None
    with pytest.raises(RuntimeError):
        _credentials_from_session(object())
    assert _chat_usage(None) is None
    assert _chat_usage(object()) == {}
    assert _responses_usage(None) is None
    assert _responses_usage(object()) == {}
    lines = _marker_lines(
        "src/awsai_demo/bedrock_openai_demo.py",
        "openai-client",
    )
    assert len(lines) <= 14
    assert all(len(line) <= 72 for line in lines)
    assert "OpenAI(" in "\n".join(lines)
    assert "chat.completions.create(" in "\n".join(lines)


def _no_retry_policy() -> ExecutionPolicy:
    return ExecutionPolicy(total_max_attempts=1, openai_max_retries=0)


def _sdk_factory(transport: httpx.BaseTransport) -> object:
    def factory(
        *,
        api_key: str,
        base_url: str,
        policy: ExecutionPolicy,
    ) -> OpenAISdkPort:
        del policy
        from openai import OpenAI

        return OpenAISdkPort(
            OpenAI(
                api_key=api_key,
                base_url=base_url,
                max_retries=0,
                http_client=cast(
                    "Any",
                    httpx.Client(transport=transport),
                ),
            ),
        )

    return factory


def _catalog(model_id: str) -> FakeCatalogPort:
    return FakeCatalogPort(
        [
            {
                "providerName": "OpenAI",
                "modelId": model_id,
                "modelLifecycle": {"status": "ACTIVE"},
                "inputModalities": ["TEXT"],
                "outputModalities": ["TEXT"],
            },
        ],
    )


def _marker_lines(path: str, marker: str) -> list[str]:
    start = f"# slide: {marker}"
    end = f"# end-slide: {marker}"
    in_marker = False
    lines: list[str] = []
    with Path(path).open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line.strip() == start:
                in_marker = True
                continue
            if line.strip() == end:
                break
            if in_marker:
                lines.append(line)
    return [line.removeprefix("    ") for line in lines]
