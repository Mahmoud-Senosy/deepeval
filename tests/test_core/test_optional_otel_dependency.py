"""OpenTelemetry and grpcio are optional extras (``deepeval[otel]``,
``deepeval[grok]``). Core deepeval must import and run without them, and the
OTel-only pieces must fail with an actionable message rather than a bare
ModuleNotFoundError.

Each check runs in a fresh interpreter with the packages blocked on the import
path, so the result doesn't depend on what happens to be installed.
"""

import os
import subprocess
import sys
import textwrap

import pytest

_BLOCKER = textwrap.dedent("""
    import sys

    class _Block:
        BLOCKED = ("opentelemetry", "grpc")

        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in self.BLOCKED:
                raise ModuleNotFoundError(f"No module named {name!r}")
            return None

    sys.meta_path.insert(0, _Block())
    """)


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _BLOCKER + textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=300,
        env={
            **os.environ,
            "DEEPEVAL_TELEMETRY_OPT_OUT": "1",
            "PYTHONIOENCODING": "utf-8",
        },
    )


@pytest.mark.parametrize(
    "module",
    [
        "deepeval",
        "deepeval.metrics",
        "deepeval.evaluate",
        "deepeval.tracing",
        "deepeval.dataset",
        "deepeval.models",
        "deepeval.models.retry_policy",
        "deepeval.synthesizer",
        "deepeval.tracing.otel",
        "deepeval.tracing.otel.attributes",
        "deepeval.tracing.otel.utils",
        "deepeval.integrations.strands",
        "deepeval.integrations.agentcore",
        "deepeval.integrations.openinference",
    ],
)
def test_module_imports_without_otel_or_grpc(module):
    result = _run(f"import {module}")
    assert result.returncode == 0, result.stderr


def test_observe_runs_without_otel_or_grpc():
    result = _run("""
        from deepeval.tracing import observe

        @observe()
        def app(q):
            return q.upper()

        assert app("hi") == "HI"
        """)
    assert result.returncode == 0, result.stderr


def test_otel_exporter_points_to_extra_when_missing():
    result = _run("""
        from deepeval.tracing.otel import ConfidentSpanExporter
        """)
    assert result.returncode != 0
    assert 'pip install "deepeval[otel]"' in result.stderr
    assert "ModuleNotFoundError" not in result.stderr.splitlines()[-1]
