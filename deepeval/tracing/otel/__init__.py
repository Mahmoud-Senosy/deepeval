# The exporter and processor need the OpenTelemetry SDK, which is an optional
# dependency (``pip install "deepeval[otel]"``). They are resolved lazily so
# that ``deepeval.tracing.otel.attributes`` / ``.utils`` stay importable
# without it.

__all__ = [
    "ConfidentSpanExporter",
    "ContextAwareSpanProcessor",
]


def __getattr__(name):
    if name == "ConfidentSpanExporter":
        from .exporter import ConfidentSpanExporter

        return ConfidentSpanExporter
    if name == "ContextAwareSpanProcessor":
        from .context_aware_processor import ContextAwareSpanProcessor

        return ContextAwareSpanProcessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
