from __future__ import annotations

import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_TRACING_INITIALIZED = False


def init_tracing(service_name: str | None = None) -> None:
    global _TRACING_INITIALIZED
    if _TRACING_INITIALIZED:
        return

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "localhost:4317")
    if endpoint.startswith("http://"):
        endpoint = endpoint[len("http://") :]
    elif endpoint.startswith("https://"):
        endpoint = endpoint[len("https://") :]

    insecure_env = os.getenv("OTEL_EXPORTER_OTLP_INSECURE")
    if insecure_env is None:
        insecure = True
    else:
        insecure = insecure_env.strip().lower() in {"true", "1", "yes"}

    service = service_name or os.getenv("OTEL_SERVICE_NAME", "researcher-multi-agent")
    resource = Resource(attributes={"service.name": service})

    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)
    exporter = OTLPSpanExporter(endpoint=endpoint, insecure=insecure)
    provider.add_span_processor(BatchSpanProcessor(exporter))

    _TRACING_INITIALIZED = True
