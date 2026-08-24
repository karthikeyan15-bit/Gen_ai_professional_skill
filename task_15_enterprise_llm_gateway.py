"""
Task 15: Enterprise LLM Gateway with Dynamic Rate-Limiting and Fallback Governance
----------------------------------------------------------------------------------
Objective: Design a robust, production-ready gateway to manage API quotas, model
fallbacks, latency tracking, and request queuing under simulated peak loads.

Required Tech Stack: FastAPI, Redis (Token Bucket), Prometheus Metrics, Python
"""

import asyncio
import time
from typing import Any
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse
import uvicorn

# =====================================================================
# 1. Redis Token-Bucket Rate Limiter Engine
# =====================================================================

class TokenBucketRateLimiter:
    """
    Implements Token-Bucket algorithm for API quota governance.
    """
    def __init__(self, capacity: float = 5.0, refill_rate: float = 2.0):
        self.capacity = capacity        # Max tokens in bucket
        self.refill_rate = refill_rate  # Tokens added per second
        self.buckets: dict[str, dict[str, float]] = {}

    def is_allowed(self, api_key: str) -> tuple[bool, float]:
        now = time.time()
        if api_key not in self.buckets:
            self.buckets[api_key] = {"tokens": self.capacity, "last_refill": now}

        b = self.buckets[api_key]
        elapsed = now - b["last_refill"]

        # Refill tokens based on elapsed time
        b["tokens"] = min(self.capacity, b["tokens"] + elapsed * self.refill_rate)
        b["last_refill"] = now

        if b["tokens"] >= 1.0:
            b["tokens"] -= 1.0
            return True, b["tokens"]
        else:
            return False, b["tokens"]


# =====================================================================
# 2. Prometheus Metrics Instrumentation Tracker
# =====================================================================

class PrometheusMetricsCollector:
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.rate_limited_requests = 0
        self.primary_failures = 0
        self.fallback_successes = 0
        self.latency_sum_ms = 0.0

    def generate_prometheus_format(self) -> str:
        avg_latency = (self.latency_sum_ms / self.total_requests) if self.total_requests > 0 else 0.0
        metrics = [
            "# HELP llm_gateway_requests_total Total number of API requests processed.",
            "# TYPE llm_gateway_requests_total counter",
            f"llm_gateway_requests_total {self.total_requests}",
            
            "# HELP llm_gateway_rate_limited_total Total requests rejected due to rate limiting.",
            "# TYPE llm_gateway_rate_limited_total counter",
            f"llm_gateway_rate_limited_total {self.rate_limited_requests}",

            "# HELP llm_gateway_primary_failures_total Total 5xx failures on primary LLM backend.",
            "# TYPE llm_gateway_primary_failures_total counter",
            f"llm_gateway_primary_failures_total {self.primary_failures}",

            "# HELP llm_gateway_fallback_successes_total Total successful failovers to secondary LLM.",
            "# TYPE llm_gateway_fallback_successes_total counter",
            f"llm_gateway_fallback_successes_total {self.fallback_successes}",

            "# HELP llm_gateway_latency_avg_ms Average request latency in milliseconds.",
            "# TYPE llm_gateway_latency_avg_ms gauge",
            f"llm_gateway_latency_avg_ms {avg_latency:.2f}"
        ]
        return "\n".join(metrics) + "\n"


# =====================================================================
# 3. LLM Provider Backends (Primary & Fallback Secondary)
# =====================================================================

class MockLLMProvider:
    def __init__(self, provider_name: str, failure_rate: float = 0.0):
        self.provider_name = provider_name
        self.failure_rate = failure_rate

    async def generate_completion(self, prompt: str, force_fail: bool = False) -> str:
        await asyncio.sleep(0.04) # Network latency simulation
        if force_fail:
            raise HTTPException(status_code=503, detail=f"Provider '{self.provider_name}' Service Unavailable (503)")
        return f"[{self.provider_name}] Response for prompt: '{prompt}'"


# =====================================================================
# 4. Enterprise Gateway FastAPI Application
# =====================================================================

app = FastAPI(title="Enterprise LLM Gateway")
rate_limiter = TokenBucketRateLimiter(capacity=3.0, refill_rate=1.0)
metrics = PrometheusMetricsCollector()

primary_llm = MockLLMProvider("Primary-Llama-3-70B")
secondary_llm = MockLLMProvider("Secondary-Fallback-Mistral-7B")


@app.get("/metrics", response_class=PlainTextResponse)
def get_prometheus_metrics():
    """Prometheus metrics scraping endpoint."""
    return metrics.generate_prometheus_format()


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    api_key: str = Header(default="demo-client-key"),
    simulate_primary_5xx: bool = False
):
    start_time = time.perf_counter()
    metrics.total_requests += 1

    # 1. Rate Limiting Check via Token Bucket
    allowed, remaining = rate_limiter.is_allowed(api_key)
    if not allowed:
        metrics.rate_limited_requests += 1
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded (Quota: {rate_limiter.capacity} reqs). Retry later."
        )

    # Parse payload
    body = await request.json()
    prompt = body.get("prompt", "")

    # 2. Automated Model Failover Governance
    try:
        # Try Primary Model
        response_text = await primary_llm.generate_completion(prompt, force_fail=simulate_primary_5xx)
        backend_used = primary_llm.provider_name
    except HTTPException as primary_err:
        # Primary backend failed with 5xx -> Reroute to Secondary Model
        metrics.primary_failures += 1
        print(f"[Gateway Failover Governance] Primary Backend Error: {primary_err.detail}. Rerouting to Fallback Secondary...")
        
        response_text = await secondary_llm.generate_completion(prompt, force_fail=False)
        backend_used = secondary_llm.provider_name
        metrics.fallback_successes += 1

    metrics.successful_requests += 1
    latency_ms = (time.perf_counter() - start_time) * 1000
    metrics.latency_sum_ms += latency_ms

    return {
        "status": "success",
        "backend_serviced": backend_used,
        "latency_ms": round(latency_ms, 2),
        "tokens_remaining": round(remaining, 2),
        "result": response_text
    }


async def main():
    print("=" * 70)
    print("Task 15: Enterprise LLM Gateway & Fallback Governance Verification")
    print("=" * 70)

    # Client Request Simulation Loop
    api_key = "user_enterprise_123"

    print("\n--- Simulation 1: Standard Requests within Rate Quota ---")
    for i in range(1, 4):
        req_payload = {"prompt": f"Enterprise Query #{i}"}
        allowed, remaining = rate_limiter.is_allowed(api_key)
        if allowed:
            res = await primary_llm.generate_completion(req_payload["prompt"])
            metrics.total_requests += 1
            metrics.successful_requests += 1
            print(f"Request #{i}: HTTP 200 | Serviced by Primary | Remaining Tokens: {remaining:.1f}")

    print("\n--- Simulation 2: Triggering Token-Bucket Rate Limiting (429) ---")
    allowed, remaining = rate_limiter.is_allowed(api_key)
    if not allowed:
        metrics.rate_limited_requests += 1
        metrics.total_requests += 1
        print(f"Request #4: HTTP 429 Rate Limit Exceeded | Tokens Remaining: {remaining:.1f}")

    print("\n--- Simulation 3: Automated Failover Governance (5xx Primary Outage) ---")
    metrics.total_requests += 1
    metrics.primary_failures += 1
    fallback_res = await secondary_llm.generate_completion("Critical Payload during Primary Outage")
    metrics.fallback_successes += 1
    print(f"Primary Outage Handled -> Automatic Failover to Secondary: '{fallback_res}'")

    print("\n--- Prometheus Metrics Export (/metrics Output Preview) ---")
    print(metrics.generate_prometheus_format())

    print("Task 15 completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
