# Lab 19: production serving, Kubernetes, autoscaling, observability

**Status:** Planned  
**Baseten posting items:** "large-scale distributed serving: autoscaling, load balancing, multi-region or multi-cloud capacity"; "operational excellence and incident response"; overlap with Baseten's Infrastructure and Observability roles (Kubernetes, Prometheus/Grafana/OpenTelemetry, SLOs)  
**Needs:** inexpensive GPU(s), about $5-10, plus time learning Kubernetes · **Depends on:** Labs 05, 08

## 1. The question
Can I serve the model reliably, scale it on demand, see what is happening, and prove it survives failure?

## 2. Mental model

### 2.1 Containers and Kubernetes in five ideas
- **Container image**: the server and its dependencies packaged to run identically anywhere.
- **Pod**: the smallest unit Kubernetes runs (one or more containers); requests resources such as `nvidia.com/gpu: 1`.
- **Deployment**: keeps N replicas of a pod running, replaces failures, performs rolling updates.
- **Service**: a stable network address that load-balances across the pods behind it.
- **Scheduler and nodes**: the cluster places pods on nodes that have the requested resources. GPU nodes need the NVIDIA device plugin (the GPU Operator installs it, plus the DCGM exporter for GPU metrics). **Taints/tolerations** and node selectors keep GPU pods on GPU nodes.
- **Helm** templates the YAML so one command deploys a configurable release.

### 2.2 Probes and shutdown, which matter more for LLMs
- Loading a model takes minutes. A **startup probe** gives it time; a **readiness probe** (a real inference-ready check, not just "port open") keeps traffic away until the model is loaded; a **liveness probe** restarts a hung server.
- **Graceful shutdown**: on termination the pod should stop accepting new requests, finish in-flight streaming responses, then exit; set `terminationGracePeriodSeconds` longer than your longest request. Otherwise a rollout truncates user responses.

### 2.3 Autoscaling an LLM service
- **CPU is a poor signal**: the GPU is the bottleneck, and CPU usage stays low even when the service is saturated. Scale on **running + queued requests per replica**, KV usage, or queue time, exposed as Prometheus metrics (KEDA or an HPA with custom metrics).
- **Scale-up is slow** (schedule a GPU node, pull the image, load weights: minutes). Mitigate with a minimum replica count, headroom (scale before saturation), cached weights on a shared volume, and scale-down delays to avoid thrashing.
- Cost vs latency: headroom costs money; no headroom costs latency spikes during bursts.

### 2.4 Observability
- **Metrics** (numbers over time: engine Prometheus metrics, GPU metrics from DCGM, node and cluster metrics) → dashboards and alerts.
- **Logs** (structured, with request IDs) → what happened.
- **Traces** (OpenTelemetry spans following a request through gateway, router, engine) → where time went.
- **SLI / SLO / error budget**: an SLI is a measured quantity (for example the fraction of requests with TTFT < 500 ms); the SLO is the target (99% over 30 days); the error budget is what you may miss. Alert on **burn rate** (budget being consumed too fast) rather than every blip, to reduce noise.
- For LLM serving the key SLIs are TTFT, ITL/TPOT, error rate, queue time and goodput.

### 2.5 Reliability practice
Failure drills (kill a pod, saturate the GPU, push a bad config) find weaknesses before customers do. After an incident: a **blameless post-mortem** with timeline, impact, root cause, what detected it, what fixed it, and follow-ups.

### 2.6 Multi-region and multi-cloud capacity (design topic)
Baseten's own posts describe managing capacity across several clouds. The issues a design document should address:
- **Why**: GPU supply is scarce and uneven; a single provider or region may not have capacity when needed; customers have data-residency and latency needs; failures of a whole region must not take the service down.
- **Capacity management**: a view of what each cloud and region has available, pricing and quotas; placing a deployment where capacity exists; **overflow** from a saturated region to another; reserved vs on-demand vs spot capacity and what each failure mode costs.
- **Routing**: send users to the nearest healthy region; keep **prefix-cache locality** within a region (Lab 16); fail over without a thundering herd.
- **Weights and artifacts**: model weights must be present near the GPUs; cold-start time is dominated by loading them, so replicate or cache weights per region and measure the loading time.
- **Consistency of the platform**: the same container images, metrics and alerts everywhere, with per-region SLOs; provider differences in networking, GPU types (an H100 on one cloud is not identical to another) and observability.
- **Failure scenarios** to analyse: region outage, provider quota exhaustion, a bad rollout in one region, a weights-store outage.
Because a real multi-cloud deployment is outside this plan's budget, the deliverable is a **design document** with the decisions, the reasoning, the numbers you can measure cheaply (weight-loading time, cold-start time, cross-region latency) and the failure analysis. Say clearly which parts were measured and which were designed.

## 3. Predictions (fill in BEFORE running)
| Quantity | Prediction | Real | Verdict |
|---|---|---|---|
| Time from scale-up trigger to a ready replica | | | |
| Requests dropped when a pod is killed mid-stream (with/without graceful shutdown) | | | |
| Autoscaler behavior on a step load increase | | | |
| Metric that first shows saturation | | | |

## 4. Steps and why
1. **Containerize** the server with a pinned vLLM version. *Reproducibility.*
2. **Local cluster first** (k3s or kind on a GPU machine, or a managed cluster): install the GPU Operator; run a trivial GPU pod. *Verify GPU scheduling before adding complexity.*
3. **Helm chart** for the server: resources, probes, shutdown, model cache volume.
4. **Metrics stack**: Prometheus, Grafana, DCGM exporter; scrape the engine's `/metrics`.
5. **Autoscaling** on running/queued requests; test with open-loop load from the harness (step, ramp, burst).
6. **SLOs and alerts** for TTFT, ITL, errors; tune out noise.
7. **Tracing** with OpenTelemetry across a thin gateway and the engine.
8. **Failure drills** and a post-mortem for each.

## 5. Results, 6. Interpretation, 7. Engineering takeaway, 8. Pitfalls
_To fill in._ Pitfalls: readiness that passes before the model is loaded; autoscaling on CPU; GPU nodes without taints grabbing non-GPU pods; metrics with high-cardinality labels; alerts on single spikes.

## 9. Done when
- One-command deploy.
- Dashboard screenshot and an SLO document.
- A post-mortem of at least one injected failure.
- A multi-region/multi-cloud design document (decisions, failure analysis, and the cold-start and weight-loading numbers you measured).

## 10. Interview questions
1. Why is CPU a poor autoscaling signal for an LLM server?
2. How would you set an alert on TTFT without paging on noise?
3. Walk through what happens to in-flight requests when a pod is killed, and how you would reduce harm.
4. How do you keep cold-start time down when scaling out?
5. How would you design capacity management across several clouds, and what are the failure modes?

## Further reading
- Beyer et al., "Site Reliability Engineering" and "The Site Reliability Workbook": SLOs, error budgets, burn-rate alerts, post-mortems.
- Kubernetes documentation: Deployments, probes, HPA, taints and tolerations.
- KEDA documentation (event-driven autoscaling).
- NVIDIA GPU Operator and DCGM exporter documentation.
- Prometheus and OpenTelemetry documentation.
