# Lab 14: Production serving: Kubernetes, autoscaling, observability

**Status:** Planned  
**Baseten posting items:** Large-scale distributed serving (autoscaling, load balancing, multi-region/multi-cloud); operational excellence and incident response; Infrastructure and Observability role overlap  
**Needs:** inexpensive GPU(s), ~$5-10, plus time learning Kubernetes · **Depends on:** Labs 05, 08

## 1. The question
Can I serve the model reliably, scale it on demand, see what is happening, and prove it survives failure?

## 2. Plan
1. Containerize the server; deploy on Kubernetes (k3s or a managed cluster) with the NVIDIA GPU operator and a Helm chart.
2. Autoscale on queue depth or running requests (KEDA or HPA with custom metrics), not CPU.
3. Observability: Prometheus and Grafana, GPU metrics via the DCGM exporter, OpenTelemetry traces and structured logs; define SLOs for TTFT, ITL and error rate with low-noise alerts.
4. Load test with the harness in open-loop mode; failure drills (kill a pod, saturate the GPU, bad rollout); write a post-mortem.

## 3. Predictions (write before running)
| Quantity | Prediction (with arithmetic) | Real | Verdict |
|---|---|---|---|


## 4. Done when
- One-command deploy
- Dashboard screenshot and SLO document
- Post-mortem of at least one injected failure

## 5. Results, interpretation, production takeaway, pitfalls
_To fill in using `templates/LAB_TEMPLATE.md`: raw data in `results/`, p50 and p99, repeats, and a check against a hand calculation._

## 6. Interview questions
1. Why is CPU a poor autoscaling signal for an LLM server?
2. How would you set an alert on TTFT without paging on noise?
3. Walk through what happens to in-flight requests when a pod is killed.
