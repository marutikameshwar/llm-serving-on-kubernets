# Project Engineering Notes & Decision Log

This document tracks technical decisions, architecture rationales, and milestone notes across each stage of building `llm-serving-k8s`.

---

## Part 1.1: Local Repository Scaffolding & Version Registry
- **Date**: 2026-10-30
- **Summary**: Established the initial directory layout, version registry (`docs/versions.md`), engineering standards (`.agent/rules/engineering-standards.md`), and the master interview study guide (`docs/interview-study-guide.md`).
- **Key Decision**:
  - Pinned k3s to `v1.30.4+k3s1` and vLLM to `v0.6.3.post1` with CUDA 12.4. In GPU environments, unpinned versions lead to CUDA Driver API vs Runtime API mismatches that silently crash tensor kernels.
  - Selected `Qwen/Qwen2.5-7B-Instruct` as the primary evaluation target because its 7B parameter size fits squarely on a cost-effective 24GB GPU (L4 or A10) while providing top-tier coding and reasoning benchmarks.
