# Registry and Orchestrator Implementation Plan

> For agentic workers: use superpowers:executing-plans. User explicitly requested implementation in the current repo; execute inline without another design gate.

**Goal:** Dispatch synchronous profiling requests to a live in-memory Agent registry.
**Architecture:** AgentRecord snapshots come from live Agent objects. Orchestrator validates, selects, runs and cleans up; Agent retains lifecycle ownership.
**Tech Stack:** Python 3.10, dataclasses, unittest, existing Agent/Manifest APIs.
**Spec:** User attachment 0becf7e9-abf8-4d6f-85ff-0bc20b7ef2b6, superseding docs/superpowers/specs/2026-10-03-registry-orchestrator-design.md with request_id and result timestamps.

## Global constraints
- Work directly in current repo; preserve existing changes and PRoof core.
- No database, API, networking, concurrency, fallback or new converter.
- Keep old unload behavior by default; preserve ERROR only when requested.
- Run each integration backend/device case in a separate subprocess.

## Review focus
- Duplicate IDs and mutable metadata: reject duplicates and return deep JSON-compatible snapshots.
- State changed after selection: recheck eligibility before touching a model.
- Caller-owned model: never unload an already-loaded candidate.
- Workload plus cleanup failures: original exception remains primary.
- GPU advertised but unusable: fail the request and preserve runtime error.

## Tasks
- [x] Write failing tests in test_registry.py and test_orchestrator.py for all public APIs, serialization, selection and cleanup. Run tests and confirm missing implementation failures.
- [x] Implement registry/models.py, registry/registry.py and public exports; verify Registry tests.
- [x] Implement request/result dataclasses and Orchestrator; add preserve_error and capabilities in Agent. Verify fake-Agent tests and old lifecycle behavior.
- [x] Add demo_registry.py and demo_orchestrator.py with single-case and isolated --all modes. Validate reports and return real failures.
- [x] Run new tests plus agent.test_lifecycle, agent.test_backends and artifact.test_artifacts. Run Registry and all four integration cases. Review changes and document actual results.

## Verification results
- 30 unittest cases passed (7 Registry, 12 Orchestrator/Agent contract, 11 existing tests).
- Registry demo registered four live READY agents and observed STOPPED after cleanup.
- Final isolated integration run: ONNX Runtime CPU/GPU and PyTorch CPU/GPU all PASS; four JSON reports validated, exit code 0.
- Review found interruption cleanup could be skipped. Added failing KeyboardInterrupt/SystemExit regression tests, fixed finally cleanup and Agent ERROR preservation, then reran tests and all four integrations.

