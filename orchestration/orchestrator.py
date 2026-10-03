"""Synchronous dispatch only; no concurrency or atomic reservation guarantee."""
from copy import deepcopy
from datetime import datetime, timezone
import logging
from pathlib import Path
from time import perf_counter

from agent.manifest import Manifest
from .request import ProfilingRequest, ProfilingResult
from .errors import NoEligibleAgentError

log = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, registry):
        self.registry = registry

    @staticmethod
    def _no_agent(request):
        return NoEligibleAgentError(
            f"No eligible agent for backend={request.backend!r}, device={request.device!r}. "
            "Required status=READY and model_loaded=False."
        )

    def select_agent(self, candidates, request):
        for record in candidates:
            if (record.backend == request.backend and record.device == request.device
                    and record.status == "READY" and not record.model_loaded):
                return record
        raise self._no_agent(request)

    def _cleanup(self, agent, agent_id, preserve_error):
        error = None
        try:
            agent.unload_model(preserve_error=preserve_error)
        except BaseException as cleanup_error:
            error = cleanup_error
        try:
            self.registry.update_agent_status(agent_id, agent.get_status())
        except BaseException as refresh_error:
            if error is None:
                error = refresh_error
            else:
                log.error("Registry refresh also failed for %s: %s", agent_id, refresh_error)
        return error

    def execute(self, request):
        if not isinstance(request, ProfilingRequest):
            raise TypeError("execute() requires a ProfilingRequest.")
        request.validate()
        manifest = Manifest.load(request.manifest_path)
        candidates = self.registry.find_agents(backend=request.backend, device=request.device, status="READY")
        selected = self.select_agent(candidates, request)
        agent = self.registry.get_agent(selected.agent_id)
        current = agent.get_info()
        if (agent.get_status() != "READY" or current["model_loaded"]
                or current["backend"] != request.backend or current["device"] != request.device):
            raise self._no_agent(request)
        started_at = datetime.now(timezone.utc)
        started = perf_counter()
        original_error = None
        try:
            agent.load_model(manifest)
            report_path = Path(agent.profile(output_path=request.report_path))
            metadata = deepcopy(agent.artifact_metadata)
            model_id = manifest.model_id
        except BaseException as error:
            original_error = error
            raise
        finally:
            cleanup_error = self._cleanup(
                agent, selected.agent_id, preserve_error=original_error is not None,
            )
            if cleanup_error is not None:
                if original_error is not None:
                    raise original_error from cleanup_error
                raise cleanup_error
        return ProfilingResult(
            request_id=request.request_id, agent_id=selected.agent_id,
            backend=request.backend, device=request.device, action=request.action,
            model_id=model_id, status="success", report_path=report_path,
            artifact_metadata=metadata, started_at=started_at,
            finished_at=datetime.now(timezone.utc), duration_ms=(perf_counter()-started)*1000,
        )
