"""Local synchronous registry. Agent lifecycle remains owned by the caller."""
from agent.agent import AgentStatus
from .models import AgentRecord


class AgentRegistry:
    def __init__(self):
        self._agents = {}

    def register_agent(self, agent):
        record = AgentRecord.from_info(agent.get_info())
        if record.agent_id in self._agents:
            raise ValueError(f"Agent ID already registered: {record.agent_id}")
        self._agents[record.agent_id] = agent
        return record

    def unregister_agent(self, agent_id):
        del self._agents[agent_id]

    def get_agent(self, agent_id):
        agent = self._agents[agent_id]
        self._record(agent_id, agent)
        return agent

    @staticmethod
    def _record(agent_id, agent):
        record = AgentRecord.from_info(agent.get_info())
        if record.agent_id != agent_id:
            raise ValueError(f"Registered Agent identity changed: {agent_id}")
        return record

    def list_agents(self):
        return [self._record(key, agent) for key, agent in self._agents.items()]

    def find_agents(self, backend=None, device=None, status=None):
        if status is not None:
            try:
                status = AgentStatus(status).value
            except (ValueError, TypeError) as error:
                raise ValueError(f"Invalid Agent status filter: {status}") from error
        return [record for record in self.list_agents()
                if (backend is None or record.backend == backend)
                and (device is None or record.device == device)
                and (status is None or record.status == status)]

    def update_agent_status(self, agent_id, status):
        try:
            expected = AgentStatus(status).value
        except (ValueError, TypeError) as error:
            raise ValueError(f"Invalid Agent status: {status}") from error
        agent = self.get_agent(agent_id)
        record = self._record(agent_id, agent)
        if expected != agent.get_status() or expected != record.status:
            raise ValueError(f"Status {expected} does not match live Agent {agent_id}: {agent.get_status()}")
        return record

    def clear(self):
        self._agents.clear()
