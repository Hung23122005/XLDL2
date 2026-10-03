"""Register live Agent metadata without loading any model."""
import json
import sys

from agent.agent import Agent
from agent.config import AgentConfig
from registry import AgentRegistry

CONFIG_PATHS = (
    'configs/agent_onnx_cpu.yaml', 'configs/agent_onnx_gpu.yaml',
    'configs/agent_pytorch_cpu.yaml', 'configs/agent_pytorch_gpu.yaml',
)


def demo_configs(backend=None):
    configs = [AgentConfig.load(path) for path in CONFIG_PATHS]
    return [config for config in configs if backend is None or config.backend == backend]


def main():
    registry = AgentRegistry()
    agents, failures = [], []
    try:
        for config in demo_configs():
            agent = Agent(config)
            agents.append(agent)
            try:
                agent.start()
                registry.register_agent(agent)
            except Exception as error:
                failures.append({'agent_id': config.agent_id, 'error': str(error)})
                print(f'Runtime unavailable for {config.agent_id}: {error}', file=sys.stderr)
        print(json.dumps([record.to_dict() for record in registry.list_agents()], indent=2))
        print('Capabilities are declared/detected; workload execution is not yet verified.')
    finally:
        for agent in agents:
            try:
                agent.stop()
            except Exception as error:
                failures.append({'agent_id': agent.agent_id, 'error': str(error)})
                print(f'Stop failed for {agent.agent_id}: {error}', file=sys.stderr)
        print(json.dumps([{'agent_id': r.agent_id, 'status': r.status} for r in registry.list_agents()], indent=2))
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
