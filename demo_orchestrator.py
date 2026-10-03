"""Run real registry/orchestrator requests, isolating each --all runtime case."""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import subprocess
import sys

from agent.agent import Agent
from agent.config import REPO_ROOT
from demo_registry import demo_configs
from orchestration import Orchestrator, ProfilingRequest
from registry import AgentRegistry


def execute_case(backend, device, manifest, report_path=None):
    registry = AgentRegistry()
    agents = []
    failure = None
    try:
        for config in demo_configs(backend):
            agent = Agent(config)
            agents.append(agent)
            try:
                agent.start()
                registry.register_agent(agent)
            except Exception as error:
                print(f'Runtime unavailable for {config.agent_id}: {error}', file=sys.stderr)
        request = ProfilingRequest(manifest_path=manifest, backend=backend, device=device, report_path=report_path)
        result = Orchestrator(registry).execute(request)
        selected = registry.get_agent(result.agent_id)
        if selected.model_loaded or selected.get_status() != 'READY':
            raise RuntimeError('Agent was not unloaded and READY after execute().')
        report = json.loads(result.report_path.read_text(encoding='utf-8'))
        if report.get('model_id') != result.model_id:
            raise RuntimeError('Report model_id does not match ProfilingResult.')
        for key, value in result.artifact_metadata.items():
            if report.get(key) != value:
                raise RuntimeError(f'Report artifact metadata mismatch: {key}')
        return result.to_dict()
    except BaseException as error:
        failure = error
        raise
    finally:
        stop_errors = []
        for agent in agents:
            try:
                agent.stop()
            except Exception as error:
                stop_errors.append(error)
                print(f'Stop failed for {agent.agent_id}: {error}', file=sys.stderr)
        if stop_errors and failure is None:
            raise stop_errors[0]


def run_all(manifest):
    summary = []
    for backend, device in (('onnxruntime','cpu'), ('onnxruntime','gpu'), ('pytorch','cpu'), ('pytorch','gpu')):
        command = [sys.executable, str(Path(__file__).resolve()), '--backend', backend,
                   '--device', device, '--manifest', str(manifest)]
        process = subprocess.run(command, cwd=REPO_ROOT, capture_output=True)
        output = process.stdout.decode('utf-8', errors='replace').replace('\x00','')
        error = process.stderr.decode('utf-8', errors='replace').replace('\x00','')
        item = {'backend': backend, 'device': device}
        try:
            payload = json.loads(output)
            if process.returncode != 0 or payload.get('status') != 'success':
                raise RuntimeError(payload.get('error') or error or f'Exit code {process.returncode}')
            item.update(status='PASS', agent_id=payload['agent_id'], report_path=payload['report_path'], result=payload)
        except (ValueError, KeyError, RuntimeError) as exc:
            item.update(status='FAILED', error=str(exc), runtime_log=error[-6000:])
        summary.append(item)
        print(f"{backend} {device}: {item['status']}", file=sys.stderr, flush=True)
    print(json.dumps(summary, indent=2))
    return int(any(item['status'] != 'PASS' for item in summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--backend')
    parser.add_argument('--device')
    parser.add_argument('--manifest', default='manifests/resnet18.yaml')
    parser.add_argument('--report-path')
    args = parser.parse_args()
    if args.all:
        if args.backend or args.device or args.report_path:
            parser.error('--all cannot be combined with --backend, --device or --report-path')
        return run_all(args.manifest)
    if not args.backend or not args.device:
        parser.error('Provide --backend and --device, or --all')
    try:
        # Keep stdout machine-readable; lifecycle and native runtime logs go to stderr.
        with redirect_stdout(sys.stderr):
            result = execute_case(args.backend, args.device, args.manifest, args.report_path)
        print(json.dumps(result, indent=2))
        return 0
    except Exception as error:
        print(json.dumps({'status': 'failed', 'backend': args.backend, 'device': args.device,
                          'error': f'{type(error).__name__}: {error}'}, indent=2))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
