"""Process transport only. All lifecycle and workload logic stays in Agent."""
import logging
import multiprocessing
import signal

from agent.config import REPO_ROOT

log = logging.getLogger(__name__)
CONFIG_PATHS = tuple(REPO_ROOT / 'configs' / name for name in (
    'agent_onnx_cpu.yaml', 'agent_onnx_gpu.yaml',
    'agent_pytorch_cpu.yaml', 'agent_pytorch_gpu.yaml',
))


def _serve(connection, config):
    # Uvicorn owns Ctrl+C. Children exit through Agent.stop() during lifespan shutdown.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from agent.agent import Agent
    agent = None
    try:
        agent = Agent(config)
        while True:
            method, args, kwargs = connection.recv()
            try:
                result = getattr(agent, method)(*args, **kwargs)
                connection.send((True, result))
            except Exception as error:
                connection.send((False, ('value' if isinstance(error, ValueError) else 'runtime',
                                         f'{type(error).__name__}: {error}')))
            if method == 'stop':
                break
    except EOFError:
        pass
    finally:
        if agent is not None and agent.get_status() != 'STOPPED':
            try:
                agent.stop()
            except Exception:
                log.exception('Worker cleanup failed')
        connection.close()


class ProcessAgent:
    """Forward the Agent interface used by Registry/Orchestrator to a child."""

    def __init__(self, config):
        context = multiprocessing.get_context('spawn')
        self._connection, child = context.Pipe()
        self._process = context.Process(target=_serve, args=(child, config))
        self._process.start()
        child.close()

    def _call(self, method, *args, _timeout=None, **kwargs):
        try:
            self._connection.send((method, args, kwargs))
            if _timeout is not None and not self._connection.poll(_timeout):
                raise RuntimeError(f'Agent worker timed out during {method}')
            success, result = self._connection.recv()
        except (EOFError, OSError) as error:
            raise RuntimeError(f'Agent worker unavailable during {method}') from error
        if not success:
            kind, message = result
            raise (ValueError if kind == 'value' else RuntimeError)(message)
        return result

    def start(self):
        return self._call('start')

    def get_info(self):
        return self._call('get_info')

    def get_status(self):
        return self._call('get_status')

    @property
    def artifact_metadata(self):
        return self.get_info()['artifact']

    def load_model(self, manifest):
        return self._call('load_model', manifest)

    def profile(self, output_path=None):
        return self._call('profile', output_path=output_path)

    def unload_model(self, *, preserve_error=False):
        return self._call('unload_model', preserve_error=preserve_error)

    def stop(self):
        try:
            if self._process.is_alive():
                self._call('stop', _timeout=10)
        finally:
            self._connection.close()
            self._process.join(timeout=10)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join()
            self._process.close()
