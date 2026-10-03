"""Registry contract tests; fake Agent never loads framework runtimes."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from registry import AgentRegistry


class FakeAgent:
    def __init__(self, agent_id='arbitrary-worker', backend='onnxruntime', device='cpu', status='READY', loaded=False):
        self.agent_id, self.backend, self.device = agent_id, backend, device
        self.status, self.model_loaded = status, loaded
        self.hardware = {'nested': [1]}
        self.artifact_metadata = {}
        self.calls = []
        self.fail_load = self.fail_profile = self.fail_cleanup = None

    def get_info(self):
        return dict(agent_id=self.agent_id, backend=self.backend, device=self.device,
                    provider='test-provider', status=self.status, hardware=self.hardware,
                    capabilities={'model_formats': ['onnx'], 'supported_actions': ['profile']},
                    model_loaded=self.model_loaded)

    def get_status(self):
        return self.status

    def stop(self):
        self.status = 'STOPPED'

    def load_model(self, manifest):
        self.calls.append(('load', manifest.framework))
        if self.fail_load:
            self.status = 'ERROR'
            raise self.fail_load
        self.model_loaded = True
        self.artifact_metadata = {'model_id': manifest.model_id, 'artifact_path': Path('model.onnx'), 'nested': [1]}

    def profile(self, output_path=None):
        self.calls.append(('profile', output_path))
        if self.fail_profile:
            self.status = 'ERROR'
            raise self.fail_profile
        return Path(output_path or 'reports/test.json')

    def unload_model(self, *, preserve_error=False):
        self.calls.append(('unload', preserve_error))
        if self.fail_cleanup:
            self.status = 'ERROR'
            raise self.fail_cleanup
        self.model_loaded = False
        self.artifact_metadata.clear()
        if not (preserve_error and self.status == 'ERROR'):
            self.status = 'READY'


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.registry = AgentRegistry()
        self.agent = FakeAgent()

    def test_register_and_duplicate(self):
        record = self.registry.register_agent(self.agent)
        self.assertEqual(record.agent_id, self.agent.agent_id)
        self.assertIs(self.registry.get_agent(record.agent_id), self.agent)
        with self.assertRaises(ValueError):
            self.registry.register_agent(FakeAgent())

    def test_missing_ids(self):
        for action in (self.registry.get_agent, self.registry.unregister_agent):
            with self.assertRaises(KeyError):
                action('missing')

    def test_filters_and_order(self):
        for a in (self.agent, FakeAgent('two', device='gpu'), FakeAgent('three', backend='pytorch', status='ERROR')):
            self.registry.register_agent(a)
        self.assertEqual([r.agent_id for r in self.registry.list_agents()], ['arbitrary-worker', 'two', 'three'])
        for filters, expected in [({'backend':'onnxruntime'},2), ({'device':'gpu'},1), ({'status':'ERROR'},1),
                                  ({'backend':'onnxruntime','device':'gpu','status':'READY'},1), ({'backend':'unknown'},0)]:
            self.assertEqual(len(self.registry.find_agents(**filters)), expected)
        with self.assertRaises(ValueError):
            self.registry.find_agents(status='INVALID')

    def test_snapshot_and_live_status(self):
        record = self.registry.register_agent(self.agent)
        record.hardware_info['nested'].append(2)
        copy = record.to_dict()
        copy['hardware_info']['nested'].append(3)
        self.assertEqual(self.agent.hardware, {'nested': [1]})
        self.assertEqual(record.hardware_info['nested'], [1,2])
        json.dumps(copy)
        self.agent.stop()
        self.assertEqual(self.registry.list_agents()[0].status, 'STOPPED')
        self.assertEqual(record.status, 'READY')

    def test_status_confirmation(self):
        self.registry.register_agent(self.agent)
        self.assertEqual(self.registry.update_agent_status(self.agent.agent_id,'READY').status,'READY')
        for status in ('WRONG','STOPPED'):
            with self.assertRaises(ValueError):
                self.registry.update_agent_status(self.agent.agent_id,status)
        self.assertEqual(self.agent.status, 'READY')

    def test_unregister_clear_do_not_stop(self):
        self.registry.register_agent(self.agent)
        self.registry.unregister_agent(self.agent.agent_id)
        self.assertEqual(self.agent.status,'READY')
        self.registry.register_agent(self.agent)
        self.registry.clear()
        self.assertEqual(self.registry.list_agents(),[])
        self.assertEqual(self.agent.status,'READY')

    def test_invalid_metadata(self):
        for key, value in [('agent_id',''),('status','wrong'),('model_loaded',1),('hardware',[]),('capabilities',[])]:
            with self.subTest(key=key):
                info=deepcopy(self.agent.get_info()); info[key]=value
                agent=FakeAgent(); agent.get_info=lambda: info
                with self.assertRaises(ValueError):
                    self.registry.register_agent(agent)


if __name__=='__main__':
    unittest.main()
