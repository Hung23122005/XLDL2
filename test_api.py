"""HTTP contract tests use the real Registry and Orchestrator with fake Agents."""
import unittest
from fastapi.testclient import TestClient

from api.main import create_app
from test_registry import FakeAgent


class StartedAgent(FakeAgent):
    def start(self):
        self.status = 'READY'


class ApiTests(unittest.TestCase):
    body = dict(manifest_path='manifests/resnet18.yaml', backend='onnxruntime',
                device='cpu', action='profile', report_path=None)

    def setUp(self):
        self.agent = StartedAgent()
        self.app = create_app(lambda config: self.agent, ['test-config'])

    def test_health_agents_docs_and_shutdown(self):
        with TestClient(self.app) as client:
            self.assertEqual(client.get('/health').json(), {'status': 'ok'})
            self.assertEqual(client.get('/agents').json()[0]['agent_id'], self.agent.agent_id)
            self.assertEqual(client.get('/docs').status_code, 200)
            self.assertIn('/profile', client.get('/openapi.json').json()['paths'])
        self.assertEqual(self.agent.status, 'STOPPED')

    def test_success_uses_orchestrator_and_unloads(self):
        with TestClient(self.app) as client:
            response = client.post('/profile', json=self.body)
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            self.assertEqual(result['status'], 'success')
            self.assertEqual(result['agent_id'], self.agent.agent_id)
            self.assertTrue(result['request_id'])
            self.assertEqual([call[0] for call in self.agent.calls], ['load', 'profile', 'unload'])
            self.assertFalse(self.agent.model_loaded)

    def test_invalid_request_and_manifest_are_400(self):
        with TestClient(self.app) as client:
            for change in ({'action': 'predict'}, {'manifest_path': 'missing.yaml'}, {'backend': ''}):
                self.assertEqual(client.post('/profile', json={**self.body, **change}).status_code, 400)
            self.assertEqual(self.agent.calls, [])

    def test_no_matching_agent_is_503(self):
        with TestClient(self.app) as client:
            response = client.post('/profile', json={**self.body, 'device': 'gpu'})
            self.assertEqual(response.status_code, 503)
            self.assertEqual(self.agent.calls, [])

    def test_runtime_failure_500_preserves_error_and_cleanup(self):
        self.agent.fail_profile = RuntimeError('profile failed')
        with TestClient(self.app) as client:
            response = client.post('/profile', json=self.body)
            self.assertEqual(response.status_code, 500)
            self.assertIn('profile failed', response.json()['detail'])
            self.assertFalse(self.agent.model_loaded)
            self.assertEqual(client.get('/agents').json()[0]['status'], 'ERROR')
            self.assertEqual(client.post('/profile', json=self.body).status_code, 503)

    def test_failed_start_is_visible_and_stopped(self):
        def fail_start():
            self.agent.status = 'ERROR'
            raise RuntimeError('runtime unavailable')
        self.agent.start = fail_start
        with TestClient(self.app) as client:
            self.assertEqual(client.get('/agents').json()[0]['status'], 'ERROR')
            self.assertEqual(client.post('/profile', json=self.body).status_code, 503)
        self.assertEqual(self.agent.status, 'STOPPED')


if __name__ == '__main__':
    unittest.main()
