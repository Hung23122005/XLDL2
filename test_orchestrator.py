"""Synchronous dispatch and failure cleanup contracts."""
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from uuid import UUID

from agent.agent import Agent, AgentStatus
from orchestration import Orchestrator, ProfilingRequest
from registry import AgentRegistry
from test_registry import FakeAgent


class OrchestratorTests(unittest.TestCase):
    def setUp(self):
        self.registry=AgentRegistry()
        self.orchestrator=Orchestrator(self.registry)

    def request(self, **changes):
        values=dict(manifest_path='manifests/resnet18.yaml',backend='onnxruntime',device='cpu')
        values.update(changes)
        return ProfilingRequest(**values)

    def add(self, **kwargs):
        a=FakeAgent(**kwargs); self.registry.register_agent(a); return a

    def test_request_validation_and_immutability(self):
        request=self.request()
        UUID(request.request_id)
        self.assertNotEqual(request.request_id,self.request().request_id)
        with self.assertRaises(FrozenInstanceError):
            request.device='gpu'
        for change in ({'action':'predict'},{'manifest_path':''},{'manifest_path':None},{'backend':''},
                       {'device':' '},{'report_path':''},{'request_id':''},{'report_path':42}):
            with self.subTest(change=change), self.assertRaises(ValueError): self.request(**change)

    def test_success_selection_result_and_cleanup(self):
        self.add(agent_id='wrong', device='gpu')
        a=self.add(agent_id='first'); second=self.add(agent_id='second')
        request=self.request(report_path=Path('reports/override.json'))
        result=self.orchestrator.execute(request)
        self.assertEqual(result.agent_id,'first')
        self.assertEqual(result.request_id,request.request_id)
        self.assertEqual(result.model_id,'resnet18_v1')
        self.assertEqual(result.status,'success')
        self.assertGreaterEqual(result.duration_ms,0)
        self.assertGreaterEqual(result.finished_at,result.started_at)
        self.assertEqual(a.calls,[('load','pytorch'),('profile',Path('reports/override.json')),('unload',False)])
        self.assertEqual(second.calls,[])
        self.assertFalse(a.model_loaded)
        self.assertEqual(a.status,'READY')
        data=result.to_dict(); json.dumps(data)
        self.assertEqual(data['artifact_metadata']['artifact_path'],'model.onnx')
        data['artifact_metadata']['nested'].append(2)
        self.assertEqual(result.artifact_metadata['nested'],[1])

    def test_exact_backend_and_device(self):
        self.add(agent_id='onnx',device='gpu')
        self.add(agent_id='torch_cpu',backend='pytorch')
        self.add(agent_id='desired',backend='pytorch',device='gpu')
        result=self.orchestrator.execute(self.request(backend='pytorch',device='gpu'))
        self.assertEqual(result.agent_id,'desired')

    def test_ineligible_agents_untouched(self):
        agents=[self.add(agent_id=s,status=s) for s in ('BUSY','ERROR','STOPPED')]
        agents.append(self.add(agent_id='owned',loaded=True))
        with self.assertRaisesRegex(RuntimeError,'READY.*model_loaded=False'):
            self.orchestrator.execute(self.request())
        self.assertTrue(all(a.calls==[] for a in agents))
        self.assertTrue(agents[-1].model_loaded)

    def test_missing_and_invalid_manifest_no_mutation(self):
        a=self.add()
        with self.assertRaises(ValueError): self.orchestrator.execute(self.request(manifest_path='missing.yaml'))
        self.assertEqual(a.calls,[])
        with self.assertRaises(TypeError): self.orchestrator.execute({})
        self.assertEqual(a.calls,[])
        self.registry.clear()
        with self.assertRaisesRegex(RuntimeError,"backend='onnxruntime'.*device='cpu'"):
            self.orchestrator.execute(self.request())

    def test_recheck_after_selection(self):
        a=self.add()
        original=self.orchestrator.select_agent
        def select(candidates, request):
            chosen=original(candidates,request); a.model_loaded=True; return chosen
        with patch.object(self.orchestrator,'select_agent',side_effect=select):
            with self.assertRaises(RuntimeError): self.orchestrator.execute(self.request())
        self.assertEqual(a.calls,[])

    def test_workload_errors_preserved_and_unloaded(self):
        for phase in ('load','profile'):
            with self.subTest(phase=phase):
                self.registry.clear(); a=self.add()
                error=RuntimeError('workload failure'); setattr(a,'fail_'+phase,error)
                with self.assertRaises(RuntimeError) as caught: self.orchestrator.execute(self.request())
                self.assertIs(caught.exception,error)
                self.assertEqual(a.calls[-1],('unload',True))
                self.assertFalse(a.model_loaded)
                self.assertEqual(self.registry.list_agents()[0].status,'ERROR')

    def test_double_failure_keeps_original_exception(self):
        a=self.add(); a.fail_profile=ValueError('original'); a.fail_cleanup=RuntimeError('cleanup')
        with self.assertRaises(ValueError) as caught: self.orchestrator.execute(self.request())
        self.assertIs(caught.exception,a.fail_profile)
        self.assertIs(caught.exception.__cause__,a.fail_cleanup)

    def test_cleanup_only_failure(self):
        a=self.add(); a.fail_cleanup=RuntimeError('cleanup')
        with self.assertRaisesRegex(RuntimeError,'cleanup'): self.orchestrator.execute(self.request())
        self.assertEqual(a.status,'ERROR')

    def test_interruption_always_cleans_up(self):
        for failure in (KeyboardInterrupt(), SystemExit(2)):
            with self.subTest(failure=type(failure).__name__):
                self.registry.clear()
                a = self.add()
                a.fail_profile = failure
                with self.assertRaises(type(failure)):
                    self.orchestrator.execute(self.request())
                self.assertFalse(a.model_loaded)
                self.assertEqual(a.calls[-1], ('unload', True))
                self.assertEqual(a.status, 'ERROR')

    def test_agent_interruption_sets_error_for_cleanup(self):
        a = Agent()
        a.status = AgentStatus.READY
        a._runtime_ready = True
        a.model_loaded = True
        with patch.object(a.predictor, 'predict', side_effect=KeyboardInterrupt()):
            with self.assertRaises(KeyboardInterrupt):
                a.predict(None)
        self.assertEqual(a.get_status(), 'ERROR')
        a.unload_model(preserve_error=True)
        self.assertFalse(a.model_loaded)
        self.assertEqual(a.get_status(), 'ERROR')

    def test_agent_preserve_error_and_capabilities(self):
        a=Agent()
        a._runtime_ready=True; a.status=AgentStatus.ERROR
        a.model_loaded=True; a.model_path=Path('model'); a.manifest=object(); a.artifact_metadata={'x':1}
        a.unload_model(preserve_error=True)
        self.assertEqual(a.get_status(),'ERROR')
        self.assertFalse(a.model_loaded); self.assertIsNone(a.manifest); self.assertIsNone(a.model_path)
        self.assertEqual(a.artifact_metadata,{})
        a.unload_model(); self.assertEqual(a.get_status(),'READY')
        self.assertEqual(a.get_info()['capabilities']['model_formats'],['onnx'])
        self.assertEqual(a.get_info()['capabilities']['supported_actions'],['profile'])


if __name__=='__main__': unittest.main()
