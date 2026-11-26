import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from tests.test_algorithms.test_dflash_vl_mrope import config
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
from specforge.algorithms.dflash.providers import resume_contract
from specforge.export.checkpoint_io import materialize_draft, apply_legacy_rope_scaling


class ReviewFixesTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(541)
        c = config(); c.eos_token_id = 7
        self.draft = DFlashDraftModel(c).eval()
        self.model = OnlineDFlashModel(self.draft, nn.Linear(16, 32, bias=False),
            nn.Embedding(32, 16), 31, block_size=4, num_anchors=2,
            attention_backend="sdpa", teacher_metrics=False, objective_chunk_blocks=0).eval()

    def test_cross_turn_label_cannot_change_loss_or_gradient(self):
        m = self.model
        m._sample_anchor_positions = lambda *a, **kw: (torch.tensor([[1]]), torch.tensor([[True]]))
        ids = torch.tensor([[1,2,3,4,5,6,8,9]])
        hidden = torch.randn(1,8,16)
        pos = torch.arange(8)[None,:,None].expand(1,8,3)
        mask = torch.tensor([[0.,1.,1.,0.,1.,1.,0.,0.]])
        def run(ids):
            m.zero_grad()
            loss = m(ids,hidden,mask,position_ids=pos,collect_detailed_metrics=False)[0]
            loss.backward()
            return loss.detach(), [p.grad.clone() for p in m.parameters() if p.grad is not None]
        a, ga = run(ids)
        ids[0,4] = 17
        b, gb = run(ids)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        for x,y in zip(ga,gb): torch.testing.assert_close(x,y,rtol=0,atol=0)

    def test_eos_remains_label_but_never_anchor_or_post_eos_target(self):
        ids = torch.tensor([[1,2,7,8,9,4,5,7,6]])
        mask = torch.tensor([[0.,1.,1.,1.,0.,1.,1.,1.,1.]])
        trimmed = self.model._generation_loss_mask(ids,mask)
        self.assertEqual(trimmed.tolist(), [[0,1,1,0,0,1,1,1,0]])
        anchors, keep = self.model._sample_anchor_positions(9,trimmed,ids.device)
        self.assertTrue(set(anchors[keep].tolist()) <= {1,5,6})
        self.draft.config.eos_token_id = [7,5]
        self.assertEqual(self.model._generation_loss_mask(ids,mask).tolist(), [[0,1,1,0,0,1,1,0,0]])

    def test_export_requires_matching_position_provenance(self):
        c = self.draft.config; c.architectures = ['DFlashDraftModel']
        state = {'draft_state_dict':self.draft.state_dict(), **resume_contract(None,self.draft,self.model)}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'config.json'
            def export(payload, checkpoint=state):
                path.write_text(json.dumps(payload))
                return materialize_draft(checkpoint,str(path))
            self.assertTrue(export(c.to_dict()).use_interleaved_mrope)
            for key,value in [('dflash_mrope_theta',1),('dflash_mrope_section',(1,1,2)),
                              ('dflash_block_size',8),('dflash_target_layer_ids',(1,))]:
                with self.subTest(key=key), self.assertRaises(ValueError):
                    export(c.to_dict(),dict(state,**{key:value}))
            with self.assertRaises(ValueError): export(c.to_dict(),{'draft_state_dict':self.draft.state_dict()})
            plain=copy.deepcopy(c.to_dict()); plain.pop('rope_scaling',None)
            plain['rope_parameters']={'rope_type':'default','rope_theta':5000000}
            with self.assertRaises(ValueError): export(plain)
            path.write_text(json.dumps(c.to_dict()))
            apply_legacy_rope_scaling(temp)
            out=json.loads(path.read_text())
            self.assertEqual(out['rope_scaling']['mrope_section'],[2,1,1])
            self.assertTrue(out['rope_scaling']['mrope_interleaved'])

    def test_trainer_ddp_accumulation_communicates_once(self):
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        from specforge.training.backend import FSDPTrainingBackend
        from specforge.training.controller import TrainerCore
        from tests.test_runtime.test_trainer import FakeBackend, StepOutput, _batch
        with tempfile.TemporaryDirectory() as temp:
            dist.init_process_group('gloo',init_method='file://'+str(Path(temp)/'init'),rank=0,world_size=1)
            try:
                model=DDP(nn.Linear(2,1,bias=False)); count=[0]
                def hook(state,bucket):
                    state[0]+=1
                    future=torch.futures.Future(); future.set_result(bucket.buffer()); return future
                model.register_comm_hook(count,hook)
                backend=FakeBackend(model)
                wrapped=object.__new__(FSDPTrainingBackend)
                wrapped.module=model; wrapped._wrapped=True
                backend.accumulation_context=wrapped.accumulation_context
                strategy=SimpleNamespace(forward_loss=lambda *args: StepOutput(loss=model(torch.ones(1,2)).sum(), metrics={}))
                core=TrainerCore(strategy,backend,accumulation_steps=2)
                core.train_step(_batch()); self.assertEqual(count[0],0)
                core.train_step(_batch()); self.assertEqual(count[0],1)
                torch.testing.assert_close(model.module.weight.grad,torch.ones(1,2))
            finally: dist.destroy_process_group()
