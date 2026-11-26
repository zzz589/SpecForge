"""Real DFlash2 model under the repaired common VL training contract."""
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from specforge.modeling.auto import AutoDraftModel
from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
from specforge.algorithms.dflash.providers import resume_contract
from specforge.runtime.contracts import TrainBatch
from specforge.training.strategies.base import DFlashTrainStrategy, StepContext
from tests.test_algorithms.test_dflash_vl_mrope import config


def build_model(dtype=torch.float32, loss_type='dflash'):
    torch.manual_seed(47)
    c=config();c.architectures=['DFlash2DraftModel']
    c.dflash_config.update(block_size=4,conv_group_size=4,conv_kernel_size=2,
        selector_rank=4,selector_top_k=32,target_layer_ids=[0,1])
    draft=AutoDraftModel.from_config(c).to(dtype)
    return OnlineDFlashModel(draft_model=draft,
        target_lm_head=nn.Linear(16,32,bias=False).to(dtype).requires_grad_(False),
        target_embed_tokens=nn.Embedding(32,16).to(dtype).requires_grad_(False),
        mask_token_id=31,block_size=4,attention_backend='sdpa',num_anchors=6,
        loss_type=loss_type,selector_loss_alpha=1.,teacher_metrics=False)


class DFlash2UnifiedVlmTest(unittest.TestCase):
    def test_packed_convolution_blocks_match_independent_cached_prefixes(self):
        from tests.test_algorithms.test_dspark_mrope_block_parity import TestDSparkMropeBlockParity
        def nontrivial_model(dtype):
            model=build_model(dtype=dtype)
            with torch.no_grad():
                for layer in model.draft_model.layers:
                    for conv in (layer.attention_conv,layer.mlp_conv):
                        conv.kernel_projection.weight.normal_(std=.08)
                        conv.base_kernel.normal_(std=.08)
            return model
        with patch('tests.test_algorithms.test_dspark_mrope_block_parity.build_model',side_effect=nontrivial_model):
            case=TestDSparkMropeBlockParity('test_packed_blocks_match_independent_and_cached_prefixes')
            case.test_packed_blocks_match_independent_and_cached_prefixes()

    def test_mrope_objectives_train_convolution_selector_and_reload(self):
        for loss_type in ('dflash','dpace'):
            with self.subTest(loss_type=loss_type):
                model=build_model(loss_type=loss_type);draft=model.draft_model
                ids=torch.randint(0,30,(1,12))
                pos=torch.tensor([[[0,0,0],[1,1,1],[2,2,2],[2,2,3],[2,3,2],[2,3,3],
                    [4,4,4],[5,5,5],[6,6,6],[7,7,7],[8,8,8],[9,9,9]]])
                batch=TrainBatch(sample_ids=['image'],strategy='dflash',tensors=dict(
                    input_ids=ids,hidden_states=torch.randn(1,12,32),
                    loss_mask=torch.tensor([[0]*7+[1]*5]),position_ids=pos))
                captured=[]
                hook=draft.rotary_emb.register_forward_pre_hook(lambda m,a:captured.append(a[1].detach().clone()))
                result=DFlashTrainStrategy(model).forward_loss(batch,StepContext(global_step=1,total_steps=4,collect_detailed_metrics=False))
                self.assertTrue(torch.isfinite(result.loss));result.loss.backward();hook.remove()
                torch.testing.assert_close(captured[0][:,:,:12],pos.permute(2,0,1))
                for name in ('fc.weight','layers.0.attention_conv.kernel_projection.weight',
                    'layers.0.mlp_conv.kernel_projection.weight','candidate_selector.successor_codebook'):
                    grad=dict(draft.named_parameters())[name].grad
                    self.assertIsNotNone(grad,name);self.assertTrue(torch.isfinite(grad).all(),name)
                    self.assertGreater(grad.abs().sum().item(),0,name)
                contract=resume_contract(None,draft,model)
                self.assertIn('dflash_compute_config',contract)
                self.assertEqual(contract['dflash_position_convention'],'qwen3_vl_triplets_anchor_continuation_v1')
                with tempfile.TemporaryDirectory() as path:
                    draft.save_pretrained(path);restored=AutoDraftModel.from_pretrained(path)
                    self.assertTrue(restored.use_interleaved_mrope)
                    for name,value in draft.state_dict().items():
                        torch.testing.assert_close(value,restored.state_dict()[name],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
