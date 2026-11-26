import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from tests.test_algorithms import test_dflash_review_fixes as fixtures
from specforge.algorithms.dflash.providers import resume_contract
from specforge.data.loss_mask import trim_loss_mask_after_eos, has_consecutive_supervised_tokens
from specforge.data.vlm_preprocessing import _prepare_one_record
from specforge.export.checkpoint_io import materialize_draft
from specforge.modeling.draft.dflash_config import COMPUTE_CONFIG_KEY
from specforge.training.provenance import model_source_identity


class ConfigAndEligibilityTest(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.ReviewFixesTest()
        fixture.setUp()
        self.draft, self.model = fixture.draft, fixture.model
        self.draft.config.architectures = ['DFlashDraftModel']

    def state(self):
        return {'draft_state_dict': self.draft.state_dict(),
                **resume_contract(None, self.draft, self.model)}

    def test_export_rejects_attention_and_norm_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'config.json'
            for changes in [dict(is_causal=True), dict(rms_norm_eps=0.1),
                            dict(layer_types=['sliding_attention'], sliding_window=4, use_sliding_window=True),
                            dict(eos_token_id=8)]:
                payload = self.draft.config.to_dict()
                payload.update(changes)
                path.write_text(json.dumps(payload))
                with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'computation mismatch'):
                    materialize_draft(self.state(), str(path))

    def test_export_accepts_matching_config_and_runtime_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            payload = self.draft.config.to_dict()
            payload.update(dtype='bfloat16', transformers_version='different', _name_or_path='moved')
            path = Path(temp)/'config.json'; path.write_text(json.dumps(payload))
            self.assertTrue(materialize_draft(self.state(), str(path)).use_interleaved_mrope)

    def test_legacy_export_uses_original_config_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)/'original.json'
            payload = self.draft.config.to_dict(); source.write_text(json.dumps(payload))
            state = self.state(); state.pop(COMPUTE_CONFIG_KEY)
            state['specforge_model_provenance'] = (('draft_config', model_source_identity(str(source))),)
            self.assertTrue(materialize_draft(state, str(source)).use_interleaved_mrope)
            changed = Path(temp)/'changed.json'
            payload['is_causal'] = True; changed.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, 'computation mismatch'):
                materialize_draft(state, str(changed))
            source.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, 'differs from checkpoint provenance'):
                materialize_draft(state, str(changed))

    def test_missing_legacy_provenance_is_explicit_error(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'config.json';path.write_text(json.dumps(self.draft.config.to_dict()))
            state=self.state();state.pop(COMPUTE_CONFIG_KEY)
            with self.assertRaisesRegex(ValueError, 'provenance'):
                materialize_draft(state,str(path))

    def test_shared_mask_handles_batches_multiple_eos_and_idempotence(self):
        ids = torch.tensor([[2,7,8,0,3,9,4], [0,0,0,0,0,0,0]])
        mask = torch.tensor([[1,1,1,0,1,1,1], [0,0,0,0,0,0,0]])
        expected = torch.tensor([[1,1,0,0,1,1,0], [0,0,0,0,0,0,0]])
        result=trim_loss_mask_after_eos(ids,mask,[7,9])
        torch.testing.assert_close(result,expected)
        torch.testing.assert_close(trim_loss_mask_after_eos(ids,result,[7,9]),result)
        self.assertEqual(trim_loss_mask_after_eos(ids[0].tolist(),mask[0].tolist(),[7,9]),expected[0].tolist())
        self.assertEqual(trim_loss_mask_after_eos([],[],7),[])

    def test_preprocessor_drops_empty_response_but_keeps_one_token_response(self):
        for ids,mask,accepted in [([1,7,8],[0,1,1],False),
                                  ([1,2,7,8],[0,1,1,1],True),
                                  ([1,7,8,0,2,7,8],[0,1,1,0,1,1,1],True)]:
            parsed={'input_ids':[torch.tensor([ids])], 'loss_mask':[torch.tensor([mask])]}
            with patch('specforge.data.vlm_preprocessing._render_conversation_text',return_value='text'), \
                 patch('specforge.data.preprocessing.preprocess_conversations',return_value=parsed):
                output=_prepare_one_record({'conversations':[{'role':'assistant','content':''}]},
                    source='test',tokenizer=None,processor=None,template=None,pad_token_id=99,
                    merge_size=2,max_length=128,min_loss_tokens=2,generation_eos_token_id=7)
            self.assertEqual(output is not None,accepted)
            if output:
                self.assertTrue(has_consecutive_supervised_tokens(output['payload']['loss_mask']))
                self.assertEqual(output['payload']['loss_mask'][-1],0)

    def test_strategy_counts_trimmed_mask_before_forward(self):
        from specforge.training.strategies.base import DFlashTrainStrategy
        from specforge.runtime.contracts import TrainBatch
        ids=torch.tensor([[1,7,8,0,2,3,7,8]])
        mask=torch.tensor([[0,1,1,0,1,1,1,1]])
        tensors={'input_ids':ids,'loss_mask':mask,'hidden_states':torch.randn(1,8,16),
                 'position_ids':torch.arange(8)[None,:,None].expand(1,8,3)}
        strategy=DFlashTrainStrategy(self.model)
        with patch.object(self.model,'forward',wraps=self.model.forward) as forward:
            strategy.forward_loss(TrainBatch(tensors=tensors,sample_ids=['test'],strategy='dflash'))
        self.assertEqual(forward.call_args.kwargs['max_valid_anchors'],2)
        self.assertEqual(forward.call_args.kwargs['loss_mask'].tolist(),[[0,1,0,0,1,1,1,0]])

    def test_legacy_empty_rank_retains_finite_zero_gradient_graph(self):
        from specforge.training.strategies.base import DFlashTrainStrategy
        from specforge.runtime.contracts import TrainBatch

        tensors={'input_ids':torch.tensor([[1,7,8]]),
                 'loss_mask':torch.tensor([[0,1,1]]),
                 'hidden_states':torch.randn(1,3,16),
                 'position_ids':torch.arange(3)[None,:,None].expand(1,3,3)}
        output=DFlashTrainStrategy(self.model).forward_loss(
            TrainBatch(tensors=tensors,sample_ids=['empty'],strategy='dflash'))
        numerator, denominator=output.loss_terms
        self.assertEqual(denominator.item(),0)
        numerator.backward()
        for parameter in self.draft.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertEqual(parameter.grad.abs().sum().item(),0)

    def test_vlm_adapter_preserves_zero_eos_and_resolves_missing_eos(self):
        from specforge.algorithms.common.vlm_input import VlmServerInputAdapter

        config=SimpleNamespace(model=SimpleNamespace(target_model_path='target',cache_dir=None,trust_remote_code=False),
            training=SimpleNamespace(strategy='dflash'),data=SimpleNamespace(prompts_path='data.jsonl',
            train_data_path=None,chat_template='qwen',max_length=128,max_prompts=None))
        for configured,expected in [(0,0),(None,7)]:
            with patch('transformers.AutoProcessor.from_pretrained'), \
                 patch('transformers.AutoConfig.from_pretrained',return_value=SimpleNamespace(model_type='qwen3_vl')), \
                 patch('specforge.data.vlm_preprocessing.build_vlm_prompt_payloads',return_value=[]) as prepare:
                VlmServerInputAdapter(config,minimum_loss_tokens=lambda *args:2).prepare_prompts(
                    config,SimpleNamespace(eos_token_id=7),draft_config=SimpleNamespace(eos_token_id=configured))
            self.assertEqual(prepare.call_args.kwargs['generation_eos_token_id'],expected)
