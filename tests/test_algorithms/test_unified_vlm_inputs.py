"""Regressions for merging exact-token and Arrow-cache input into PR785."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from specforge.data.vlm_preprocessing import (
    _MappedVlmPromptSequence, _token_mask_digest, _collapse_mapped_image_region,
    build_mapped_vlm_prompt_payloads, _exact_qwen_turn,
)
from tests.test_algorithms.test_regenerated_vl_alignment import Tokenizer


class UnifiedVlmInputTest(unittest.TestCase):
    def sequence(self, rows, images=None, **kw):
        return _MappedVlmPromptSequence(
            rows, images or [None]*len(rows), pad_token_id=60,
            max_length=100, min_loss_tokens=2, merge_size=2,
            generation_eos_token_id=99, **kw)

    def test_filters_after_eos_before_publishing_count_and_limit(self):
        rows=[dict(input_ids=[[7,99,4]],loss_mask=[[0,1,1]]),
              dict(input_ids=[[7,10,99,4]],loss_mask=[[0,1,1,1]]),
              dict(input_ids=[[7,11,99,4]],loss_mask=[[0,1,1,1]])]
        seq=self.sequence(rows,limit=1)
        self.assertEqual(len(seq),1)
        self.assertEqual(seq[0]['payload']['input_ids'],[7,10,99,4])
        self.assertEqual(seq[-1]['payload']['loss_mask'],[0,1,1,0])
        self.assertEqual(len(seq[:]),1)
        with self.assertRaises(IndexError):seq[1]

    def test_cached_tokens_and_source_masks_are_bound(self):
        rows=[dict(input_ids=[[7,10,99,4]],loss_mask=[[0,1,1,1]])]
        correct=_token_mask_digest([7,10,99,4],[0,1,1,0])
        self.assertEqual(len(self.sequence(rows,expected_digests=[correct])),1)
        for ids,mask in [([8,10,99,4],[0,1,1,0]),([7,10,99,4],[1,1,1,0])]:
            with self.assertRaisesRegex(ValueError,'disagree with source'):
                self.sequence(rows,expected_digests=[_token_mask_digest(ids,mask)])

    def test_rejects_invalid_cached_mask_and_image_supervision(self):
        for ids,mask in [([7,10],[1]),([7,10],[0,2]),([7,1.5],[0,1]),([60,10,99],[1,1,1])]:
            with self.subTest(ids=ids,mask=mask),self.assertRaises(ValueError):
                self.sequence([dict(input_ids=ids,loss_mask=mask)])

    def test_same_token_count_but_wrong_grid_is_rejected(self):
        rows=[dict(input_ids=[[7,60,60,60,60,10,99,4]],
                   loss_mask=[[0,0,0,0,0,1,1,1]],image_grid_thw=[[1,4,4]])]
        with tempfile.TemporaryDirectory() as tmp:
            image=Path(tmp)/'one.png';Image.new('RGB',(32,32)).save(image)
            processor=SimpleNamespace(image_processor=lambda **kw:dict(image_grid_thw=torch.tensor([[1,2,8]])))
            seq=self.sequence(rows,[str(image)],processor=processor)
            with self.assertRaisesRegex(ValueError,'differs from target processor'):seq[0]
            processor.image_processor=lambda **kw:dict(image_grid_thw=torch.tensor([[1,4,4]]))
            payload=seq[0]['payload']
            self.assertTrue(payload['image_data'].startswith('data:image/'))
            self.assertEqual(payload['request_input_ids'],[7,60,10,99,4])
            self.assertEqual(payload['loss_mask'][-3:],[1,1,0])
            self.assertEqual(payload['position_ids'][5],[3,3,3])

    def test_text_only_rows_cannot_smuggle_image_grid(self):
        seq=self.sequence([dict(input_ids=[7,10,99],loss_mask=[0,1,1],image_grid_thw=[1,2,2])])
        with self.assertRaisesRegex(ValueError,'grid/token-run mismatch'):seq[0]

    def test_sharded_cache_reconstructs_order_and_detects_wrong_source(self):
        import numpy as np
        from datasets import Dataset
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            originals=[dict(input_ids=[7,i,99,4],loss_mask=[0,1,1,1]) for i in (10,11,12,13,14)]
            order=[]; shards=[];offset=0
            for number,count in enumerate((3,2)):
                indices=(offset+np.random.default_rng(42).permutation(count)).tolist()
                order.extend(indices)
                part=tmp/f'shard{number}'
                Dataset.from_list([originals[i] for i in indices]).save_to_disk(str(part))
                shards.append(dict(rows=count,files=[str(p) for p in part.glob('*.arrow')]))
                offset+=count
            manifest=dict(complete=True,storage_format='sharded_manifest',chat_template='qwen',
                max_length=100,target_model_path='/target',dataset_size=5,shuffle_seed=42,shards=shards)
            mp=tmp/'manifest.json';mp.write_text(json.dumps(manifest))
            raw=tmp/'raw.jsonl';raw.write_text('\n'.join(json.dumps(r) for r in originals))
            def prepare(record,**kw):return record['input_ids'],record['loss_mask']
            with patch('specforge.data.vlm_preprocessing._prepare_collapsed_record',side_effect=prepare):
                seq=build_mapped_vlm_prompt_payloads(str(raw),str(mp),Tokenizer(),target_model_path='/target',
                    chat_template='qwen',max_length=100,min_loss_tokens=2,
                    processor=SimpleNamespace(image_processor=SimpleNamespace(merge_size=2)))
                self.assertEqual([seq[i]['payload']['input_ids'][1] for i in range(5)],
                                 [originals[i]['input_ids'][1] for i in order])
                originals[0]['input_ids'][1]=15
                raw.write_text('\n'.join(json.dumps(r) for r in originals))
                with self.assertRaisesRegex(ValueError,'disagree with source'):
                    build_mapped_vlm_prompt_payloads(str(raw),str(mp),Tokenizer(),target_model_path='/target',
                        chat_template='qwen',max_length=100,
                        processor=SimpleNamespace(image_processor=SimpleNamespace(merge_size=2)))

    def test_exact_single_turn_rejects_extra_rounds(self):
        row=dict(generation_prompt_ids=[7],response_token_ids=[10,11],conversations=[
            dict(role='user',content='q'),dict(role='assistant',content='ab'),
            dict(role='user',content='q2'),dict(role='assistant',content='ab')])
        with self.assertRaisesRegex(ValueError,'one user and one assistant'):
            _exact_qwen_turn(row,Tokenizer(),False,'case')


if __name__=='__main__':unittest.main()
