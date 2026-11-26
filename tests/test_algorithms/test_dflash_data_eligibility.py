import gzip
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.data.prompt_builder import prepare_prompt_tasks
from specforge.data.loss_mask import has_consecutive_supervised_tokens


class DataEligibilityTest(unittest.TestCase):
    def setUp(self):
        self.algorithm=builtin_algorithm_registry().resolve('dflash')
        self.model=SimpleNamespace(draft_model=SimpleNamespace(config=SimpleNamespace(eos_token_id=7)))
        self.rows=[{'input_ids':[1,7,8], 'loss_mask':[0,1,1]},
                   {'input_ids':[1,2,7,8], 'loss_mask':[0,1,1,1]},
                   {'input_ids':[1,3,7,8], 'loss_mask':[0,1,1,1]}]

    def prepare(self,path,**options):
        return prepare_prompt_tasks(str(path),None,chat_template=None,max_length=4,
            is_preformatted=False,train_only_last_turn=False,cache_dir=None,cache_key=None,
            num_proc=1,min_loss_tokens=2,loss_mask_filter=has_consecutive_supervised_tokens,
            generation_eos_token_id=7,**options)

    def test_pretokenized_filters_before_limit_and_preserves_eos(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'data.jsonl';path.write_text('\n'.join(json.dumps(row) for row in self.rows))
            prompts=self.prepare(path,max_prompts=1)
        self.assertEqual(len(prompts),1)
        self.assertEqual(prompts[0]['payload'],{'input_ids':[1,2,7,8],'loss_mask':[0,1,1,0]})

    def test_training_assembly_passes_resolved_eos_to_text_preparation(self):
        from specforge.training.assembly import _prepare_prompts

        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'data.jsonl';path.write_text('\n'.join(json.dumps(row) for row in self.rows))
            config=SimpleNamespace(model=SimpleNamespace(input_modality='text'),
                data=SimpleNamespace(prompts_path=str(path),train_data_path=None,cache_key='test',
                    cache_dir=str(Path(temp)/'cache'),chat_template=None,max_length=4,
                    is_preformatted=False,train_only_last_turn=False,build_dataset_num_proc=1,max_prompts=1))
            prompts=_prepare_prompts(config,None,algorithm=self.algorithm,
                draft_config=SimpleNamespace(block_size=4,eos_token_id=7))
            self.assertEqual(len(prompts),1)
            self.assertEqual(prompts[0]['payload']['input_ids'],[1,2,7,8])
            self.assertEqual(prompts[0]['payload']['loss_mask'],[0,1,1,0])

    def test_raw_cached_rows_filter_before_cap_and_stay_lazy(self):
        from datasets import Dataset
        raw=Dataset.from_list([{'conversations':[]}]*3)
        processed=Dataset.from_list(self.rows)
        processed.set_format(type='torch')
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'raw.jsonl';path.write_text(json.dumps({'conversations':[]}))
            with patch('datasets.load_dataset',return_value=raw), \
                 patch('specforge.data.preprocessing.build_eagle3_dataset',return_value=processed) as build:
                prompts=self.prepare(path,max_prompts=1)
                self.assertEqual(len(build.call_args.kwargs['dataset']),3)
                self.assertEqual(len(prompts),1)
                self.assertEqual(prompts[0]['payload']['loss_mask'],[0,1,1,0])
                self.assertEqual(prompts[-1],prompts[0])
                self.assertEqual(len(processed),3)
                self.assertEqual(processed[1]['loss_mask'].tolist(),[0,1,1,1])

    def write_features(self,temp):
        paths=[]
        for index,row in enumerate(self.rows):
            path=Path(temp)/f'{index}.ckpt'
            data={key:torch.tensor(value) for key,value in row.items()}
            data['hidden_states']=torch.zeros(len(row['input_ids']),16)
            if index==2:
                path=Path(str(path)+'.gz')
                with gzip.open(path,'wb') as stream:torch.save(data,stream)
            else:torch.save(data,path)
            paths.append(path)
        return paths

    def test_offline_launcher_filters_before_counts_shards_and_resume_seek(self):
        from specforge.launch import build_offline_runtime
        with tempfile.TemporaryDirectory() as temp:
            self.write_features(temp)
            with patch('specforge.launch._assemble_trainer',side_effect=lambda **kw:kw):
                built=build_offline_runtime(algorithm=self.algorithm,hidden_states_path=temp,
                    draft_model=self.model,target_head=None,optimizer_factory=None,run_id='test',
                    output_dir=str(Path(temp)/'output'),max_len=4,seed=19,resume_from='checkpoint')
            self.assertEqual(built['checkpoint_extra']['source_dataset_size'],2)
            self.assertEqual(built['resume_from'],'checkpoint')
            for epoch in [0,1]:
                refs=built['ref_source']['refs_for_epoch'](epoch)
                self.assertEqual({ref.metadata['file_index'] for ref in refs},{1,2})
                self.assertEqual([ref.sample_id for ref in refs],
                    [ref.sample_id for ref in built['ref_source']['refs_for_epoch'](epoch)])
            ref=built['ref_source']['refs'][0]
            tensors,handle=built['store'].get(ref)
            try:normalized=built['per_sample_transform'](tensors)
            finally:built['store'].release(handle)
            self.assertEqual(normalized['loss_mask'].tolist(),[[0,1,1,0]])

    def test_disaggregated_offline_plan_retains_files(self):
        from specforge.launch import build_disagg_offline_runtime
        from specforge.runtime.data_plane.disaggregated import SharedDirFeatureStore
        from specforge.runtime.data_plane.disagg_ingest import ingest_offline_features
        with tempfile.TemporaryDirectory() as temp:
            feature_dir=Path(temp)/'features';feature_dir.mkdir();self.write_features(feature_dir)
            store=SharedDirFeatureStore(str(Path(temp)/'shared'),store_id='test',retain_on_release=True)
            refs=ingest_offline_features(store,str(feature_dir),algorithm_name='dflash',
                build_reader=self.algorithm.providers.offline_for('text').build_reader,run_id='test',max_len=4)
            with patch('specforge.launch._assemble_trainer',side_effect=lambda **kw:kw):
                built=build_disagg_offline_runtime(algorithm=self.algorithm,feature_store=store,refs=refs,
                    draft_model=self.model,target_head=None,optimizer_factory=None,run_id='test',
                    output_dir=str(Path(temp)/'out'),max_len=4)
            self.assertEqual(built['checkpoint_extra']['source_dataset_size'],2)
            for ref in refs:
                tensors,handle=store.get(ref)
                store.release(handle)
                self.assertIn('input_ids',tensors)

    def test_all_invalid_offline_fails_before_trainer_build(self):
        from specforge.launch import build_offline_runtime
        with tempfile.TemporaryDirectory() as temp:
            self.rows=self.rows[:1];self.write_features(temp)
            with patch('specforge.launch._assemble_trainer') as build, \
                 self.assertRaisesRegex(ValueError,'no eligible generation anchors'):
                build_offline_runtime(algorithm=self.algorithm,hidden_states_path=temp,
                    draft_model=self.model,target_head=None,optimizer_factory=None,run_id='test',
                    output_dir=str(Path(temp)/'out'),max_len=4)
            build.assert_not_called()

    def test_eval_uses_the_same_filtered_plan(self):
        from specforge.launch import _make_offline_eval_data_factory
        with tempfile.TemporaryDirectory() as temp:
            self.write_features(temp)
            factory=_make_offline_eval_data_factory(algorithm=self.algorithm,modality='text',
                hidden_states_path=temp,run_id='test',batch_size=2,max_len=4,ttt_length=1,
                use_usp_preprocess=False,dataloader_num_workers=0,draft_model=self.model)
            loader=factory()
            self.assertEqual(len(loader._refs),2)
            batch=next(iter(loader))
            self.assertEqual(batch.tensors['loss_mask'].tolist(),[[0,1,1,0],[0,1,1,0]])
