import ast
import importlib.util
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from PIL import Image
from transformers import AutoTokenizer,AutoProcessor,AutoConfig
from specforge.algorithms.common.vlm_input import VlmServerInputAdapter
from specforge.data.vlm_preprocessing import _render_conversation_text

torch.set_num_threads(1)
target_path='/HOME/nsccgz_zgchen/nsccgz_zgchen_8/HDD_POOL/xj/models/Qwen3-VL-4B-Instruct'
tokenizer=AutoTokenizer.from_pretrained(target_path,local_files_only=True)
processor=AutoProcessor.from_pretrained(target_path,local_files_only=True)
target_config=AutoConfig.from_pretrained(target_path,local_files_only=True)
sgroot=Path(importlib.util.find_spec('sglang').origin).parent

def source_methods(path,names):
    tree=ast.parse(path.read_text())
    methods=[]
    for cls in tree.body:
        if isinstance(cls,ast.ClassDef):
            methods.extend(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names)
    assert {n.name for n in methods} == set(names)
    cls=ast.ClassDef(name='ActualMethods',bases=[],keywords=[],body=methods,decorator_list=[])
    namespace={'torch':torch}
    code='from __future__ import annotations\n'+ast.unparse(ast.fix_missing_locations(cls))
    exec(compile(code,str(path),'exec'),namespace)
    return namespace['ActualMethods']

Base=source_methods(sgroot/'srt/multimodal/processors/base_processor.py',
    {'resolve_image_token_counts','_expand_input_ids'})
Positions=source_methods(sgroot/'srt/multimodal/processors/qwen_vl.py',
    {'_as_grid_batch','_compute_image_only_mrope_positions_from_offsets'})
server=Base();server._processor=processor
position_server=Positions();position_server.model_type='qwen3_vl'
position_server._spatial_merge_size=target_config.vision_config.spatial_merge_size
import hashlib
import subprocess
results={'image_cases':[], 'sglang_processor_root':str(sgroot.resolve()),
    'sglang_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=sgroot,text=True).strip(),
    'sglang_processor_hashes':{str(path.relative_to(sgroot)):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [sgroot/'srt/multimodal/processors/base_processor.py',sgroot/'srt/multimodal/processors/qwen_vl.py']}}
with tempfile.TemporaryDirectory() as temp:
    cfg=SimpleNamespace(model=SimpleNamespace(target_model_path=target_path,cache_dir=None,trust_remote_code=False),
        training=SimpleNamespace(strategy='dspark'),data=SimpleNamespace(prompts_path=str(Path(temp)/'data.jsonl'),
        train_data_path=None,chat_template='qwen',max_length=8192,max_prompts=None,train_only_last_turn=False))
    cases=[((32,32),'RGB','PNG','Yes.'),((63,95),'RGB','PNG','这是一只猫。'),
           ((601,379),'RGB','JPEG','A red object.'),((224,224),'L','PNG','The number is 17.'),
           ((1024,768),'RGB','JPEG','A small boat.'),((128,64),'RGBA','PNG','Blue.')]
    for index,(size,mode,fmt,answer) in enumerate(cases):
        path=Path(temp)/f'image-{index}.{fmt.lower()}'
        color=70 if mode=='L' else ((20,50,70,100) if mode=='RGBA' else (20,50,70))
        Image.new(mode,size,color).save(path,format=fmt)
        conv=[{'role':'user','content':'Describe this image.'},{'role':'assistant','content':answer}]
        Path(cfg.data.prompts_path).write_text(json.dumps({'image':str(path),'conversations':conv},ensure_ascii=False)+'\n')
        payload=VlmServerInputAdapter(cfg,minimum_loss_tokens=lambda *a:2).prepare_prompts(
            cfg,tokenizer,draft_config=SimpleNamespace(eos_token_id=tokenizer.eos_token_id))[0]['payload']
        pil=Image.open(path).convert('RGB')
        counts=server.resolve_image_token_counts([pil])
        server_ids=server._expand_input_ids(payload['request_input_ids'],counts,target_config.image_token_id)
        assert server_ids==payload['input_ids']
        rendered=_render_conversation_text(tokenizer,conv,with_image=True,source='audit')
        hf_inputs=processor(text=[rendered],images=[pil],return_tensors='pt')
        assert hf_inputs['input_ids'][0].tolist()==server_ids
        slots=[i for i,t in enumerate(server_ids) if t==target_config.image_token_id]
        item=SimpleNamespace(is_image=lambda:True,offsets=[(slots[0],slots[-1])],
            model_specific_data={'image_grid_thw':hf_inputs['image_grid_thw']})
        sp,_=position_server._compute_image_only_mrope_positions_from_offsets(len(server_ids),[item],torch.long,torch.device('cpu'))
        cp=torch.tensor(payload['position_ids']).T[:,None]
        assert torch.equal(sp,cp)
        supervised=[t for t,m in zip(server_ids,payload['loss_mask']) if m]
        expected=tokenizer.encode(answer,add_special_tokens=False)+[tokenizer.eos_token_id]
        assert supervised==expected,(answer,tokenizer.decode(supervised),tokenizer.decode(expected))
        results['image_cases'].append({'size':size,'mode':mode,'format':fmt,'image_tokens':counts[0],
            'grid':hf_inputs['image_grid_thw'].tolist(),'ids_equal':True,'positions_equal':True,'labels_equal':True})
        if index==0:
            tiny_inputs={k:v.clone() for k,v in hf_inputs.items()}
            tiny_payload=payload

    # An empty answer has only EOS and cannot supply a DSpark anchor/next-token pair.
    Path(cfg.data.prompts_path).write_text(json.dumps({'image': str(path),
        'conversations': [{'role': 'user', 'content': 'Describe this image.'},
                          {'role': 'assistant', 'content': ''}]}) + '\n')
    empty = VlmServerInputAdapter(cfg, minimum_loss_tokens=lambda *a: 2).prepare_prompts(
        cfg, tokenizer, draft_config=SimpleNamespace(eos_token_id=tokenizer.eos_token_id))
    assert empty == []
    results['empty_answer_filtered'] = True

    # Real Transformers Qwen3-VL architecture with tiny random weights, not the 4B target.
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
    from tests.test_algorithms.test_dspark_vl_mrope import build_model
    def draft_config(): return build_model().draft_model.config
    from specforge.modeling.draft.dspark import DSparkDraftModel
    from specforge.algorithms.common.dflash_family_model import OnlineDSparkModel
    from specforge.algorithms.dspark.providers import resume_contract
    from specforge.training.strategies.base import DSparkTrainStrategy,StepContext
    from specforge.runtime.contracts import TrainBatch
    from specforge.export.checkpoint_io import materialize_draft

    tc=Qwen3VLConfig(text_config=dict(vocab_size=151936,hidden_size=16,intermediate_size=32,
        num_hidden_layers=4,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
        rope_theta=5000000,rope_scaling={'rope_type':'default','mrope_section':[2,1,1],'mrope_interleaved':True}),
        vision_config=dict(depth=3,hidden_size=16,intermediate_size=32,num_heads=2,out_hidden_size=16,
            patch_size=16,temporal_patch_size=2,spatial_merge_size=2,deepstack_visual_indexes=[0,1,2],num_position_embeddings=64))
    tc._attn_implementation='sdpa';tc.text_config._attn_implementation='sdpa';tc.vision_config._attn_implementation='sdpa'
    torch.manual_seed(91)
    target=Qwen3VLModel(tc).eval().requires_grad_(False)
    with torch.no_grad():
        output=target(**tiny_inputs,output_hidden_states=True,use_cache=False)
    assert len(output.hidden_states)>=4
    hidden=torch.cat([output.hidden_states[i+1] for i in [0,1,2]],-1)
    changed={**tiny_inputs,'pixel_values':tiny_inputs['pixel_values']+0.5}
    with torch.no_grad():
        changed_output=target(**changed,output_hidden_states=True,use_cache=False)
    changed_hidden=torch.cat([changed_output.hidden_states[i+1] for i in [0,1,2]],-1)
    image_effect=(hidden-changed_hidden).abs().max().item()
    assert image_effect>0
    results['tiny_target']={'tokens':tiny_inputs['input_ids'].shape[1],
        'captured_shape':list(hidden.shape),'image_perturbation_hidden_difference':image_effect}

    for dtype in [torch.float32,torch.bfloat16]:
        dc=draft_config();dc.num_hidden_layers=2;dc.layer_types=['full_attention']*2
        dc.num_target_layers=4;dc.dflash_config.update({'target_layer_ids':[0,1,2],'mask_token_id':151669})
        dc.vocab_size=151936;dc.draft_vocab_size=151936;dc.eos_token_id=tokenizer.eos_token_id
        dc.architectures=['DSparkDraftModel']
        draft=DSparkDraftModel(dc).to(dtype=dtype)
        head=nn.Linear(16,151936,bias=False).to(dtype=dtype).requires_grad_(False)
        embedding=nn.Embedding(151936,16).to(dtype=dtype).requires_grad_(False)
        embedding.weight.data.copy_(target.language_model.embed_tokens.weight.to(dtype))
        model=OnlineDSparkModel(draft,head,embedding,151669,block_size=4,num_anchors=16,
            attention_backend='sdpa',objective_chunk_blocks=1)
        strategy=DSparkTrainStrategy(model)
        batch=TrainBatch(sample_ids=['single-image'],strategy='dspark',tensors={
            'input_ids':torch.tensor([tiny_payload['input_ids']]),'loss_mask':torch.tensor([tiny_payload['loss_mask']]),
            'position_ids':torch.tensor([tiny_payload['position_ids']]),'hidden_states':hidden.to(dtype),
            'target_last_hidden_states':output.last_hidden_state.to(dtype)})
        opt=torch.optim.SGD(draft.parameters(),lr=0.01)
        losses=[]
        for step in range(2):
            opt.zero_grad(set_to_none=True)
            out=strategy.forward_loss(batch,StepContext(collect_detailed_metrics=False))
            assert torch.isfinite(out.loss)
            out.loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in draft.parameters() if p.requires_grad)
            assert head.weight.grad is None and embedding.weight.grad is None
            opt.step();losses.append(out.loss.item())
        state={'draft_state_dict':strategy.checkpoint_state_filter(model.state_dict()),**resume_contract(None,draft,model)}
        config_path=Path(temp)/'draft.json';config_path.write_text(json.dumps(dc.to_dict()))
        loaded=materialize_draft(state,str(config_path));loaded.config._attn_implementation='sdpa'
        before=strategy.forward_loss(batch,StepContext(collect_detailed_metrics=False)).loss
        model.draft_model=loaded
        after=strategy.forward_loss(batch,StepContext(collect_detailed_metrics=False)).loss
        diff=(before-after).abs().item()
        assert diff==0,(dtype,diff)
        results[str(dtype)]={'losses':losses,'reload_loss_difference':diff,'finite_all_draft_gradients':True,
            'frozen_embedding_and_head':True,'mrope_enabled_after_reload':loaded.use_interleaved_mrope}

    from specforge.modeling.auto import AutoDraftModel
    from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
    from specforge.algorithms.dflash.providers import resume_contract as dflash_contract
    from specforge.training.strategies.base import DFlashTrainStrategy
    from tests.test_algorithms.test_dflash_vl_mrope import config as dflash_config
    for name in ('DFlashDraftModel','DFlash2DraftModel'):
        for dtype in (torch.float32,torch.bfloat16):
            dc=dflash_config();dc.architectures=[name]
            dc.vocab_size=151936;dc.eos_token_id=tokenizer.eos_token_id
            dc.num_target_layers=4
            dc.dflash_config.update(target_layer_ids=[0,1,2],mask_token_id=151669,block_size=4,
                conv_kernel_size=2,conv_group_size=4,selector_rank=4,selector_top_k=16)
            draft=AutoDraftModel.from_config(dc).to(dtype)
            head=nn.Linear(16,151936,bias=False).to(dtype).requires_grad_(False)
            embedding=nn.Embedding(151936,16).to(dtype).requires_grad_(False)
            embedding.weight.data.copy_(target.language_model.embed_tokens.weight.to(dtype))
            model=OnlineDFlashModel(draft,head,embedding,151669,block_size=4,num_anchors=16,
                attention_backend='sdpa',objective_chunk_blocks=1,teacher_metrics=False,
                loss_type='dpace' if name=='DFlash2DraftModel' else 'dflash',selector_loss_alpha=1.)
            batch=TrainBatch(sample_ids=['single-image'],strategy='dflash',tensors={
                'input_ids':torch.tensor([tiny_payload['input_ids']]),'loss_mask':torch.tensor([tiny_payload['loss_mask']]),
                'position_ids':torch.tensor([tiny_payload['position_ids']]),'hidden_states':hidden.to(dtype)})
            strategy=DFlashTrainStrategy(model);opt=torch.optim.SGD(draft.parameters(),lr=.01)
            losses=[]
            for _ in range(2):
                opt.zero_grad(set_to_none=True)
                out=strategy.forward_loss(batch,StepContext(collect_detailed_metrics=False))
                assert torch.isfinite(out.loss)
                out.loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in draft.parameters() if p.requires_grad)
                assert head.weight.grad is None and embedding.weight.grad is None
                opt.step();losses.append(out.loss.item())
            state={'draft_state_dict':strategy.checkpoint_state_filter(model.state_dict()),**dflash_contract(None,draft,model)}
            path=Path(temp)/'dflash-config.json';path.write_text(json.dumps(dc.to_dict()))
            loaded=materialize_draft(state,str(path));loaded.config._attn_implementation='sdpa'
            before=strategy.forward_loss(batch,StepContext(collect_detailed_metrics=False)).loss
            model.draft_model=loaded
            after=strategy.forward_loss(batch,StepContext(collect_detailed_metrics=False)).loss
            assert torch.equal(before,after)
            results[name+'-'+str(dtype)]={'losses':losses,'reload_loss_difference':0.,
                'finite_all_draft_gradients':True,'mrope_enabled_after_reload':loaded.use_interleaved_mrope}

    from datasets import Dataset
    from specforge.data.vlm_preprocessing import build_vlm_prompt_payloads,build_mapped_vlm_prompt_payloads,_exact_qwen_turn
    import numpy as np
    image_path=Path(temp)/'exact.png';Image.new('RGB',(96,64),(13,71,29)).save(image_path)
    messages=[{'role':'user','content':[{'type':'text','text':'Read this: '},{'type':'image'},
        {'type':'text','text':' What is visible?'}]},{'role':'assistant','content':'ab'}]
    # Intentionally preserve a valid noncanonical tokenization of the response.
    saved_ids=tokenizer.encode('a',add_special_tokens=False)+tokenizer.encode('b',add_special_tokens=False)
    prefix_text=tokenizer.apply_chat_template(messages[:-1],tokenize=False,add_generation_prompt=True)
    prefix=tokenizer.encode(prefix_text,add_special_tokens=False)
    row=dict(image=str(image_path),conversations=messages,generation_prompt_ids=prefix,response_token_ids=saved_ids)
    raw=Path(temp)/'exact.jsonl';raw.write_text(json.dumps(row)+'\n')
    payload=build_vlm_prompt_payloads(str(raw),tokenizer,processor,chat_template='qwen',max_length=4096,
        min_loss_tokens=2,generation_eos_token_id=tokenizer.eos_token_id)[0]['payload']
    supervised=[i for i,m in zip(payload['input_ids'],payload['loss_mask']) if m]
    assert supervised==saved_ids+[tokenizer.eos_token_id]
    pil=Image.open(image_path).convert('RGB')
    assert server._expand_input_ids(payload['request_input_ids'],server.resolve_image_token_counts([pil]),target_config.image_token_id)==payload['input_ids']
    grid=processor.image_processor(images=[pil],return_tensors='pt')['image_grid_thw'].tolist()
    cached_mask=list(payload['loss_mask']);cached_mask[-1]=1  # Legacy cache supervised the suffix newline.
    cache=Path(temp)/'arrow';Dataset.from_list([dict(input_ids=[payload['input_ids']],
        loss_mask=[cached_mask],image_grid_thw=grid)]).save_to_disk(str(cache))
    manifest=dict(complete=True,storage_format='sharded_manifest',chat_template='qwen',max_length=4096,
        target_model_path=target_path,dataset_size=1,shuffle_seed=42,
        shards=[dict(rows=1,files=[str(p) for p in cache.glob('*.arrow')])])
    mp=Path(temp)/'manifest.json';mp.write_text(json.dumps(manifest))
    reused=build_mapped_vlm_prompt_payloads(str(raw),str(mp),tokenizer,target_model_path=target_path,
        chat_template='qwen',max_length=4096,min_loss_tokens=2,processor=processor)[0]['payload']
    for key in ('input_ids','request_input_ids','loss_mask','position_ids','image_data'):
        assert reused[key]==payload[key],key
    results['real_exact_and_cached']={'saved_response_ids':saved_ids,
        'canonical_response_ids':tokenizer.encode('ab',add_special_tokens=False),
        'exact_response_preserved':True,'image_after_text':True,'cache_equal':True,'eos_only':True}

# Validate every saved single-turn token prefix in the existing aligned dataset,
# without loading its images or claiming a full-size GPU training run.
aligned=Path('/XYAIFS00/HDD_POOL/nsccgz_zgchen/nsccgz_zgchen_8/xj/experiments/pressure-original2272-input-alignment-audit-20261123/train.aligned.jsonl')
count=0
if aligned.is_file():
    for line in aligned.read_text().splitlines():
        row=json.loads(line)
        ids,mask=_exact_qwen_turn(row,tokenizer,True,'historical aligned row')
        assert [i for i,m in zip(ids,mask) if m]==row['response_token_ids']+[tokenizer.eos_token_id]
        count+=1
    assert count==2272,count
results['aligned_exact_rows_checked']=count

from specforge.config.schema import load_config
recipes=Path('examples/configs/online/disaggregated/managed-local')
results['recipes']={}
for filename in ('qwen3-vl-4b-redhatai-dflash2-b9-mrope-mixed40031-dsparkmatched.yaml',
                 'qwen3-vl-4b-redhatai-dflash2-b9-mrope-mixed41561.yaml',
                 'qwen3-vl-4b-dflash2-mrope-original2272.yaml'):
    c=load_config(str(recipes/filename));dc=AutoConfig.from_pretrained(c.model.draft_model_config,local_files_only=True)
    assert dc.architectures==['DFlash2DraftModel']
    from specforge.modeling.draft.dflash import get_rope_scaling_value
    assert get_rope_scaling_value(dc,'mrope_interleaved') is True
    results['recipes'][filename]={'strategy':c.training.strategy,'block_size':dc.dflash_config['block_size'],
        'theta':get_rope_scaling_value(dc,'rope_theta',getattr(dc,'rope_theta',None)),
        'output_dir':c.output_dir}
print('UNIFIED_SINGLE_IMAGE_AUDIT_RESULTS')
print(json.dumps(results,indent=2,ensure_ascii=False))
