#!/usr/bin/env python3
import argparse, collections, datetime, hashlib, json, os, pathlib, platform, subprocess
import yaml

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--size',type=int,required=True); ap.add_argument('--steps',type=int,required=True); ap.add_argument('--exp',required=True); ap.add_argument('--data',required=True); ap.add_argument('--base-config',required=True); ap.add_argument('--repo',required=True); ap.add_argument('--env-root',required=True); ap.add_argument('--status',choices=['prepared','complete'],required=True); a=ap.parse_args()
    exp=pathlib.Path(a.exp); [ (exp/x).mkdir(parents=True,exist_ok=True) for x in ('config','commands','provenance','training_curves','validation','evaluation','runtime_audit') ]
    data=pathlib.Path(a.data); rows=[json.loads(x) for x in data.read_text().splitlines() if x.strip()]
    if len(rows)!=a.size: raise ValueError((len(rows),a.size))
    ranges=collections.Counter(str(r.get('rpm10k_metadata',{}).get('range','unknown')) for r in rows)
    grads=collections.Counter(str(r.get('rpm10k_metadata',{}).get('graduation value','unknown')) for r in rows)
    image_kinds=collections.Counter('resized' if 'resized-images' in str(r.get('image','')) else 'original' for r in rows)
    answers=[len(str(r.get('reference_answer',''))) for r in rows]
    manifest={'schema_version':1,'dataset':'RPM-10K','split':'train','rows':len(rows),'images_present':sum(bool(r.get('image')) for r in rows),'nested_subset_seed':20261109,'nested_subsets':'1000 ⊂ 2000 ⊂ ... ⊂ 8729','data_path':str(data),'sha256':sha(data),'id_sha256':hashlib.sha256(('\n'.join(str(r.get('id')) for r in rows)+'\n').encode()).hexdigest(),'range_distribution':dict(sorted(ranges.items())),'graduation_value_distribution':dict(sorted(grads.items())),'image_path_distribution':dict(image_kinds),'reference_answer_chars':{'min':min(answers),'max':max(answers),'mean':sum(answers)/len(answers)}}
    (exp/'data_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    dist=['# 数据分布说明','',f'- 实际训练行数：{a.size}',f'- 数据文件：`{data}`',f'- 数据 SHA256：`{manifest["sha256"]}`','- 每行包含一张图、一个英文读表问题和重生成回答。','- 子集固定种子 20261109，保持 1k ⊂ 2k ⊂ ... ⊂ 8729。','','## 量程分布','', '| 量程 | 数量 |','|---|---:|']+[f'| {k} | {v} |' for k,v in sorted(ranges.items())]+['','## 最小刻度值分布','', '| 最小刻度 | 数量 |','|---|---:|']+[f'| {k} | {v} |' for k,v in sorted(grads.items())]
    (exp/'DATA_DISTRIBUTION.md').write_text('\n'.join(dist)+'\n')
    base=pathlib.Path(a.base_config); (exp/'config'/'base.yaml').write_text(base.read_text())
    cfg=yaml.safe_load(base.read_text()); root='/HOME/nsccgz_zgchen/nsccgz_zgchen_8/HDD_POOL/xj'; run=f'qwen3-vl-4b-dflash2-dpace-rpm10k-n{a.size}-e3-lr1e-5-20261110'
    cfg['data']['train_data_path']=str(data); cfg['data']['cache_dir']=str(exp/'cache'); cfg['training']['max_steps']=a.steps; cfg['training']['save_interval']=a.steps; cfg['run_id']=run; cfg['output_dir']=str(exp/'output'); cfg['deployment']['disaggregated']['control_dir']=str(exp/'control'); cfg['deployment']['disaggregated']['consumer_state_dir']=str(exp/'consumer-state')
    (exp/'config'/'resolved.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    cmd=f'''cd {a.repo}\nexport CUDA_HOME={a.env_root}/lib/python3.11/site-packages/nvidia/cu13\nexport PATH={a.env_root}/bin:$CUDA_HOME/bin:$PATH\nexport LD_LIBRARY_PATH=$CUDA_HOME/lib:/lib/x86_64-linux-gnu:${{LD_LIBRARY_PATH:-}}\n{a.env_root}/bin/python -m specforge.cli train -c {exp}/config/resolved.yaml --role auto\n'''; (exp/'commands'/'train.sh').write_text(cmd); os.chmod(exp/'commands'/'train.sh',0o755)
    commit=subprocess.run(['git','-C',a.repo,'rev-parse','HEAD'],capture_output=True,text=True).stdout.strip(); status=subprocess.run(['git','-C',a.repo,'status','--short'],capture_output=True,text=True).stdout
    (exp/'provenance'/'code.txt').write_text(f'commit={commit}\n\n{status}')
    (exp/'provenance'/'environment.txt').write_text(f'generated_at={datetime.datetime.now().astimezone().isoformat()}\npython={platform.python_version()}\nenv={a.env_root}\ncapture_gpu=6\ntrainer_gpus=3,5\n')
    card=['# '+run,'',f'- 状态：{a.status}',f'- 数据量：{a.size}',f'- Optimizer steps：{a.steps}','- Epochs：3','- Global batch：4','- LR：1e-5，cosine，warmup 0.03','- 目标：DFlash2 + D-PACE（loss_type=dpace，dpace_alpha=0.5）','- LK：lambda，kl_scale=0.1，kl_decay=0.0','- Selector：alpha=1.0，stop_gradient=false','- Anchors：32；DFlash2 block size：9','- GPU：capture 2；trainer 3,5','- Init：RedHatAI DFlash2 b9 Qwen3-VL mRoPE v1','- 与其他规模独立训练，不从前一规模续训。','','数据详情见 `DATA_DISTRIBUTION.md` 与 `data_manifest.json`。训练完成后曲线、验证和推理链路审计分别写入对应目录。']
    (exp/'EXPERIMENT_CARD.md').write_text('\n'.join(card)+'\n')
    expman={'schema_version':1,'experiment_id':run,'status':a.status,'training':{'size':a.size,'steps':a.steps,'epochs':3,'global_batch':4,'learning_rate':1e-5,'loss_type':'dpace','dpace_alpha':0.5,'init_checkpoint':root+'/experiments_results/qwen3-4b-redhatai-dflash2-specforge-b9-20261105/Qwen3-4B-speculator.dflash2-specforge-b9-qwen3vl-mrope-v1','independent_from_other_scales':True},'data_manifest':str(exp/'data_manifest.json'),'resolved_config':str(exp/'config'/'resolved.yaml')}
    (exp/'experiment_manifest.json').write_text(json.dumps(expman,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__': main()
