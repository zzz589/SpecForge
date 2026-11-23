#!/usr/bin/env python3
import argparse, datetime, hashlib, json, os, pathlib, platform, subprocess, yaml
ROOT='/HOME/nsccgz_zgchen/nsccgz_zgchen_8/HDD_POOL/xj'
INIT=ROOT+'/experiments_results/qwen3-4b-redhatai-dflash2-specforge-b9-20261105/Qwen3-4B-speculator.dflash2-specforge-b9-qwen3vl-mrope-v1'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser()
 for x in ('exp','data','manifest','base-config','repo','env-root'): ap.add_argument('--'+x,required=True)
 ap.add_argument('--status',choices=['prepared','complete'],required=True); a=ap.parse_args()
 exp=pathlib.Path(a.exp)
 for d in ('config','commands','provenance','training_curves','validation','evaluation','runtime_audit'): (exp/d).mkdir(parents=True,exist_ok=True)
 data=pathlib.Path(a.data); manifest=json.loads(pathlib.Path(a.manifest).read_text()); n=sum(1 for x in data.open() if x.strip()); steps=(n*3+3)//4
 if n!=manifest['output_rows'] or sha(data)!=manifest['output_sha256']: raise ValueError('merged data/manifest mismatch')
 base=pathlib.Path(a.base_config); (exp/'config/base.yaml').write_text(base.read_text()); cfg=yaml.safe_load(base.read_text())
 cfg['data']['train_data_path']=str(data); cfg['data']['cache_dir']=str(exp/'cache'); cfg['training']['max_steps']=steps; cfg['training']['save_interval']=steps; cfg['output_dir']=str(exp/'output'); cfg['deployment']['disaggregated']['control_dir']=str(exp/'control'); cfg['deployment']['disaggregated']['consumer_state_dir']=str(exp/'consumer-state')
 (exp/'config/resolved.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False)); (exp/'data_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 dist=['# 混合训练数据分布','',f'- 总数：{n}',f'- Optimizer steps：{steps}','- 3 epochs，global batch 4，LR 1e-5，D-PACE','- 合并后按图像 SHA256 跨数据集去重，种子 20261116。','','| 场景 | 数量 |','|---|---:|']+[f'| {k} | {v} |' for k,v in manifest['scene_distribution'].items()]
 (exp/'DATA_DISTRIBUTION.md').write_text('\n'.join(dist)+'\n')
 cmd=f"cd {a.repo}\nexport CUDA_HOME={a.env_root}/lib/python3.11/site-packages/nvidia/cu13\nexport PATH={a.env_root}/bin:$CUDA_HOME/bin:$PATH\nexport LD_LIBRARY_PATH=$CUDA_HOME/lib:/lib/x86_64-linux-gnu:${{LD_LIBRARY_PATH:-}}\n{a.env_root}/bin/python -m specforge.cli train -c {exp}/config/resolved.yaml --role auto\n"; (exp/'commands/train.sh').write_text(cmd); os.chmod(exp/'commands/train.sh',0o755)
 commit=subprocess.run(['git','-C',a.repo,'rev-parse','HEAD'],capture_output=True,text=True).stdout.strip(); dirty=subprocess.run(['git','-C',a.repo,'status','--short'],capture_output=True,text=True).stdout; (exp/'provenance/code.txt').write_text(f'commit={commit}\n\n{dirty}'); (exp/'provenance/environment.txt').write_text(f'generated_at={datetime.datetime.now().astimezone().isoformat()}\npython={platform.python_version()}\nenv={a.env_root}\ncapture_gpu=0\ntrainer_gpus=3,4\n')
 card=f'''# qwen3-vl-4b-dflash2-dpace-mixed-all-scenes-e3-lr1e-5-20261116

- 状态：{a.status}
- 数据量：{n}
- Optimizer steps：{steps}
- Epochs：3；global batch：4
- LR：1e-5，cosine，warmup 0.03
- DFlash2-VL + D-PACE（alpha=0.5）；selector alpha=1.0；32 anchors；block size 9
- GPU：capture 0；trainer 3,4
- Init：RedHatAI DFlash2 b9 Qwen3-VL mRoPE v1，不从任何场景 checkpoint 续训
'''; (exp/'EXPERIMENT_CARD.md').write_text(card)
 em={'schema_version':1,'experiment_id':cfg['run_id'],'status':a.status,'training':{'rows':n,'steps':steps,'epochs':3,'global_batch':4,'learning_rate':1e-5,'loss_type':'dpace','dpace_alpha':0.5,'init_checkpoint':INIT},'scene_distribution':manifest['scene_distribution'],'resolved_config':str(exp/'config/resolved.yaml')}; (exp/'experiment_manifest.json').write_text(json.dumps(em,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'rows':n,'steps':steps,'status':a.status},indent=2))
if __name__=='__main__': main()
