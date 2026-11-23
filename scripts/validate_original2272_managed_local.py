"""Two-step isolated managed-local validation; reserve and recheck idle devices."""
import csv
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import yaml

repo=Path(__file__).resolve().parents[1]
root=repo.parents[1]
audit=root/"experiments/pressure-original2272-input-alignment-audit-20261123"
old=root/"experiments/pressure-external-originalbytes-thinking128-fullattn-mrope-n2272-20261122"
assert repo.name.endswith("mrope-aligned-20261123") and (audit/"verified.json").is_file()
work=audit/"managed-local-smoke"
work.mkdir()
evidence=[json.loads(s) for s in (audit/"records.jsonl").read_text().splitlines()]
repairs={r["id"] for r in evidence if r["corrected_tokenization"]}
rows=[json.loads(s) for s in (audit/"train.aligned.jsonl").read_text().splitlines()]
samples=[next(r for r in rows if r["id"] in repairs),*rows[:3]]
assert len({r["id"] for r in samples})==4
(work/"samples.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in samples))
def gpu_rows():
    args=["nvidia-smi","--format=csv,noheader,nounits","--query-gpu=index,uuid,memory.used,utilization.gpu"]
    return list(csv.reader(subprocess.check_output(args,text=True).splitlines(),skipinitialspace=True))
def busy():
    return set(subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid","--format=csv,noheader"],text=True).splitlines())
locks=[]
cards=[]
for index,uuid,mem,util in reversed(gpu_rows()):
    if int(mem)>0 or int(util)>0 or uuid in busy():
        continue
    lock=open(f"/tmp/xj-dflash2-alignment-{uuid}.lock","a")
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        continue
    locks.append(lock)
    cards.append(index)
    if len(cards)==2:
        break
assert len(cards)==2, "two completely idle GPUs required"
assert all(int(m)==int(u)==0 and uuid not in busy() for i,uuid,m,u in gpu_rows() if i in cards)
ports=[]
for _ in range(5):
    s=socket.socket()
    s.bind(("127.0.0.1",0))
    ports.append(s)
capture,master,rpc,metadata,metrics=[s.getsockname()[1] for s in ports]
cfg=yaml.safe_load((old/"config.yaml").read_text())
cfg["run_id"]="original2272-alignment-smoke"
cfg["output_dir"]=str(work/"output")
cfg["data"].update(train_data_path=str(work/"samples.jsonl"),cache_dir=str(work/"cache"),build_dataset_num_proc=1,dataloader_num_workers=0)
cfg["training"].update(max_steps=2,num_epochs=1,accumulation_steps=1,log_interval=1,save_interval=2)
cfg["deployment"]["trainer"].update(nproc_per_node=1,master_port=master)
dis=cfg["deployment"]["disaggregated"]
dis.update(control_dir=str(work/"control"),consumer_state_dir=str(work/"consumer-state"))
local=dis["managed_local"]
local["trainer_cuda_visible_devices"]=[cards[0]]
local["capture_servers"][0].update(cuda_visible_devices=[cards[1]],port=capture)
local["mooncake"].update(rpc_port=rpc,metadata_port=metadata,metrics_port=metrics)
config=work/"config.yaml"
config.write_text(yaml.safe_dump(cfg,sort_keys=False))
for s in ports:
    s.close()
env = dict(os.environ,PYTHONPATH=str(repo),PYTHONDONTWRITEBYTECODE="1",TOKENIZERS_PARALLELISM="false")
with (work/"train.log").open("w") as log:
    subprocess.run([sys.executable,"-m","specforge.cli","train","-c",str(config),"--role","auto"],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
ckpt=work/"output"/(cfg["run_id"]+"-step2")
assert (ckpt/"training_state.pt").stat().st_size>0
cmd=[sys.executable,"-m","specforge.cli","export","--to","hf","--checkpoint",str(ckpt),
     "--draft-config",str(Path(cfg["model"]["draft_model_config"])/"config.json"),"--output-dir",str(ckpt/"exported-draft"),
     "--embedding-source",cfg["model"]["target_model_path"],"--embedding-key",cfg["model"]["embedding_key"]]
with (work/"export.log").open("w") as log:
    subprocess.run(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
from safetensors import safe_open
keys=["candidate_selector.hidden_projection.weight","layers.0.attention_conv.base_kernel"]
changes={}
with safe_open(str(Path(cfg["model"]["draft_checkpoint_path"])/"model.safetensors"),framework="pt") as a, safe_open(str(ckpt/"exported-draft/model.safetensors"),framework="pt") as b:
    for key in keys:
        original=a.get_tensor(key)
        trained=b.get_tensor(key)
        assert trained.isfinite().all()
        changes[key]=int((original!=trained).sum())
        assert changes[key]>0
(work/"passed.json").write_text(json.dumps(dict(checkpoint=str(ckpt),changed_elements=changes,cards=cards),indent=2))
