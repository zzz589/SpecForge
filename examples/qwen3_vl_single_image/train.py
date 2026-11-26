"""Render one portable PR785 recipe, train, then export and normalize for SGLang."""
import argparse,os,sys,socket,pathlib,subprocess,json
import yaml
P=pathlib.Path
def run(argv,env=None):subprocess.run(argv,check=True,env=env)
def main():
    a=argparse.ArgumentParser()
    a.add_argument('algorithm',choices=['dflash2','dflash','dspark'])
    for key in ('target','data','out','sglang'):a.add_argument('--'+key,required=True)
    a.add_argument('--init');a.add_argument('--steps',required=True,type=int);a.add_argument('--epochs',type=int,default=1)
    a.add_argument('--train-gpus',default='0,1');a.add_argument('--capture-gpu',default='2')
    args=a.parse_args();algo=args.algorithm;repo=P.cwd();target=P(args.target).resolve();data=P(args.data).resolve()
    train=args.train_gpus.split(',');assert args.capture_gpu not in train and len(set(train))==len(train)
    assert args.steps>0 and args.epochs>0
    base=P(__file__).with_name('base.yaml')
    cfg=yaml.safe_load(base.read_text());work=P(args.out).resolve()/algo;work.mkdir(parents=True,exist_ok=False)
    model=cfg['model'];t=cfg['training'];model['target_model_path']=str(target)
    if algo=='dflash2':
        if not args.init:raise ValueError('DFlash2 requires --init (HF mRoPE draft with weights)')
        dc=P(args.init).resolve();model['draft_checkpoint_path']=str(dc)
    else:
        dc=repo/f'configs/qwen3-vl-4b-{algo}-mrope.json';model.pop('draft_checkpoint_path')
        for key in list(t):
            if key.startswith('dflash2_') or key in ('lk_loss_type','kl_scale','kl_decay','dpace_alpha'):del t[key]
        t.update(strategy=algo,loss_type='dflash')
        if algo=='dspark':
            t.pop('dflash_teacher_metrics',None);t.update(dspark_ce_loss_alpha=.1,dspark_l1_loss_alpha=.9,dspark_confidence_head_alpha=1.)
    model['draft_model_config']=str(dc)
    cfg.update(run_id='vl-'+algo,output_dir=str(work/'output'))
    cfg['data'].update(train_data_path=str(data),cache_dir=str(work/'cache'),build_dataset_num_proc=1,dataloader_num_workers=0)
    t.update(num_epochs=args.epochs,max_steps=args.steps,save_interval=args.steps,log_interval=1,accumulation_steps=1)
    socks=[socket.socket() for _ in range(5)]
    for s in socks:s.bind(('127.0.0.1',0))
    capture,master,rpc,metadata,metrics=[s.getsockname()[1] for s in socks]
    dep=cfg['deployment'];dep['trainer'].update(nproc_per_node=len(train),master_port=master)
    d=dep['disaggregated'];d.update(control_dir=str(work/'control'),consumer_state_dir=str(work/'consumer-state'))
    local=d['managed_local'];local['trainer_cuda_visible_devices']=train
    local['capture_servers']=[dict(port=capture,cuda_visible_devices=[args.capture_gpu],tp_size=1,mem_fraction_static=.45,startup_timeout_s=420,env={'SGLANG_ENABLE_JIT_DEEPGEMM':'0'})]
    local['mooncake'].update(rpc_port=rpc,metadata_port=metadata,metrics_port=metrics,global_segment_size_bytes=8<<30,local_buffer_size_bytes=1<<30,default_kv_lease_ttl_ms=500)
    conf=work/'train.yaml';conf.write_text(yaml.safe_dump(cfg,sort_keys=False))
    sg=P(args.sglang).resolve();overlay=subprocess.check_output([sys.executable,'scripts/prepare_sglang_capture_overlay.py','--base',str(sg)],text=True).strip()
    env=dict(os.environ,PYTHONPATH=os.pathsep.join(map(str,[repo,overlay,sg/'python'])))
    env.pop('CUDA_VISIBLE_DEVICES',None)
    for s in socks:s.close()
    cli=[sys.executable,'-m','specforge.cli'];run(cli+['train','-c',str(conf),'--role','auto'],env)
    ckpt=work/'output'/f'vl-{algo}-step{args.steps}';export=work/'exported'
    dc=dc/'config.json' if dc.is_dir() else dc
    run(cli+['export','--to','hf','--checkpoint',str(ckpt),'--draft-config',str(dc),'--output-dir',str(export),'--embedding-source',str(target),'--embedding-key',model['embedding_key']],dict(env,CUDA_VISIBLE_DEVICES=''))
    run([sys.executable,'scripts/gates/normalize_dflash_export.py','--config',str(export/'config.json'),'--block-size',str(json.loads(dc.read_text())['block_size'])],env)
    cfg=json.loads((export/'config.json').read_text());rope=cfg.get('rope_parameters') or cfg['rope_scaling']
    assert rope['mrope_interleaved'] and rope['mrope_section']==[24,20,20]
    print('TRAIN_EXPORT_OK',algo,flush=True)
if __name__=='__main__':main()
