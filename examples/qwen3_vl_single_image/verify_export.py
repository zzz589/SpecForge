"""Load one exported draft and run one real image through the local target."""
import argparse,base64,json,mimetypes
from pathlib import Path
def main():
    import sglang
    parser=argparse.ArgumentParser()
    for key in ('target','draft','data'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    config=json.loads((Path(args.draft)/'config.json').read_text())
    arch,=config['architectures'];b=config['block_size']
    assert arch in ('DFlashDraftModel','DFlash2DraftModel','Qwen3DSparkModel'),arch
    dspark=arch=='Qwen3DSparkModel'
    kw={'speculative_dspark_block_size':b} if dspark else {'speculative_dflash_block_size':b}
    if arch=='DFlashDraftModel':kw['speculative_draft_attention_backend']='triton'
    with open(args.data,encoding='utf8') as f:row=json.loads(next(f))
    image=Path(row['image']);uri='data:'+(mimetypes.guess_type(image.name)[0] or 'image/jpeg')+';base64,'+base64.b64encode(image.read_bytes()).decode()
    engine=sglang.Engine(model_path=args.target,speculative_draft_model_path=args.draft,speculative_algorithm='DSPARK' if dspark else 'DFLASH',speculative_num_steps=1,speculative_eagle_topk=1,speculative_num_draft_tokens=b+int(dspark),dtype='bfloat16',context_length=6159,mem_fraction_static=.55,max_running_requests=1,disable_radix_cache=True,disable_cuda_graph=True,disable_overlap_schedule=True,**kw)
    try:
        answer=engine.generate(input_ids=row['generation_prompt_ids'],image_data=uri,sampling_params={'temperature':0,'max_new_tokens':24})
        n=answer['meta_info']['completion_tokens'];assert n>0
        print('SERVE_OK',arch,n,flush=True)
    finally:engine.shutdown()
if __name__=='__main__':main()
