"""Run from the project root. Images in the input JSONL are relative to cwd."""
import argparse, base64, json, mimetypes, os
from pathlib import Path

def main():
    parser=argparse.ArgumentParser()
    for key in ('target','input','output'):parser.add_argument('--'+key, required=True)
    parser.add_argument('--max-new-tokens',type=int,default=1024)
    args=parser.parse_args()
    import sglang
    from transformers import AutoTokenizer
    from specforge.data.vlm_preprocessing import _exact_qwen_turn
    tok=AutoTokenizer.from_pretrained(args.target)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError(out)
    tmp=out.with_name(out.name+'.partial')
    with tmp.open('x',encoding='utf8') as sink:
        engine=sglang.Engine(model_path=args.target,dtype='bfloat16',context_length=6159,mem_fraction_static=.55,disable_cuda_graph=True)
        try:
            count=0
            with open(args.input,encoding='utf8') as source:
                for line in source:
                    if not line.strip():continue
                    row=json.loads(line); path=Path(row['image']); question=row['question']
                    assert isinstance(question,str) and question.strip()
                    user={'role':'user','content':[{'type':'text','text':question},{'type':'image'}]}
                    template={'enable_thinking':False}
                    prompt=tok.apply_chat_template([user],tokenize=False,add_generation_prompt=True,**template)
                    prefix=tok.encode(prompt,add_special_tokens=False)
                    image='data:'+(mimetypes.guess_type(path.name)[0] or 'image/jpeg')+';base64,'+base64.b64encode(path.read_bytes()).decode()
                    answer=engine.generate(input_ids=prefix,image_data=image,sampling_params={'temperature':0,'max_new_tokens':args.max_new_tokens})
                    meta=answer['meta_info']; reason=meta['finish_reason']
                    if reason['type']!='stop':raise ValueError(f'row {count+1}: {reason["type"]}; increase token budget or filter sample')
                    ids=list(answer['output_ids'])
                    if ids and ids[-1]==tok.eos_token_id:ids.pop()
                    text=tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
                    record=dict(id=row.get('id',str(count)),image=row['image'],conversations=[user,{'role':'assistant','content':text}],generation_prompt_ids=prefix,response_token_ids=ids,chat_template_kwargs=template)
                    _exact_qwen_turn(record,tok,True,f'row {count+1}')
                    sink.write(json.dumps(record,ensure_ascii=False)+'\n');count+=1
            if not count:raise ValueError('empty input')
        finally:engine.shutdown()
    os.rename(tmp,out)
    print('REGENERATED',count,flush=True)
if __name__=='__main__':main()
