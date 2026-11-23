#!/usr/bin/env python3
import argparse, collections, hashlib, json, math, random, statistics
from pathlib import Path
from PIL import Image
from transformers import AutoTokenizer

TEMPLATES = [
"You are a professional substation meter recognition expert. This image shows a meter in a substation. Please describe the image in detail, including the meter type, displayed reading, installation location, and surrounding environment. If any detail is not clearly visible, state that explicitly instead of guessing.",
"As a professional substation instrumentation expert, examine this image and describe the meter type, displayed value and unit, installation position, and surrounding environment. Do not guess details that are not clearly visible.",
"Please analyze this substation meter photograph and report the instrument type, the visible reading with its unit, where the meter is installed, and the nearby equipment or environment. Clearly identify anything that cannot be determined from the image.",
"Inspect the meter shown in this substation image. Provide a detailed description covering its type, indicated value and unit, mounting location, and surrounding scene. Avoid inventing information when the image is unclear.",
"Act as a substation meter recognition specialist. Describe the pictured instrument, including the meter category, displayed measurement, installation setting, and visible surroundings. State when a requested detail is not visible.",
"Describe this substation instrument image in detail. Include the meter type, its displayed reading and unit, its installation location, and the surrounding environment, without making unsupported assumptions.",
]
STYLE = " Respond using exactly four concise labeled bullet points: Meter type, Displayed reading, Installation location, and Surrounding environment. Keep the complete response under 150 words."

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False)+'\n')

def describe(xs):
    ys=sorted(xs)
    def p(q):
        x=(len(ys)-1)*q; a=math.floor(x); b=math.ceil(x)
        return ys[a] if a==b else ys[a]*(b-x)+ys[b]*(x-a)
    return {'count':len(ys),'min':min(ys),'mean':statistics.fmean(ys),'p50':p(.5),'p90':p(.9),'p95':p(.95),'max':max(ys)}

def read_split(root, split):
    rows=[]
    for i,line in enumerate((root/f'{split}.txt').read_text().splitlines(),1):
        parts=line.split()
        if len(parts)!=2: raise ValueError(f'{split}.txt line {i}: expected image and annotation')
        image=(root/parts[0]).resolve(); annotation=(root/parts[1]).resolve()
        if not image.is_file() or not annotation.is_file(): raise FileNotFoundError((image,annotation))
        with Image.open(image) as im: im.verify()
        with Image.open(annotation) as im: im.verify()
        if image.stem != annotation.stem: raise ValueError(f'basename mismatch: {image} {annotation}')
        rows.append((image,annotation))
    return rows

def prepare(a):
    root=Path(a.input_root).resolve(); out=Path(a.output_dir).resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError(f'non-empty output: {out}')
    out.mkdir(parents=True,exist_ok=True)
    rng=random.Random(a.seed); all_rows=[]; split_counts={}; template_counts=collections.Counter(); seen=set()
    for split in ('train','val'):
        pairs=read_split(root,split); split_counts[split]=len(pairs)
        for image,annotation in pairs:
            sid=f'meterseg_{split}_{image.stem}'
            if image.name in seen: raise ValueError(f'train/val overlap: {image.name}')
            seen.add(image.name)
            # Exactly half use the canonical prompt; the remainder use five paraphrases.
            tid=0 if rng.random()<0.5 else rng.randrange(1,len(TEMPLATES))
            template_counts[tid]+=1
            prompt=TEMPLATES[tid]+STYLE
            all_rows.append({'id':sid,'_dataset_source':'meter_seg','_meter_seg_split':split,
                'image':str(image),'segmentation_annotation':str(annotation),'question_template_id':tid,
                'conversations':[{'role':'user','content':'<image>\n'+prompt},
                                 {'role':'assistant','content':'Placeholder for target-model regeneration.'}]})
    write_jsonl(out/'meter-seg-regen-input-414.jsonl',all_rows)
    write_jsonl(out/'meter-seg-regen-input-train-374.jsonl',[r for r in all_rows if r['_meter_seg_split']=='train'])
    write_jsonl(out/'meter-seg-regen-input-val-40.jsonl',[r for r in all_rows if r['_meter_seg_split']=='val'])
    manifest={'status':'prepared','seed':a.seed,'rows':len(all_rows),'split_counts':split_counts,
              'template_counts':dict(sorted(template_counts.items())),'templates':[x+STYLE for x in TEMPLATES],
              'combined_path':str(out/'meter-seg-regen-input-414.jsonl')}
    (out/'prepare_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (out/'QUESTION_TEMPLATES.md').write_text('# English question templates\n\n'+''.join(f'{i}. {x+STYLE}\n\n' for i,x in enumerate(TEMPLATES)))

def finalize(a):
    out=Path(a.output_dir).resolve(); regen=Path(a.regenerated).resolve(); tokenizer=AutoTokenizer.from_pretrained(a.tokenizer,trust_remote_code=True)
    by_split={'train':[],'val':[]}; answer_tokens=[]; answer_chars=[]; templates=collections.Counter(); ids=set()
    for n,line in enumerate(regen.read_text().splitlines(),1):
        row=json.loads(line); sid=row.get('id')
        if row.get('status')!='success' or not sid or sid in ids: raise ValueError(f'bad regenerated row {n}: {sid}')
        ids.add(sid); conv=row.get('conversations')
        if not isinstance(conv,list) or len(conv)!=2 or conv[0].get('role')!='user' or conv[1].get('role')!='assistant': raise ValueError(f'{sid}: conversation shape')
        content=conv[0].get('content')
        if isinstance(content,str):
            question=content.replace('<image>\n','').replace('<image>','').strip()
        elif isinstance(content,list):
            texts=[part.get('text','') for part in content if isinstance(part,dict) and part.get('type')=='text']
            images=[part for part in content if isinstance(part,dict) and part.get('type')=='image_url']
            if len(images)!=1: raise ValueError(f'{sid}: expected one embedded image, got {len(images)}')
            question='\n'.join(texts).strip()
        else: raise ValueError(f'{sid}: unsupported user content')
        if not question: raise ValueError(f'{sid}: empty question')
        answer=conv[1].get('content','').strip()
        if not answer or '<think' in answer.lower(): raise ValueError(f'{sid}: invalid answer')
        split=row['_meter_seg_split']; compact={k:v for k,v in row.items() if k not in {'status','error'}}
        compact['reference_answer']=answer
        compact['conversations']=[{'role':'user','content':'<image>\n'+question},{'role':'assistant','content':answer}]
        by_split[split].append(compact); templates[row['question_template_id']]+=1
        answer_chars.append(len(answer)); answer_tokens.append(len(tokenizer.encode(answer,add_special_tokens=False)))
    if len(by_split['train'])!=374 or len(by_split['val'])!=40: raise ValueError({k:len(v) for k,v in by_split.items()})
    for split in by_split: by_split[split].sort(key=lambda r:r['id'])
    ready=out/'train-ready'; write_jsonl(ready/'meter-qwen3vl4b-regen-train-374.jsonl',by_split['train']); write_jsonl(ready/'meter-qwen3vl4b-regen-val-40.jsonl',by_split['val'])
    report={'status':'pass','source':str(regen),'split_counts':{k:len(v) for k,v in by_split.items()},
            'train_val_id_overlap':len({r['id'] for r in by_split['train']} & {r['id'] for r in by_split['val']}),
            'template_counts':dict(sorted(templates.items())),'answer_characters':describe(answer_chars),
            'answer_tokens':describe(answer_tokens),'estimated_anchor_positions':describe([max(0,x) for x in answer_tokens])}
    (ready/'train_ready_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (ready/'README.zh-CN.md').write_text('# meter_seg DFlash2-VL 数据\n\n- 训练集：374 条\n- 独立验证集：40 条\n- 答案由 Qwen3-VL-4B-Instruct greedy 重生成。\n- 原始 pointer/scale 分割标注路径保留在 `segmentation_annotation`。\n- train 与 val 无 ID 重叠。\n')

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='cmd',required=True)
    q=sub.add_parser('prepare'); q.add_argument('--input-root',required=True); q.add_argument('--output-dir',required=True); q.add_argument('--seed',type=int,default=20261110)
    q=sub.add_parser('finalize'); q.add_argument('--output-dir',required=True); q.add_argument('--regenerated',required=True); q.add_argument('--tokenizer',required=True)
    a=p.parse_args(); prepare(a) if a.cmd=='prepare' else finalize(a)
if __name__=='__main__': main()
