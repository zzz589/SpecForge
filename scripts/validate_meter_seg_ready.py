#!/usr/bin/env python3
import hashlib, json, pathlib, re

ROOT=pathlib.Path('/HOME/nsccgz_zgchen/nsccgz_zgchen_8/HDD_POOL/xj/data/meter_seg/processed-dflash2-v2')
FILES={'train':ROOT/'train-ready/meter-qwen3vl4b-regen-train-374.jsonl','val':ROOT/'train-ready/meter-qwen3vl4b-regen-val-40.jsonl'}
rows={}; errors=[]; coverage={k:0 for k in ('meter_type','reading','location','environment')}; word_counts=[]
patterns={'meter_type':r'meter type|instrument type|meter category','reading':r'displayed reading|displayed value|displayed measurement|indicated value','location':r'installation location|installation position|mounting location|installation setting','environment':r'surrounding environment|surrounding scene|visible surroundings|surroundings'}
for split,path in FILES.items():
    rows[split]=[]
    for n,line in enumerate(path.read_text().splitlines(),1):
        r=json.loads(line); rows[split].append(r); sid=r.get('id',f'{split}:{n}')
        c=r.get('conversations'); image=pathlib.Path(r.get('image','')); mask=pathlib.Path(r.get('segmentation_annotation',''))
        if not isinstance(c,list) or len(c)!=2 or [x.get('role') for x in c]!=['user','assistant']: errors.append(f'{sid}: roles') ; continue
        q=c[0].get('content'); a=c[1].get('content')
        if not isinstance(q,str) or not q.startswith('<image>\n') or 'base64,' in q: errors.append(f'{sid}: question format')
        if not isinstance(a,str) or not a.strip() or '<think' in a.lower(): errors.append(f'{sid}: answer format'); continue
        if r.get('reference_answer')!=a: errors.append(f'{sid}: reference mismatch')
        if not image.is_file() or not mask.is_file() or image.stem!=mask.stem: errors.append(f'{sid}: image/mask')
        low=a.lower(); word_counts.append(len(re.findall(r"[A-Za-z0-9.]+",a)))
        for k,p in patterns.items(): coverage[k]+=bool(re.search(p,low))
train_ids={r['id'] for r in rows['train']}; val_ids={r['id'] for r in rows['val']}
train_images={r['image'] for r in rows['train']}; val_images={r['image'] for r in rows['val']}
if len(rows['train'])!=374 or len(rows['val'])!=40: errors.append('split counts')
if train_ids & val_ids or train_images & val_images: errors.append('train/val overlap')
report={'status':'pass' if not errors else 'fail','rows':{k:len(v) for k,v in rows.items()},'errors':errors,
        'train_val_id_overlap':len(train_ids&val_ids),'train_val_image_overlap':len(train_images&val_images),
        'field_label_coverage':coverage,'answer_word_count':{'min':min(word_counts),'mean':sum(word_counts)/len(word_counts),'max':max(word_counts)}}
(ROOT/'train-ready/validation_report.json').write_text(json.dumps(report,indent=2)+'\n')
with (ROOT/'train-ready/MANIFEST.sha256').open('w') as out:
    for path in [FILES['train'],FILES['val'],ROOT/'prepare_manifest.json',ROOT/'train-ready/train_ready_report.json',ROOT/'train-ready/validation_report.json']:
        out.write(f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(ROOT/"train-ready") if path.is_relative_to(ROOT/"train-ready") else "../"+path.name}\n')
print(json.dumps(report,indent=2))
if errors: raise SystemExit(1)
