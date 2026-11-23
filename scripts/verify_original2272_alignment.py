"""Rebuild and audit the exact single-turn inputs of the original-image n2272 run."""
import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
from specforge.data.vlm_preprocessing import build_vlm_prompt_payloads

root = Path("/HOME/nsccgz_zgchen/nsccgz_zgchen_8/HDD_POOL/xj")
old = root/"experiments/pressure-external-originalbytes-thinking128-fullattn-mrope-n2272-20261122"
out = root/"experiments/pressure-original2272-input-alignment-audit-20261123"
out.mkdir(exist_ok=True)
model = root/"models/Qwen3-VL-4B-Instruct"
proc = AutoProcessor.from_pretrained(model)
tok = proc.tokenizer
reference = SimpleNamespace(config=AutoConfig.from_pretrained(model))
reference.get_vision_position_ids = Qwen3VLModel.get_vision_position_ids.__get__(reference)
rows = [json.loads(s) for s in (old/"train.jsonl").read_text().splitlines()]
answers = [json.loads(s) for s in (old/"regenerated.jsonl").read_text().splitlines()]
gens = {r["id"]:r for r in answers}
assert len(gens)==len(answers)==len(rows)==len({r["id"] for r in rows})==2272
marker = tok.convert_tokens_to_ids("<|image_pad|>")
suffix = tok.encode("<|im_end|>\n",add_special_tokens=False)
fixed = 0
anchors_checked = 0
staging = out/"train.aligned.pending"
audit = out/"records.pending"
with staging.open("w") as data, audit.open("w") as log:
    for i,row in enumerate(rows):
        g = gens[row["id"]]
        assert row["conversations"][0]["content"]==g["prompt"] and row["conversations"][-1]["content"]==g["raw"]
        message = {"role":"user","content":[{"type":"text","text":g["prompt"]},{"type":"image"}]}
        prompt = tok.apply_chat_template([message],tokenize=False,add_generation_prompt=True,enable_thinking=True)
        prefix = tok.encode(prompt,add_special_tokens=False)
        record = dict(row,conversations=[message,row["conversations"][-1]],generation_prompt_ids=prefix,
                      response_token_ids=g["response_token_ids"],chat_template_kwargs={"enable_thinking":True})
        line = json.dumps(record,ensure_ascii=False)+"\n"
        single = out/"record.tmp.jsonl"
        single.write_text(line)
        payload = build_vlm_prompt_payloads(str(single),tok,proc,chat_template="qwen",max_length=6144)[0]["payload"]
        image_file = Path(row["image"])
        image_hash = hashlib.sha256(base64.b64decode(payload["image_data"].partition(",")[2],validate=True)).hexdigest()
        assert image_hash==g["image_sha256"]==hashlib.sha256(image_file.read_bytes()).hexdigest()
        with Image.open(image_file) as im:
            hf = proc(text=[prompt],images=[im.convert("RGB")],return_tensors="pt")
        expanded = hf.input_ids[0].tolist()
        assert payload["request_input_ids"]==prefix+g["response_token_ids"]+suffix
        expected = expanded+g["response_token_ids"]+suffix
        assert payload["input_ids"]==expected
        assert payload["loss_mask"]==[0]*len(expanded)+[1]*(len(expected)-len(expanded))
        ids = torch.tensor([expected])
        coords, delta = Qwen3VLModel.get_rope_index(reference,ids,(ids==marker).int(),hf.image_grid_thw)
        assert torch.equal(coords[:,0].T,torch.tensor(payload["position_ids"]))
        # Compare every eligible anchor, including all partial tail blocks, to
        # Qwen's independently computed autoregressive text continuation.
        anchors = torch.arange(len(expanded),len(expected)-1).view(1,-1)
        actual = OnlineDFlashModel._create_position_ids_from_context(SimpleNamespace(block_size=9),coords,anchors,len(expected))
        decode = (anchors.T + delta.item() + torch.arange(9)).reshape(1,1,-1).expand(3,1,-1)
        assert torch.equal(actual,decode)
        old_diff = tok.encode(g["raw"],add_special_tokens=False)!=g["response_token_ids"]
        fixed += int(old_diff)
        anchors_checked += anchors.numel()
        log.write(json.dumps(dict(id=row["id"],sha256=image_hash,length=len(expected),response_tokens=len(g["response_token_ids"]),corrected_tokenization=old_diff,anchors=anchors.numel()))+"\n")
        data.write(line)
        if (i+1)%250==0:
            print("Verified",i+1,flush=True)
assert fixed==34
staging.replace(out/"train.aligned.jsonl")
audit.replace(out/"records.jsonl")
summary = dict(rows=len(rows),different_retokenizations_fixed=fixed,valid_anchors_checked=anchors_checked,
               input_sha256=hashlib.sha256((out/"train.aligned.jsonl").read_bytes()).hexdigest())
(out/"verified.json").write_text(json.dumps(summary,indent=2))
print(summary)
