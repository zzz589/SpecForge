"""CPU numerical audit with real weights; no optimizer or weight updates."""
import json
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
from specforge.modeling.draft.dflash import extract_context_feature
from specforge.modeling.draft.dflash2 import DFlash2DraftModel
from specforge.data.vlm_preprocessing import build_vlm_prompt_payloads

root = Path("/HOME/nsccgz_zgchen/nsccgz_zgchen_8/HDD_POOL/xj")
out = root/"experiments/pressure-original2272-input-alignment-audit-20261123"
old = root/"experiments/pressure-external-originalbytes-thinking128-fullattn-mrope-n2272-20261122"
row = json.loads(next((out/"train.aligned.jsonl").open()))
model_path = root/"models/Qwen3-VL-4B-Instruct"
torch.set_num_threads(4)
proc = AutoProcessor.from_pretrained(model_path)
tok = proc.tokenizer
single = out/"forward_row.jsonl"
single.write_text(json.dumps(row)+"\n")
payload = build_vlm_prompt_payloads(str(single),tok,proc,chat_template="qwen",max_length=6144)[0]["payload"]
ids = torch.tensor([payload["input_ids"]])
pos = torch.tensor([payload["position_ids"]])
loss_mask = torch.tensor([payload["loss_mask"]])
with Image.open(row["image"]) as im:
    vision = proc.image_processor(images=[im.convert("RGB")],return_tensors="pt")
target = Qwen3VLForConditionalGeneration.from_pretrained(model_path,dtype=torch.bfloat16,attn_implementation="sdpa").eval().requires_grad_(False)
modality = (ids==target.config.image_token_id).int()
with torch.no_grad():
    features = target.model(input_ids=ids,mm_token_type_ids=modality,**vision,output_hidden_states=True,use_cache=False)
    aux = extract_context_feature(features.hidden_states,[0,8,16,24,32]).float()
    refpos,delta = target.model.get_rope_index(ids,modality,vision.image_grid_thw)
    assert torch.equal(pos.permute(2,0,1),refpos)
draft = DFlash2DraftModel.from_pretrained(next((old/"output").glob("*step1704/exported-draft")),dtype=torch.float32,attn_implementation="sdpa").eval()
wrapper = OnlineDFlashModel(draft,target.lm_head.float(),target.get_input_embeddings().float(),151669,
                          block_size=9,attention_backend="sdpa",num_anchors=8,loss_type="dpace",
                          selector_loss_alpha=1.0,teacher_metrics=False)
start=int(loss_mask.nonzero()[0,1])
anchors=torch.tensor([[start,start+9,ids.shape[1]-2]])
keep=torch.ones_like(anchors,dtype=torch.bool)
with patch.object(wrapper,"_sample_anchor_positions",return_value=(anchors,keep)):
    with torch.no_grad():
        _,_,hidden=wrapper._forward_draft_blocks(ids,aux,loss_mask,pos)
        errors=[]
        for i,a in enumerate(anchors[0].tolist()):
            predicted_pos=(torch.arange(a,a+9)+int(delta.item())).view(1,1,-1).expand(3,1,-1)
            block=torch.full((1,9),151669)
            block[0,0]=ids[0,a]
            expected=draft(position_ids=torch.cat((refpos[:,:,:a],predicted_pos),2),
                           noise_embedding=wrapper.embed_tokens(block),target_hidden=aux[:,:a],
                           attention_mask=torch.ones(1,1,9,a+9,dtype=torch.bool))
            actual=hidden[:,9*i:9*(i+1)]
            logits=wrapper.lm_head(actual)
            ref_logits=wrapper.lm_head(expected)
            torch.testing.assert_close(logits,ref_logits,atol=3e-4,rtol=1e-4)
            assert torch.equal(logits.argmax(-1),ref_logits.argmax(-1))
            errors.append(float((logits-ref_logits).abs().max()))
    loss,_,_=wrapper(ids,aux,loss_mask,position_ids=pos,collect_detailed_metrics=False)
    loss.backward()
    grad={}
    for name in ("candidate_selector.hidden_projection.weight","layers.0.attention_conv.base_kernel"):
        g=dict(draft.named_parameters())[name].grad
        assert g is not None and g.isfinite().all() and g.abs().max()>0
        grad[name]=float(g.abs().max())
result=dict(sample=row["id"],target_hidden_shape=list(aux.shape),anchors=anchors.tolist(),max_abs_logits_error=errors,
            argmax_matches=27,loss=float(loss.detach()),max_abs_grad=grad)
(out/"real_forward.json").write_text(json.dumps(result,indent=2))
print(result)
