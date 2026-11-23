import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
from specforge.data.vlm_preprocessing import (
    ImageDataError, _conversation_messages, _exact_qwen_turn, build_vlm_prompt_payloads,
)


class Tokenizer:
    eos_token_id = 99

    def __len__(self):
        return 100

    def encode(self, text, **kwargs):
        return {"prefix": [7], "<|im_end|>\n": [99, 4]}[text]

    def decode(self, ids, **kwargs):
        return "".join({10: "a", 11: "b"}[i] for i in ids)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        return "prefix" if add_generation_prompt else "prefix" + messages[-1]["content"] + "<|im_end|>\n"

    def convert_tokens_to_ids(self, token):
        return 60


class AlignmentTest(unittest.TestCase):
    def test_explicit_image_on_later_turn_stays_in_place(self):
        text = {"type":"text","text":"inspect"}
        turns = [{"role":"user","content":"hello"},{"role":"assistant","content":"hi"},
                 {"role":"user","content":[text,{"type":"image"},text]}]
        self.assertEqual(turns,_conversation_messages(turns,with_image=True,source="case"))

    def test_legacy_image_default(self):
        turns=[{"from":"human","value":"inspect"}]
        parts=_conversation_messages(turns,with_image=True,source="case")[0]["content"]
        self.assertEqual([p["type"] for p in parts],["image","text"])

    def test_extra_image_marker_rejected(self):
        turn={"role":"user","content":[{"type":"image"},{"type":"image"}]}
        with self.assertRaises(ImageDataError):
            _conversation_messages([turn],with_image=True,source="case")

    def test_exact_labels_include_regenerated_ids_and_full_delimiter(self):
        row={"generation_prompt_ids":[7],"response_token_ids":[10,11],"conversations":[
            {"role":"user","content":"inspect"},{"role":"assistant","content":"ab"}]}
        self.assertEqual(_exact_qwen_turn(row,Tokenizer(),False,"case"),([7,10,11,99,4],[0,1,1,1,1]))
        with self.assertRaises(ValueError):
            _exact_qwen_turn(dict(row,response_token_ids=[11,10]),Tokenizer(),False,"case")

    def test_wrong_prefix_aborts_public_builder(self):
        row={"generation_prompt_ids":[8],"response_token_ids":[10,11],"conversations":[
            {"role":"user","content":"inspect"},{"role":"assistant","content":"ab"}]}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"row.jsonl"
            path.write_text(json.dumps(row)+"\n")
            with self.assertRaisesRegex(ValueError,"prompt IDs"):
                build_vlm_prompt_payloads(str(path),Tokenizer(),SimpleNamespace(image_processor=None),
                                          chat_template="qwen",max_length=100)

    def test_mrope_tail_is_independent_of_padding(self):
        ctx=torch.tensor([[[20,21,22]],[[18,19,20]],[[19,20,21]]])
        anchors=torch.tensor([[1]])
        fn=OnlineDFlashModel._create_position_ids_from_context
        expected=ctx[:,:,1:2]+torch.arange(9)
        for n in [0,3,12]:
            padded=torch.nn.functional.pad(ctx,(0,n))
            actual=fn(SimpleNamespace(block_size=9),padded,anchors,padded.size(-1))
            torch.testing.assert_close(actual,expected)
