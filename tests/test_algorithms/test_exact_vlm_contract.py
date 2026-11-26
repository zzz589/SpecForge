import base64
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
import torch
from specforge.data.exact_vlm import ExactJournal, align_response_scores, check_response_scores, score_digest


class TestExactVlmContract(unittest.TestCase):
    def test_predictors_with_repeated_labels_and_different_prefix_lengths(self):
        end=[151645,198]
        a,b=align_response_scores([10,20,7,7]+end,[20,7,7]+end,[7,7],2,1,torch.arange(6),torch.arange(5))
        self.assertEqual(a.tolist(),[1,2])
        self.assertEqual(b.tolist(),[0,1])
        with self.assertRaises(ValueError):
            check_response_scores(a.flip([0]),torch.tensor([1,2]))
        with self.assertRaises(ValueError):
            align_response_scores([10,7,7,7]+end,[20,7,7]+end,[7,7],1,1,torch.arange(6),torch.arange(5))

    def test_reference_binds_image_and_score_order(self):
        image=b"image bytes"
        sha=hashlib.sha256(image).hexdigest()
        scores=[0.25,0.5]
        ids=[17,17]
        saved=dict(id="sample",prompt="read",image_sha256=sha,response_token_ids=ids,
                   visual_score_sha256=score_digest("sample",sha,2,ids,scores))
        record=dict(id="sample",generation_prompt_ids=[3,4],response_token_ids=ids,visual_scores=scores)
        payload={"payload":dict(image_data="data:image/png;base64,"+base64.b64encode(image).decode(),loss_mask=[0,0,1,1,1,1])}
        tokenizer=Mock()
        tokenizer.encode.return_value=[3,4]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"journal"
            path.write_text(json.dumps(saved)+"\n")
            journal=ExactJournal(path)
            with self.assertRaisesRegex(ValueError,"visual scores differ"):
                journal.validate(dict(record,visual_scores=scores[::-1]),payload,tokenizer)
            with self.assertRaisesRegex(ValueError,"image differs"):
                journal.validate(record,{"payload":dict(payload["payload"],image_data="data:,c3dhcA==")},tokenizer)
            with self.assertRaisesRegex(ValueError,"omitted"):
                journal.finish()
            journal.validate(record,payload,tokenizer)
            journal.finish()
            with self.assertRaisesRegex(ValueError,"duplicate"):
                journal.validate(record,payload,tokenizer)
