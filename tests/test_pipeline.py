import unittest
from collections import Counter
import torch
import train_authorship_attribution as t

class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=t.load_data()
    def test_splits_complete_balanced_and_disjoint(self):
        for domain in [False,True]:
            split=t.make_split(self.rows,domain)
            self.assertEqual({k:len(v) for k,v in split.items()}, {'train':320,'val':40,'test':40,'target':100} if domain else {'train':400,'val':50,'test':50})
            self.assertEqual(len(set(sum(split.values(),[]))),500)
            for name,ids in split.items():
                rows=[r for r in self.rows if r['prompt_id'] in ids]
                self.assertEqual(set(Counter(r['model_family'] for r in rows).values()),{len(ids)})
                if domain and name!='target':self.assertTrue(all(r['prompt_category'] not in t.TARGET_CATEGORIES for r in rows))
    def test_unicode_and_masking(self):
        for s in ['Γνώση','最近','생일']:
            self.assertIn(s.lower(),t.tokenize(s))
        self.assertEqual(t.tokenize('Qwen Gemini Llama'),['<model>']*3)
    def test_no_validation_vocabulary(self):
        v=t.build_vocab([['train','train']]);self.assertNotIn('validation_only',v)
    def test_input_identical_and_both_preserves_output_budget(self):
        for pid in range(1,501):
            rows=[r for r in self.rows if r['prompt_id']==pid]
            self.assertEqual(len({tuple(t.row_tokens(r,'input')) for r in rows}),1)
            for r in rows:
                both=t.row_tokens(r,'both');self.assertEqual(both[both.index('<sep>')+1:],t.row_tokens(r,'output'))
    def test_padding_invariance_and_gradients(self):
        torch.manual_seed(7)
        for cls in [t.TextCNN,t.LSTMClassifier]:
            model=cls(20).eval();x=torch.tensor([[2,3,4,5,6],[2,3,0,0,0]]);lengths=torch.tensor([5,2])
            a=model(x,lengths);b=model(torch.nn.functional.pad(x,(0,9)),lengths)
            torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
            loss=torch.nn.functional.cross_entropy(a,torch.tensor([0,1]));loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
    def test_snapshot_not_mutable(self):
        model=t.TextCNN(20);snap={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        key=next(iter(snap));old=snap[key].clone()
        with torch.no_grad():next(model.parameters()).add_(1)
        torch.testing.assert_close(old,snap[key])
if __name__=='__main__':unittest.main()
