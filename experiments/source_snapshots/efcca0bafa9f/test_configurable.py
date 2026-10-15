"""Tiny CPU architecture tests; no pretrained download or competition training."""
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import torch
from transformers import BertConfig, BertModel, RobertaConfig, RobertaModel, DistilBertConfig, DistilBertModel, DebertaV2Config, DebertaV2Model
import pipeline
import train_worker as worker
from pooling import SUPPORTED


def main():
    torch.set_num_threads(1)
    config=json.loads(Path(__file__).with_name('first_run.json').read_text())
    config.update(max_length=16,batch_size=2,eval_batch_size=2,gradient_accumulation=1,
        concat_last_n_layers=2,lstm_hidden_size=8,lstm_num_layers=1,weighted_layer_start=0,
        backbone_dropout=0.,head_dropout=0.,cpu_threads=1)
    tiny=dict(vocab_size=32,hidden_size=16,num_hidden_layers=2,num_attention_heads=2,
              intermediate_size=32,max_position_embeddings=32)
    backbones=[BertModel(BertConfig(**tiny)),RobertaModel(RobertaConfig(**tiny,pad_token_id=0)),
        DebertaV2Model(DebertaV2Config(**tiny,relative_attention=True,pos_att_type=['p2c','c2p'])),
        DistilBertModel(DistilBertConfig(vocab_size=32,dim=16,hidden_dim=32,n_layers=2,n_heads=2,max_position_embeddings=32))]
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)
        for index,backbone in enumerate(backbones):
            folder=root/str(index);backbone.save_pretrained(folder)
            inputs={'input_ids':torch.tensor([[2,5,6,3],[2,7,3,0]]),'attention_mask':torch.tensor([[1,1,1,1],[1,1,1,0]])}
            padded={k:torch.nn.functional.pad(v,(0,4)) for k,v in inputs.items()}
            for method in SUPPORTED:
                c=dict(config,pooling=method)
                worker.set_seed(42);model=worker.GrammarModel(c,folder).eval()
                prediction,_=model(inputs);extended,_=model(padded)
                torch.testing.assert_close(prediction,extended,atol=2e-5,rtol=2e-5)
                assert torch.isfinite(prediction).all() and ((prediction>=0)&(prediction<=5)).all()
                ((prediction-torch.tensor([1.,4.]))**2).mean().backward()
                assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                groups=worker.parameter_groups(model,c)
                ids=[id(p) for g in groups for p in g['params']]
                assert len(ids)==len(set(ids))==sum(p.requires_grad for p in model.parameters())
                reloaded=worker.GrammarModel(c,folder,random_backbone=True).eval()
                reloaded.load_state_dict(worker.trainable_state(model),strict=True)
                torch.testing.assert_close(model(inputs)[0],reloaded(inputs)[0])
            frozen=worker.GrammarModel(dict(config,pooling='mean',freeze_top_n_layers=1),folder)
            assert all(not p.requires_grad for p in worker.backbone_blocks(frozen.backbone)[-1].parameters())
            assert all(p.requires_grad for p in frozen.head.parameters())
            reset=worker.GrammarModel(dict(config,pooling='mean',re_init_top_n_layers=1),folder)
            assert reset.layer_report['reinitialized_indices']==[1]
        data=root/'data';data.mkdir()
        train=pd.DataFrame([{'filename':f'audio_{i}','text':'text','label':float(i%6)} for i in range(48)])
        train.to_csv(data/'train.csv',index=False)
        pd.DataFrame([{'filename':'test','text':'text','label':-1}]).to_csv(data/'test.csv',index=False)
        pd.DataFrame({'zero_filename':['audio_0'],'positive_filename':['audio_5']}).to_csv(data/'source_pairs.csv',index=False)
        import sweep_utils
        pipeline.write_json(data/'manifest.json',{'files':{p.name:sweep_utils.file_sha256(p) for p in data.glob('*.csv')}})
        for n in [2,4]:
            out=root/f'folds_{n}';out.mkdir()
            c=dict(config,n_folds=n,fold_seed=13)
            pipeline.assign_folds(data,out,c)
            before=(out/'folds.csv').read_bytes();pipeline.assign_folds(data,out,c)
            assert before==(out/'folds.csv').read_bytes()
            folds=pd.read_csv(out/'folds.csv').set_index('filename')
            assert len(folds.fold.unique())==n and folds.loc['audio_0','fold']==folds.loc['audio_5','fold']
    print('PASS: all seven poolings x four encoder families; finite backward, padding, strict reload, layer controls, optimizer groups, deterministic 2/4 grouped folds.')


if __name__=='__main__':main()
