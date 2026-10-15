"""Three explicitly hardcoded, leaderboard-informed blends; no label lookups."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/rmse_research_20261015'
DEST=Path.home()/'Downloads/shl_fixed_hurdle_submissions_20261015'


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rmse(a,b):return float(np.sqrt(np.mean((a-b)**2)))


def main():
    inputs=np.load(OUT/'inputs.npz')
    y,folds,groups,names,tnames=[inputs[k] for k in ['y','folds','groups','names','test_names']]
    h=np.load(OUT/'hurdle_development_predictions.npz')['hurdle_regression']
    ht=np.load(OUT/'hurdle_test_predictions.npz')['hurdle_regression']
    prior=ROOT/'artifacts/all_metadata_ensembles_20261015'
    j=np.load(prior/'development_predictions.npz')['joint_quality']
    jt=np.load(prior/'full_predictions.npz')['joint_quality']
    winner=Path.home()/'Downloads/shl_rmse_research_20261015/02_hurdle_joint.csv'
    winner_hash=digest(winner)
    check=pd.read_csv(winner)
    assert check.filename.tolist()==tnames.tolist()
    np.testing.assert_allclose(check.label,.25*ht+.75*jt,atol=1e-12,rtol=0)
    winning_audit=json.loads((OUT/'final_research_audit.json').read_text())
    assert winner_hash==next(r['sha256'] for r in winning_audit['delivered'] if r['file']==winner.name)
    fit=json.loads((OUT/'hurdle_research_audit.json').read_text())['hurdle_regression']
    DEST.mkdir(parents=True,exist_ok=True)
    results=[]
    for i,w in enumerate([.35,.45,.60],1):
        oof=w*h+(1-w)*j;p=w*ht+(1-w)*jt
        filename=f'{i:02d}_hurdle{int(w*100):02d}_joint{int(round((1-w)*100)):02d}.csv'
        path=DEST/filename;pd.DataFrame(dict(filename=tnames,label=p)).to_csv(path,index=False)
        reread=pd.read_csv(path)
        assert reread.shape==(216,2) and reread.filename.is_unique and reread.filename.tolist()==tnames.tolist()
        assert np.isfinite(reread.label).all() and reread.label.between(0,5).all()
        np.testing.assert_allclose(reread.label,p,atol=1e-12,rtol=0)
        scores=dict(pooled_development_rmse=rmse(y,oof),mean_fold_rmse=float(np.mean([rmse(y[folds==f],oof[folds==f]) for f in sorted(set(folds))])),public_rmse=None)
        row=dict(file=filename,recipe=dict(hurdle_weight=w,prior_joint_weight=1-w),scores=scores,sha256=digest(path),rms_test_change_from_winner=rmse(check.label.to_numpy(),p))
        metadata=dict(schema_version=1,**row,
                      selection='Hardcoded35/45/60% hurdle weights, chosen after25% hurdle won publicly at0.3158. This is leaderboard-informed exploration, not an independently validated improvement.',
                      reference_winner=dict(file=str(winner),public_rmse=.3158,sha256=winner_hash),
                      validation='Component OOF predictions are grouped cross-fitted meta-level predictions over frozen base OOF features. Weights are manual and were not fitted to held-out labels.',
                      limitations=['No guaranteed gain; no public/private score exists yet.','Public-feedback-based recipe selection can overfit the public leaderboard.','Existing base OOF dependence, repeated validation reuse and mixed test-inference distributions remain.','Training labels are used only to report development error; predictions are fixed-weight combinations, without row-ID hardcoding.'],
                      learned_hurdle_fits=fit,prior_joint_audit_sha256=digest(prior/'optimization_audit.json'),
                      oof_predictions=[dict(filename=str(n),true_score=float(a),predicted_score=float(b),fold=int(f),source_group=str(g)) for n,a,b,f,g in zip(names,y,oof,folds,groups)],
                      submission_predictions=[dict(filename=str(n),predicted_score=float(v)) for n,v in zip(tnames,p)],submission_csv_text=path.read_text(),reproduction_source=Path(__file__).read_text())
        meta=DEST/f'{i:02d}_run_metadata_{scores["pooled_development_rmse"]:.6f}.txt'
        meta.write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n');row['metadata_file']=meta.name
        back=json.loads(meta.read_text());assert back['submission_csv_text']==path.read_text()
        assert abs(rmse(np.array([r['true_score'] for r in back['oof_predictions']]),np.array([r['predicted_score'] for r in back['oof_predictions']]))-scores['pooled_development_rmse'])<1e-12
        results.append(row)
        print(filename,json.dumps(scores),flush=True)
    assert len({r['sha256'] for r in results})==3 and digest(winner)==winner_hash
    pd.DataFrame([dict(file=r['file'],hurdle_weight=r['recipe']['hurdle_weight'],joint_weight=r['recipe']['prior_joint_weight'],development_rmse=r['scores']['pooled_development_rmse'],public_rmse='') for r in results]).to_csv(DEST/'rmse_scores.csv',index=False)
    (OUT/'fixed_weight_followup_audit.json').write_text(json.dumps(dict(results=results,winner_sha256=winner_hash),indent=2)+'\n')
    print('DELIVERY',DEST,flush=True)


if __name__=='__main__':main()
