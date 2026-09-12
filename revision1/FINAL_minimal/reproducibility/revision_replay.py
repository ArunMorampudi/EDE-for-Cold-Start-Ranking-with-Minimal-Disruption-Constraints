"""Page-level, document-aligned descriptive replay; not off-policy evaluation.

Use labeled training logs only. With ten logged candidates, set preservation
and click preservation are tautologies and are reported as such. No unseen
candidate outcomes, propensities, or cold-start labels are inferred.
"""
import argparse, gzip, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from ..ede_sanity.revision_policy import rerank


def load_events(path, max_sessions=200000):
    opener=gzip.open if str(path).endswith('.gz') else open
    pages={}; events=[]; sessions=set(); current=None
    stats=dict(lines=0,unmatched_clicks=0,excluded_test_pages=0,invalid_pages=0)
    with opener(path,'rt',encoding='utf-8') as stream:
        for line in stream:
            parts=line.rstrip().split('\t'); stats['lines']+=1
            if len(parts)<3: continue
            sid=parts[0]
            if sid not in sessions and len(sessions)>=max_sessions: break
            sessions.add(sid)
            code=parts[1] if parts[1]=='M' else parts[2]
            if code=='T': stats['excluded_test_pages']+=1; continue
            if code=='Q' and len(parts)>=7:
                key=(sid,parts[3])
                docs=[int(x.split(',')[0]) for x in parts[6:] if x]
                if len(docs)!=10 or len(set(docs))!=10 or key in pages:
                    stats['invalid_pages']+=1; continue
                pages[key]=dict(session_id=sid,serp_id=parts[3],query_id=parts[4],docs=docs,clicks=[0]*10)
                events.append(('Q',key,None))
            elif code=='C' and len(parts)>=5:
                key=(sid,parts[3]); doc=int(parts[4])
                if key not in pages or doc not in pages[key]['docs']:
                    stats['unmatched_clicks']+=1; continue
                pos=pages[key]['docs'].index(doc)
                if not pages[key]['clicks'][pos]:
                    pages[key]['clicks'][pos]=1; events.append(('C',key,doc))
    stats.update(raw_sessions=len(sessions),pages=len(pages))
    return pages,events,stats


def evaluate(pages,events,lam=.4,eta=.65,m=2):
    impressions={}; clicks={}; rankings={}
    for kind,key,doc in events:
        page=pages[key]
        if kind=='C': clicks[doc]=clicks.get(doc,0)+1; continue
        docs=page['docs']; I=np.array([impressions.get(d,0) for d in docs]); C=np.array([clicks.get(d,0) for d in docs])
        indices=rerank(np.linspace(1,0,10),np.arange(10),I,C,np.zeros(10,dtype=bool),
                       penalty=0,lam=lam,eta=eta,m=m,top_L=10)
        rankings[key]=[docs[i] for i in indices]
        # Update only the displayed logging-policy documents, not hypothetical output.
        for d in docs: impressions[d]=impressions.get(d,0)+1
    rows=[]; discount=1/np.log2(np.arange(2,12))
    for key,page in pages.items():
        base=page['docs']; rank=rankings[key]; c=np.array(page['clicks']); mapping=dict(zip(base,c))
        reordered=np.array([mapping[d] for d in rank]); idcg=sorted(c,reverse=True)@discount
        pos={d:i for i,d in enumerate(rank)}
        clicked=[d for d in base if mapping[d]]
        overlaps=np.array([len(set(base[:d])&set(rank[:d]))/d for d in range(1,11)])
        rows.append(dict(session_id=page['session_id'],serp_id=page['serp_id'],query_id=page['query_id'],
            exact_match=int(base==rank),set_match=int(set(base)==set(rank)),click_preserved=int(set(clicked)<=set(rank)),
            clicks=int(c.sum()),ctr=float(c.mean()),delta_ctr=float(reordered.mean()-c.mean()),
            click_ndcg=float(c@discount/idcg) if idcg else 0.,
            delta_click_ndcg=float((reordered-c)@discount/idcg) if idcg else 0.,
            positions_changed=sum(a!=b for a,b in zip(base,rank)),
            mean_abs_shift=float(np.mean([abs(pos[d]-i) for i,d in enumerate(base)])),
            clicked_shift_sum=sum(abs(pos[d]-base.index(d)) for d in clicked),
            clicked_moved=sum(pos[d]!=base.index(d) for d in clicked),
            prefix_violations=int(base[:m]!=rank[:m]),
            kendall=sum(pos[base[a]]>pos[base[b]] for a in range(10) for b in range(a+1,10))/45,
            rbo=float(.1*np.sum(overlaps*.9**np.arange(10))+overlaps[-1]*.9**10)))
    return pd.DataFrame(rows)


def summary(df,B=2000):
    rng=np.random.default_rng(20260911); rows=[]
    metrics=['ctr','click_ndcg','delta_ctr','delta_click_ndcg','positions_changed','mean_abs_shift','kendall','rbo','clicks']
    for criterion in ['all','exact_match','set_match','click_preserved']:
        for accepted in ([True] if criterion=='all' else [True,False]):
            bucket=df if criterion=='all' else df[df[criterion]==int(accepted)]
            row=dict(criterion=criterion,accepted=accepted,pages=len(bucket),total_pages=len(df),
                     acceptance_rate=len(bucket)/len(df),sessions=bucket.session_id.nunique())
            # Cluster bootstrap whole sessions: preserves dependence of pages within session.
            if len(bucket):
                grouped=bucket.groupby('session_id')[metrics].sum(); sizes=bucket.groupby('session_id').size().to_numpy()
                vals=grouped.to_numpy(); n=len(vals); means=[]
                for _ in range(B):
                    idx=rng.integers(n,size=n); means.append(vals[idx].sum(axis=0)/sizes[idx].sum())
                means=np.asarray(means)
                for j,metric in enumerate(metrics):
                    row[metric]=float(bucket[metric].mean()); row[metric+'_ci_low']=float(np.quantile(means[:,j],.025)); row[metric+'_ci_high']=float(np.quantile(means[:,j],.975))
                row['clicked_moved_fraction']=float(bucket.clicked_moved.sum()/max(1,bucket.clicks.sum()))
                row['mean_abs_shift_clicked']=float(bucket.clicked_shift_sum.sum()/max(1,bucket.clicks.sum()))
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--log',required=True); p.add_argument('--max-sessions',type=int,default=200000)
    p.add_argument('--lambda-value',type=float,default=.4); a=p.parse_args()
    out=Path('revision1/results/replay'); out.mkdir(parents=True,exist_ok=True)
    pages,events,stats=load_events(a.log,a.max_sessions); print(stats,flush=True)
    if not pages: raise ValueError('No eligible labeled ten-result pages')
    df=evaluate(pages,events,lam=a.lambda_value); df.to_csv(out/'per_page.csv',index=False)
    summary(df).to_csv(out/'summary.csv',index=False)
    stats.update(input_name=Path(a.log).name,lambda_value=a.lambda_value,bootstrap=2000,bootstrap_unit='session',bootstrap_seed=20260911,
                 limitations='Ten-candidate logged-support diagnostic only; set/click preservation structural; no cold-start labels or propensity weights')
    (out/'manifest.json').write_text(json.dumps(stats,indent=2))
    assert df.prefix_violations.max()==0 and df.delta_ctr.abs().max()==0
    print('Replay complete',flush=True)

if __name__=='__main__': main()
