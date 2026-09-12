"""Reproducible revision experiments, paired by independent random streams.

Run: python -m src.ede_sanity.revision_experiments --workers 4
Every finished seed is checkpointed. Validation seeds 100..109 choose lambda;
test seeds 1000..1019 are not used in that choice. Original seeds 0..19 are unused.
"""
import argparse, json, os, time, sys, platform
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import t, ttest_1samp
from .config import Config
from .simulate import SyntheticEnvironment
from .ranker import BaseRanker
from .revision_policy import rerank

OUT=Path('revision1/results')
DISCOUNT=1/np.log2(np.arange(2,12))

def run(job):
    seed=job['seed']; scenario=job.get('scenario','standard'); strategy=job['strategy']
    cfg=Config(seed=seed)
    env=SyntheticEnvironment(cfg)
    labels=env.relevance_labels.copy()
    if scenario=='natural': labels=(env.rel_true>=.78).astype(float)
    if scenario=='navigational':
        labels[:]=0; labels[np.arange(cfg.n_queries),env.rel_true.argmax(axis=1)]=1
    q_rng=np.random.RandomState(seed+20000)
    c_rng=np.random.RandomState(seed+40000)
    p_rng=np.random.RandomState(seed+60000)
    nq=1000 if scenario=='sparse' else cfg.n_queries
    if nq!=cfg.n_queries:
        cfg.n_queries=nq; env=SyntheticEnvironment(cfg); labels=env.relevance_labels.copy()
    query_probs=None
    if scenario=='heterogeneous':
        query_probs=1/(np.arange(nq)+1.)**1.2; query_probs/=query_probs.sum()
    queries=q_rng.choice(nq,cfg.T,p=query_probs)
    uniforms=c_rng.uniform(size=(cfg.T,10)); stop=c_rng.uniform(size=(cfg.T,10))
    noise=c_rng.uniform(size=cfg.T); random_pos=c_rng.randint(10,size=cfg.T)
    ranker=BaseRanker(env.rel_true,seed=seed)
    docs=np.arange(cfg.n_docs); old=docs[:cfg.n_docs-cfg.n_new_docs]
    scores=np.array([ranker.score(q,docs,env.new_docs_mask) for q in range(nq)])
    I=np.zeros(cfg.n_docs); C=np.zeros(cfg.n_docs); observed_I=I.copy(); observed_C=C.copy()
    first=np.full(cfg.n_docs,cfg.T); query_first=first.copy()
    relevant_new=env.new_docs_mask & labels.any(axis=0)
    idcg=np.array([DISCOUNT[:min(10,int(row.sum()))].sum() for row in labels])
    sums=np.zeros(11); perquery=[[] for _ in range(nq)]; pending={}
    promoted_rel=[]; promoted_binary=[]; baseline_new_rel=[]
    prefix_violations=0; new_exposure=0.; total_exposure=0.
    for step,q in enumerate(queries):
        if step in pending:
            shown,clicked=pending.pop(step); observed_I[shown]+=1; observed_C[shown]+=clicked
        candidates=old if step<cfg.T0 else docs
        penalty=job['penalty']
        adjusted=scores[q,candidates]-penalty*env.new_docs_mask[candidates]
        base=candidates[np.argsort(-adjusted,kind='stable')[:10]]
        m=job.get('m',2)
        ranking=rerank(scores[q,candidates],candidates,observed_I,observed_C,env.new_docs_mask,
            penalty=penalty,lam=job.get('lam',.4),eta=job.get('eta',.65),m=m,
            alpha=job.get('alpha',.5),strategy=strategy,rng=p_rng)
        prefix_violations+=int(not np.array_equal(ranking[:m],base[:m]))
        rel=env.rel_true[q,ranking]
        b=.8 if scenario=='strong-bias' else .35
        if scenario=='heterogeneous': b=.8 if q%2 else .15
        probs=expit(3*(rel-.5)-b*np.arange(1,11))
        if scenario=='navigational': probs=expit(5*(labels[q,ranking]-.5)-b*np.arange(1,11))
        clicked=(uniforms[step]<probs).astype(int)
        if scenario=='cascade':
            for pos in range(10):
                if clicked[pos] and stop[step,pos]<.8:
                    clicked[pos+1:]=0; break
        if scenario=='noisy' and noise[step]<.1:
            clicked[:]=0; clicked[random_pos[step]]=1
        I[ranking]+=1; C[ranking]+=clicked
        if scenario=='delayed': pending[step+50]=(ranking.copy(),clicked.copy())
        else: observed_I[ranking]+=1; observed_C[ranking]+=clicked
        hit=ranking[clicked.astype(bool)]; first[hit]=np.minimum(first[hit],step)
        hit=ranking[(clicked*labels[q,ranking]).astype(bool)]
        query_first[hit]=np.minimum(query_first[hit],step)
        ndcg=float(labels[q,ranking]@DISCOUNT/idcg[q]) if idcg[q] else 0.
        base_ndcg=float(labels[q,base]@DISCOUNT/idcg[q]) if idcg[q] else 0.
        sums[0]+=ndcg; sums[1]+=clicked.mean()
        if step>=cfg.T0:
            # RBO extrapolation at depth 10, rho=.9; replacement-aware overlap.
            overlaps=np.array([len(set(base[:d])&set(ranking[:d]))/d for d in range(1,11)])
            rbo=.1*np.sum(overlaps*.9**np.arange(10))+overlaps[-1]*.9**10
            pos={d:i for i,d in enumerate(ranking)}; shared=[d for d in base if d in pos]
            discord=sum(pos[shared[a]]>pos[shared[b]] for a in range(len(shared)) for b in range(a+1,len(shared)))
            kendall=discord/max(1,len(shared)*(len(shared)-1)/2)
            sums[2]+=rbo; sums[3]+=kendall; sums[4]+=ndcg-base_ndcg
            sums[5]+=int(ndcg-base_ndcg<-.01); sums[6]+=labels[q,ranking].mean()
            sums[7]+=env.new_docs_mask[ranking].mean(); sums[8]+=ndcg; sums[9]+=base_ndcg
            sums[10]+=np.mean(ranking!=base)
            perquery[q].append(ndcg-base_ndcg)
            promoted=[d for d in ranking if env.new_docs_mask[d] and d not in base]
            promoted_rel.extend(env.rel_true[q,promoted]); promoted_binary.extend(labels[q,promoted])
            baseline_new_rel.extend(env.rel_true[q,base[env.new_docs_mask[base]]])
            new_exposure+=float(env.new_docs_mask[ranking]@DISCOUNT); total_exposure+=DISCOUNT.sum()
    npost=cfg.T-cfg.T0; qmeans=[np.mean(x) for x in perquery if x]
    result=dict(job)
    result.update(ndcg=sums[0]/cfg.T,ctr=sums[1]/cfg.T,
        coverage=float(np.mean(first[relevant_new]<cfg.T)),
        query_relevant_coverage=float(np.mean(query_first[relevant_new]<cfg.T)),
        ttf=float(np.mean(first[relevant_new]-cfg.T0)), unique=float((I[env.new_docs_mask]>0).sum()),
        rbo=sums[2]/npost,kendall=sums[3]/npost,post_delta_ndcg=sums[4]/npost,
        loss_fraction=sums[5]/npost,top10_relevant_fraction=sums[6]/npost,
        new_fraction=sums[7]/npost,post_ndcg=sums[8]/npost,post_base_ndcg=sums[9]/npost,
        positions_changed_fraction=sums[10]/npost,worst_query_delta=float(min(qmeans)),
        query_loss_fraction=float(np.mean(np.array(qmeans)<-.01)),
        promoted_relevance=float(np.mean(promoted_rel)) if promoted_rel else None,
        promoted_relevant_fraction=float(np.mean(promoted_binary)) if promoted_binary else None,
        promoted_n=len(promoted_rel),baseline_new_relevance=float(np.mean(baseline_new_rel)) if baseline_new_rel else None,
        new_discounted_exposure_share=new_exposure/total_exposure,
        relevant_count_mean=float(labels.sum(axis=1).mean()),relevant_count_min=int(labels.sum(axis=1).min()),
        relevant_count_max=int(labels.sum(axis=1).max()),prefix_violations=prefix_violations)
    return result

def evaluate(jobs,workers):
    OUT.mkdir(parents=True,exist_ok=True)
    import hashlib
    todo=[]; results=[]
    for job in jobs:
        key=hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()[:20]
        path=OUT/'runs'/f'{key}.json'; path.parent.mkdir(exist_ok=True)
        if path.exists(): results.append(json.loads(path.read_text()))
        else: todo.append((job,path))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(run,job):path for job,path in todo}
        for i,future in enumerate(as_completed(futures),1):
            result=future.result(); futures[future].write_text(json.dumps(result)); results.append(result)
            if i%20==0 or i==len(todo): print(f'{i}/{len(todo)} new runs complete',flush=True)
    return pd.DataFrame(results)

def summarize(df):
    keys=['suite','scenario','penalty','strategy','lam','eta','m','alpha']
    metrics=[c for c in df if c not in keys+['seed']]
    rows=[]
    for key,group in df.groupby(keys,dropna=False,sort=False):
        row=dict(zip(keys,key)); row['n']=len(group)
        base=df[(df.suite==row['suite'])&(df.scenario==row['scenario'])&(df.penalty==row['penalty'])&(df.strategy=='Baseline')].drop_duplicates('seed').set_index('seed')
        for metric in metrics:
            vals=group[metric].dropna().to_numpy(float)
            if not len(vals): continue
            row[metric+'_n']=len(vals)
            mean=vals.mean(); sd=vals.std(ddof=1) if len(vals)>1 else float('nan')
            half=t.ppf(.975,max(1,len(vals)-1))*sd/np.sqrt(len(vals))
            row[metric+'_mean']=mean; row[metric+'_sd']=sd; row[metric+'_ci_low']=mean-half; row[metric+'_ci_high']=mean+half
            if len(base) and metric in ['coverage','ndcg','ttf','unique','ctr','query_relevant_coverage']:
                paired=group.set_index('seed')[metric]-base[metric]
                vals=paired.dropna().to_numpy(float); mean=vals.mean(); sd=vals.std(ddof=1)
                half=t.ppf(.975,len(vals)-1)*sd/np.sqrt(len(vals))
                row['delta_'+metric]=mean; row['delta_'+metric+'_ci_low']=mean-half; row['delta_'+metric+'_ci_high']=mean+half
                row['p_'+metric]=float(ttest_1samp(vals,0).pvalue) if sd>0 else (1. if mean==0 else 0.)
        rows.append(row)
    out=pd.DataFrame(rows)
    # Holm correction over all nonbaseline test comparisons and two primary metrics.
    family=[]
    for i,row in out.iterrows():
        if row.strategy!='Baseline' and row.suite!='validation':
            for metric in ['coverage','ndcg']:
                if pd.notna(row.get('p_'+metric)): family.append((row['p_'+metric],i,metric))
    family.sort(); previous=0
    for rank,(p,i,metric) in enumerate(family):
        previous=max(previous,min(1,p*(len(family)-rank))); out.loc[i,'holm_'+metric]=previous
    return out

def job(suite,seed,penalty,strategy,lam=.4,eta=.65,m=2,alpha=.5,scenario='standard'):
    return dict(suite=suite,seed=seed,penalty=penalty,strategy=strategy,lam=lam,eta=eta,m=m,alpha=alpha,scenario=scenario)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--workers',type=int,default=4); args=parser.parse_args()
    validation=[job('validation',s,p,st,lam=l) for s in range(100,110) for p in [.15,.3] for st,l in [('Baseline',0),('EDE',.1),('EDE',.2),('EDE',.3),('EDE',.4)]]
    v=evaluate(validation,args.workers); v.to_csv(OUT/'validation_per_seed.csv',index=False)
    vs=summarize(v); vs.to_csv(OUT/'validation_summary.csv',index=False)
    eligible=[]
    for lam,g in vs[vs.strategy=='EDE'].groupby('lam'):
        if (g.delta_ndcg_ci_low>=-.01).all(): eligible.append((g.delta_coverage.mean(),lam))
    chosen=max(eligible)[1] if eligible else 0.
    (OUT/'protocol.json').write_text(json.dumps(dict(validation_seeds=list(range(100,110)),test_seeds=list(range(1000,1020)),
        selection='Highest mean coverage gain with paired 95% NDCG CI lower bound >= -0.01 at each validation penalty; fallback lambda=0',
        selected_lambda=chosen,python=sys.version,platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__),indent=2))
    print('Selected lambda:',chosen,flush=True)
    jobs=[]
    for s in range(1000,1020):
        for p in [0.,.1,.15,.2,.25,.3,.4,.6,.8]:
            for st in ['Baseline','EDE','Novelty-only','Entropy-only','Recency','UCB','Thompson','Local-swap','Exposure-quota','Random','Epsilon']:
                jobs.append(job('difficulty',s,p,st,lam=0 if st=='Baseline' else chosen))
        for p in [.15,.3,.4]:
            jobs.append(job('sensitivity',s,p,'Baseline',lam=0))
            for eta in [.35,.5,.65,.8,1.]:
                for m in [1,2,3]: jobs.append(job('sensitivity',s,p,'EDE',lam=chosen,eta=eta,m=m))
        for p in [.15,.3]:
            for st in ['Baseline','EDE','Unrefined','No-decay','No-gate','No-normalization']:
                jobs.append(job('ablation',s,p,st,lam=0 if st=='Baseline' else chosen))
            jobs.append(job('ablation',s,p,'EDE',lam=chosen,alpha=0))
            jobs.append(job('ablation',s,p,'EDE',lam=chosen,m=0))
        for scenario in ['cascade','noisy','strong-bias','heterogeneous','navigational','sparse','delayed','natural']:
            for st in ['Baseline','EDE','Novelty-only','Entropy-only']:
                jobs.append(job('stress',s,.3,st,lam=0 if st=='Baseline' else chosen,scenario=scenario))
    df=evaluate(jobs,args.workers); df.to_csv(OUT/'test_per_seed.csv',index=False)
    summarize(df).to_csv(OUT/'test_summary.csv',index=False)
    assert df.prefix_violations.max()==0
    print('Finished all revision simulations',flush=True)

if __name__=='__main__': main()
