import numpy as np
from src.ede_sanity.revision_policy import rerank
from src.ede_real_replay.revision_replay import load_events,evaluate

def test_original_prefix_survives_novelty_gate():
    docs=np.arange(80); scores=np.linspace(1,0,80); I=np.full(80,1000.); I[30:]=0
    new=docs>=30
    for m in [0,1,2,3,10]:
        for lam in [0,.1,2]:
            result=rerank(scores,docs,I,np.zeros(80),new,m=m,lam=lam)
            expected=docs[np.argsort(-(scores-.3*new),kind='stable')[:10]]
            assert np.array_equal(result[:m],expected[:m])
            assert len(result)==len(set(result))==10
            if lam==0: assert np.array_equal(result,expected)

def test_unsmoothed_zero_counts_and_single_pool():
    result=rerank(np.arange(10.),np.arange(10),np.zeros(10),np.zeros(10),np.zeros(10,dtype=bool),alpha=0,m=9,top_L=10)
    assert len(set(result))==10

def test_every_comparator_prefix_and_determinism():
    docs=np.arange(70); scores=np.random.RandomState(3).normal(size=70)
    impressions=np.arange(70); clicks=impressions//4; new=docs>50
    base=docs[np.argsort(-(scores-.3*new),kind='stable')[:10]]
    strategies=['EDE','Novelty-only','Entropy-only','Recency','UCB','Thompson','Local-swap',
                'Exposure-quota','Random','Epsilon','Unrefined','No-decay','No-gate','No-normalization']
    for strategy in strategies:
        results=[rerank(scores,docs,impressions,clicks,new,strategy=strategy,rng=np.random.RandomState(5)) for _ in range(2)]
        assert np.array_equal(results[0],results[1])
        assert np.array_equal(results[0][:2],base[:2])
        assert len(set(results[0]))==10

def test_replay_does_not_use_future_clicks():
    key=('s','p'); pages={key:dict(session_id='s',serp_id='p',query_id='q',docs=list(range(10)),clicks=[0]*9+[1])}
    df=evaluate(pages,[('Q',key,None),('C',key,9)],lam=20)
    # Equal zero-exposure bonuses before the first query must leave order intact.
    assert df.exact_match.iloc[0]==1

def test_pages_and_clicks_do_not_merge(tmp_path):
    p=tmp_path/'train.txt'
    p.write_text('1\tM\t0\t1\n1\t0\tQ\t10\t100\tterm\t'+ '\t'.join(f'{i},1' for i in range(10))+'\n'
       +'1\t1\tC\t10\t9\n1\t2\tQ\t11\t101\tterm\t'+'\t'.join(f'{i},1' for i in range(10,20))+'\n'
       +'1\t3\tC\t11\t18\n')
    pages,events,stats=load_events(p)
    assert len(pages)==2 and pages[('1','10')]['docs']==list(range(10))
    assert pages[('1','10')]['clicks'][-1]==1
    df=evaluate(pages,events)
    assert df.clicks.tolist()==[1,1]
    assert (df.delta_ctr==0).all() and (df.set_match==1).all() and (df.prefix_violations==0).all()

def test_lambda_zero_identity_replay(tmp_path):
    pages={('s','p'):dict(session_id='s',serp_id='p',query_id='q',docs=list(range(10)),clicks=[1]+[0]*9)}
    df=evaluate(pages,[('Q',('s','p'),None),('C',('s','p'),0)],lam=0)
    assert df.exact_match.iloc[0]==1 and df.delta_click_ndcg.iloc[0]==0
