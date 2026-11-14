"""Small, CPU-first transfer workloads. None reproduces RecAD/MultiVerS training."""
import csv
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from .ann import Workload, normalize, exact
from .audit import sha256


def synthetic(n=6000, queries=48, seed=11, mode='clean'):
    rng=np.random.default_rng(seed)
    x=normalize(rng.normal(size=(n,64)))
    q=normalize(rng.normal(size=(queries,64)))
    groups=np.arange(n)
    if mode=='copies':
        # Copy roots closest to each query; shared source is explicitly observable
        # here, not inferred from an attacker label.
        roots=np.unique(np.argmax(q @ x.T,axis=1))
        copies=np.repeat(roots,40)
        x=np.vstack((x,x[copies]));groups=np.r_[groups,copies]
    allowed=np.ones((queries,len(x)),dtype=bool)
    if mode=='sparse_filter':
        allowed=rng.random(allowed.shape)<.01
    elif mode=='adverse_filter':
        # Query-dependent exclusion of the closest 90%; intentionally unfavorable.
        similarities=q @ x.T
        allowed=similarities < np.quantile(similarities,.1,axis=1)[:,None]
    elif mode not in ('clean','copies'):
        raise ValueError('Unknown synthetic mode')
    return Workload('synthetic_'+mode,x,q,groups,allowed,metadata={
        'synthetic':True,'seed':seed,'not_a_reputation_attack_benchmark':True,
        'group_semantics':'declared copy roots; no claim of identity inference'})


def scifact(directory, queries=96, dimensions=96, seed=11):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from threadpoolctl import threadpool_limits
    d=Path(directory)
    cp=d/'data/corpus_candidates.jsonl';qp=d/'data/claims.jsonl'
    docs=[json.loads(s) for s in cp.read_text().splitlines()]
    claims=[json.loads(s) for s in qp.read_text().splitlines()]
    # Input order fixed by release; smoke first N, full final eval uses all 279.
    claims=claims[:queries]
    texts=[r['title']+' '+' '.join(r['abstract']) for r in docs]
    vectorizer=TfidfVectorizer(max_features=30000,min_df=2,sublinear_tf=True)
    a=vectorizer.fit_transform(texts)  # no claims or gold used in fitting
    model=TruncatedSVD(n_components=min(dimensions,min(a.shape)-1),random_state=seed)
    with threadpool_limits(limits=1):
        x=normalize(model.fit_transform(a))
        q=normalize(model.transform(vectorizer.transform([c['claim'] for c in claims])))
    lookup={str(r['doc_id']):i for i,r in enumerate(docs)}
    gold=[]
    for c in claims:
        # Missing pool items stay in recall denominator using out-of-range IDs.
        relevant={key:lookup.get(key,len(docs)+int(key)) for key in c['evidence']}
        gold.append({'evidence':list(relevant.values()),
                     'support':[relevant[key] for key,e in c['evidence'].items() if e['label']=='SUPPORT'],
                     'contradict':[relevant[key] for key,e in c['evidence'].items() if e['label']=='CONTRADICT']})
    total=sum(len(g['evidence']) for g in gold)
    covered=sum(sum(i<len(docs) for i in g['evidence']) for g in gold)
    return Workload('scifact_pool_lsa',x,q,np.arange(len(x)),np.ones((len(q),len(x)),bool),gold=gold,
        metadata={'corpus_sha256':sha256(cp),'claims_sha256':sha256(qp),
                  'representation':'corpus-only TF-IDF + truncated SVD; cosine',
                  'claim_ids':[c['id'] for c in claims], 'gold_pairs':total,
                  'candidate_ceiling':covered/total if total else None,
                  'zero_query_vectors':int((np.linalg.norm(q,axis=1)==0).sum()),
                  'scope':'pooled candidates, NOT full 500k retrieval; NOT MultiVerS comparison'})


def recad(archive, queries=96, dimensions=64, seed=11):
    from scipy.sparse import coo_matrix
    from sklearn.decomposition import TruncatedSVD
    from threadpoolctl import threadpool_limits
    with zipfile.ZipFile(archive) as z:
        def rows(split):
            with z.open(f'ml1m/ml1m_{split}.csv') as f:
                return [(int(r['user_id']),int(r['item_id']),float(r['rating']))
                        for r in csv.DictReader(io.TextIOWrapper(f,encoding='utf-8'))]
        train=rows('train');test=rows('test')
    users=sorted({u for u,i,r in train});items=sorted({i for u,i,r in train})
    um={u:i for i,u in enumerate(users)};im={v:i for i,v in enumerate(items)}
    positives=[(um[u],im[i]) for u,i,r in train if r>=4]
    a=coo_matrix((np.ones(len(positives)),tuple(zip(*positives))),shape=(len(users),len(items))).tocsr()
    model=TruncatedSVD(n_components=min(dimensions,min(a.shape)-1),random_state=seed)
    with threadpool_limits(limits=1):user_factors=model.fit_transform(a)
    x=np.ascontiguousarray(model.components_.T,dtype=np.float32)
    eligible=sorted({u for u,i,r in test if r>=4 and u in um})
    rng=np.random.default_rng(seed)
    selected=sorted(rng.choice(eligible,min(queries,len(eligible)),replace=False).tolist())
    q=np.ascontiguousarray(user_factors[[um[u] for u in selected]],dtype=np.float32)
    pos={u:j for j,u in enumerate(selected)}
    allowed=np.ones((len(q),len(x)),bool)
    for u,i,r in train:
        if u in pos:allowed[pos[u],im[i]]=False  # all seen, not only liked
    gold=[{'positive':[]} for _ in selected]
    for u,i,r in test:
        if u in pos and r>=4 and (i not in im or allowed[pos[u],im[i]]):
            gold[pos[u]]['positive'].append(im.get(i,len(items)+i))
    return Workload('recad_ml1m_train_svd',x,q,np.arange(len(x)),allowed,gold=gold,metadata={
        'archive_sha256':sha256(archive),'representation':'binary train ratings>=4 SVD; UNNORMALIZED inner product',
        'user_ids':selected,'train_only_fitting':True,
        'scope':'retrieval transfer; NOT RecAD attack, victim model, or Table 2 reproduction'})


def read_fvecs(path, limit=None):
    raw=np.memmap(path,dtype='<i4',mode='r')
    if not len(raw) or raw[0]<=0:raise ValueError('Invalid fvecs dimension')
    d=int(raw[0]);width=d+1
    if len(raw)%width:raise ValueError('Truncated fvecs')
    rows=raw.reshape(-1,width)
    if np.any(rows[:,0]!=d):raise ValueError('Variable fvecs dimension')
    return np.ascontiguousarray(rows[:limit,1:].view('<f4'))


def read_ivecs(path, limit=None):
    values=np.fromfile(path,dtype='<i4');out=[];offset=0
    while offset<len(values) and (limit is None or len(out)<limit):
        count=int(values[offset]);offset+=1
        if count<0 or offset+count>len(values):raise ValueError('Invalid ivecs record')
        out.append(values[offset:offset+count].copy());offset+=count
    return out


def fanns(directory, queries=64, k=10, query_offset=0):
    d=Path(directory)
    x=read_fvecs(d/'database_vectors.fvecs')
    q=read_fvecs(d/'query_vectors.fvecs',queries+query_offset)[query_offset:]
    if len(q)!=queries:raise ValueError('Requested query slice exceeds release')
    attrs=[json.loads(s)['number_of_sub_categories'] for s in (d/'database_attributes.jsonl').read_text().splitlines()]
    labels=[json.loads(s)['label'] for s in (d/'em_query_attributes.jsonl').read_text().splitlines()][query_offset:query_offset+queries]
    if len(attrs)!=len(x) or len(labels)!=len(q):raise ValueError('FANNS attribute count mismatch')
    allowed=np.asarray(labels)[:,None]==np.asarray(attrs)[None,:]
    scale={1000:'small',100000:'medium'}.get(len(x),'custom')
    w=Workload('arxiv_fanns_'+scale+'_em',x,q,np.arange(len(x)),allowed,metric='l2',metadata={
        'scope':('official small DEBUG set' if scale=='small' else 'original vectors/EM; selected query subset, not paper timing reproduction'),
        'query_offset':query_offset,'query_indices':list(range(query_offset,query_offset+queries)),
        'filter':'number_of_sub_categories == query label',
        'database_sha256':sha256(d/'database_vectors.fvecs'),
        'query_sha256':sha256(d/'query_vectors.fvecs')}).validate()
    supplied=read_ivecs(d/'ground_truth_em.ivecs',queries+query_offset)[query_offset:]
    if len(supplied)!=len(q):raise ValueError('Missing ground truth rows')
    # Supplied rows may have <100 entries; use min(k, eligible) denominator.
    agreements=[]
    for i,gt in enumerate(supplied):
        reference=exact(w,i,k);target=set(map(int,gt[:k]));got=set(map(int,reference))
        agreements.append(len(target&got)/len(target) if target else float(not got))
    w.metadata['supplied_ground_truth_recall']=float(np.mean(agreements))
    if min(agreements)<1.0:raise ValueError('Adapter does not reproduce supplied exact top-k; inspect ties/metric')
    return w
