"""Read actual upstream artifacts, retaining exact task definitions."""
import csv
import hashlib
import io
import json
import math
import zipfile
from collections import Counter
from pathlib import Path
from .metrics import note_rates, pair_metrics


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def notes(path):
    rows=list(csv.DictReader(open(path,encoding='utf-8')))
    mapping={'pollution':'p_u_H','suppression':'p_h_U','infiltration':'p_H_u','waste':'p_U_h'}
    errors={x:0.0 for x in mapping}; cells=[]
    for row in rows:
        counts={k:float(row[k]) for k in ['n_H','n_U','n_h','n_u','n_uH','n_hU']}
        if counts['n_H']+counts['n_U'] != counts['n_h']+counts['n_u']:
            raise ValueError('Inconsistent counts')
        rates=note_rates(counts)
        for key,column in mapping.items():
            stored=float(row[column])
            if rates[key] is not None and math.isfinite(stored):
                errors[key]=max(errors[key],abs(rates[key]-stored))
        if row['condition']=='all' and float(row['bhvr_rate'])==1.0 and float(row['st_prop']) in [0,.05,.1,.2]:
            cells.append({'attacker_fraction':float(row['st_prop']),'attack_frequency':1.0,**rates})
    return {'task':'recompute authors stored synthetic summary; no scorer rerun',
            'sha256':sha256(path),'rows':len(rows),'max_abs_metric_errors':errors,
            'selected_cells':cells,'production_cn_claim':False}


def recad(path):
    splits={}; train_positive=0
    with zipfile.ZipFile(path) as z:
        for split in ['train','valid','test']:
            member=f'ml1m/ml1m_{split}.csv'
            pairs=set();users=set();items=set();rows=0; positives=0
            with z.open(member) as f:
                for r in csv.DictReader(io.TextIOWrapper(f,encoding='utf-8')):
                    u=int(r['user_id']); i=int(r['item_id']);rating=float(r['rating'])
                    if (u,i) in pairs:raise ValueError('Duplicate user-item pair inside split')
                    pairs.add((u,i));users.add(u);items.add(i);rows+=1;positives+=rating>=4
            splits[split]={'rows':rows,'positive_rating_ge4':positives,'users':len(users),'items':len(items),'pairs':pairs}
    overlaps={f'{a}_{b}':len(splits[a]['pairs']&splits[b]['pairs']) for a,b in [('train','valid'),('train','test'),('valid','test')]}
    return {'task':'processed dataset audit, not paper result reproduction','sha256':sha256(path),
            'splits':{k:{a:b for a,b in v.items() if a!='pairs'} for k,v in splits.items()},
            'pair_overlaps':overlaps,'paper_table2_exact_recipe_resolved':False}


def scifact(claims_path,predictions_path):
    try:import pandas as pd
    except ImportError as e:raise RuntimeError('Install the audit extra: pip install -e .[audit]') from e
    claims=[json.loads(x) for x in open(claims_path,encoding='utf-8')]
    if len({c['id'] for c in claims})!=len(claims):raise ValueError('Duplicate claim IDs')
    gold={(int(c['id']),int(d)):e['label'] for c in claims for d,e in c['evidence'].items()}
    df=pd.read_parquet(predictions_path)
    if df.duplicated(['model','claim_id','doc_id']).any():raise ValueError('Duplicate model-claim-document predictions')
    claim_ids={int(c['id']) for c in claims}
    if not set(df.claim_id)<=claim_ids:raise ValueError('Prediction for unknown claim')
    models={}
    for model,g in df.groupby('model'):
        predicted={(int(r.claim_id),int(r.doc_id)):r.predicted_label for r in g.itertuples()}
        models[model]=pair_metrics(gold,predicted)
    both=sum({'SUPPORT','CONTRADICT'} <= {e['label'] for e in c['evidence'].values()} for c in claims)
    return {'task':'re-evaluate released predictions, not retraining or new inference',
            'claims':len(claims),'gold_pairs':len(gold),'claims_with_both_stances':both,
            'sha256_claims':sha256(claims_path),'sha256_predictions':sha256(predictions_path),
            'models':models,'warning':'Pooled annotation; unjudged treated as NEI only to match official metric.'}
