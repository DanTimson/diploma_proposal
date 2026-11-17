#!/usr/bin/env python3
"""B-001: fixed released predictions, exact-copy perturbation, observable text dedup."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from diploma_starters.group_screen import digest,write_json
from diploma_starters.metrics import pair_metrics

def select(candidates,k,dedup):
    result=[];seen=set()
    for c in sorted(candidates,key=lambda c:(-c['score'],c['id'])):
        if dedup and c['text_hash'] in seen:continue
        seen.add(c['text_hash']);result.append(c)
        if len(result)==k:break
    return result

def perturb(candidates,copies):
    if not candidates:return []
    ordered=sorted(candidates,key=lambda c:(-c['score'],c['id']))
    # Force copies immediately after the original at a score tie, independent of gold.
    result=[dict(c,id=f'{i:06d}:000') for i,c in enumerate(ordered)]
    result += [dict(result[0],id=f'000000:{j+1:03d}') for j in range(copies)]
    return result

def main():
    import pandas as pd
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-dir',required=True)
    p.add_argument('--out',required=True);p.add_argument('--model',default='multivers_20');a=p.parse_args()
    root=Path(a.data_dir);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    claims_path=root/'data/claims.jsonl';pred_path=root/'prediction/model_predictions.parqet';corpus_path=root/'data/corpus_candidates.jsonl'
    claims=[json.loads(s) for s in claims_path.read_text().splitlines()]
    gold={(int(c['id']),int(doc)):ev['label'] for c in claims for doc,ev in c['evidence'].items() if ev['label']!='NEI'}
    texts={}
    for line in corpus_path.read_text().splitlines():
        c=json.loads(line);text=json.dumps([c['title'],c['abstract']],ensure_ascii=False,separators=(',',':'))
        texts[int(c['doc_id'])]=hashlib.sha256(text.encode()).hexdigest()
    df=pd.read_parquet(pred_path)
    # The feature boundary deliberately excludes annotation ranks and provenance.
    df=df.loc[df.model==a.model,['claim_id','doc_id','predicted_label','p_support','p_contradict']]
    if df.empty or df.duplicated(['claim_id','doc_id']).any():raise ValueError('Missing model or duplicate predictions')
    candidates={int(c['id']):[] for c in claims};predicted={}
    for r in df.itertuples():
        cid,doc=int(r.claim_id),int(r.doc_id)
        score=max(float(r.p_support),float(r.p_contradict))
        if not 0<=score<=1:raise ValueError('Invalid confidence')
        candidates[cid].append(dict(id=str(doc),doc_id=doc,label=r.predicted_label,score=score,text_hash=texts[doc]))
        predicted[(cid,doc)]=r.predicted_label
    both={cid for cid in candidates if {'SUPPORT','CONTRADICT'} <= {lab for (c,d),lab in gold.items() if c==cid}}
    ceiling=sum(predicted.get(key)==label for key,label in gold.items())
    rows=[];selections=[]
    for copies in [0,1,5,20]:
      for k in [5,10,20]:
       for dedup in [False,True]:
        hits=set();document_hits=set();slots=0;unique_origins=0;both_count=0;both_documents=0;unjudged=0
        for cid,cs in candidates.items():
            got=select(perturb(cs,copies),k,dedup)
            correct={(cid,c['doc_id']) for c in got if c['label']!='NEI' and gold.get((cid,c['doc_id']))==c['label']}
            retrieved={(cid,c['doc_id']) for c in got if (cid,c['doc_id']) in gold}
            document_hits.update(retrieved)
            hits.update(correct);slots+=len(got);unique_origins+=len({c['doc_id'] for c in got})
            unjudged+=sum((cid,c['doc_id']) not in gold for c in got)
            both_count+=cid in both and {'SUPPORT','CONTRADICT'} <= {gold[key] for key in correct}
            both_documents+=cid in both and {'SUPPORT','CONTRADICT'} <= {gold[key] for key in retrieved}
            selections.append(dict(copies=copies,k=k,method='text_dedup' if dedup else 'score_topk',claim_id=cid,
                ids=[c['id'] for c in got],source_docs=[c['doc_id'] for c in got]))
        by_stance={label:dict(hits=sum(gold[key]==label for key in hits),total=sum(lab==label for lab in gold.values())) for label in ['SUPPORT','CONTRADICT']}
        rows.append(dict(copies=copies,k=k,method='text_dedup' if dedup else 'score_topk',signed_hits=len(hits),gold_pairs=len(gold),
            signed_recall=len(hits)/len(gold),document_recall=len(document_hits)/len(gold),document_hits=len(document_hits),
            both_stance_documents_covered=both_documents,unique_correct_per_slot=len(hits)/slots,selected_slots=slots,
            distinct_origin_slots=unique_origins,unjudged_slots=unjudged,both_stances_covered=both_count,
            both_stance_claims=len(both),by_stance=by_stance))
    write_json(out/'manifest.json',dict(config=vars(a),source_sha256=digest(__file__),
        inputs={p.name:digest(p) for p in [claims_path,pred_path,corpus_path]},
        scope='Fixed 50-candidate released predictions, not full-corpus retrieval or source-independence estimation.',
        injection='Exact copies of highest-confidence candidate per claim; no gold-driven targeting; deterministic, no seed sweep.',
        metric='Unique correct signed origins / all signed gold pairs. Precision-like unique_correct_per_slot penalizes wasted duplicate slots; not official pair precision. Unjudged does not mean scientifically false.',
        ceiling=dict(correct_signed_pairs=ceiling,gold_pairs=len(gold),recall=ceiling/len(gold),
            retrievable_gold_documents=sum(key in predicted for key in gold),
            both_signed=sum({'SUPPORT','CONTRADICT'} <= {label for key,label in gold.items() if key[0]==cid and predicted.get(key)==label} for cid in both),
            both_documents=sum({'SUPPORT','CONTRADICT'} <= {label for key,label in gold.items() if key[0]==cid and key in predicted} for cid in both)),
        official_saved_prediction_replay=pair_metrics(gold,predicted)))
    write_json(out/'summary.json',rows);write_json(out/'selections.json',selections)
    print('COMPLETE',out)
if __name__=='__main__':main()
