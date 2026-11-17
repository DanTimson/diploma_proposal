"""Prospective D-008 development gate; no automatic held-out evaluation."""
import argparse
import json
from pathlib import Path
import numpy as np
from .explore import identity,atomic

REGIMES=('em_group','date0.1_group','date0.5_group')
BUDGETS=(1024,4096,16384)


def quality(rows):
    return (len(rows)==16 and all(r['feasible'] for r in rows)
        and all(r['item_recall'] is not None for r in rows)
        and np.mean([r['item_recall'] for r in rows])>=.95-1e-12
        and np.mean([r['fill'] for r in rows])>=.99-1e-12
        and min(r['item_recall'] for r in rows)>=.80-1e-12
        and sum(r['item_recall']>=.90-1e-12 for r in rows)>=15
        and all(r[f]==0 for r in rows for f in
                ('eligibility_violations','group_violations','invalid_ids','duplicate_ids')))


def paired(candidate,baseline):
    a={r['query_id']:r for r in candidate};b={r['query_id']:r for r in baseline}
    if a.keys()!=b.keys() or len(a)!=len(candidate):raise ValueError('Unpaired or duplicate queries')
    ids=sorted(a);ratios=np.array([b[i]['ms']/a[i]['ms'] for i in ids])
    if not np.isfinite(ratios).all() or np.any(ratios<=0):raise ValueError('Invalid latency ratios')
    rng=np.random.default_rng(80017)
    logs=np.log(ratios);boot=np.exp(logs[rng.integers(len(ids),size=(2000,len(ids)))].mean(axis=1))
    return dict(geomean_speedup=float(np.exp(logs.mean())),faster_queries=int(np.sum(ratios>1)),
                bootstrap_lower95=float(np.quantile(boot,.025)))


def decide(root):
    root=Path(root);plan=json.loads((root/'plan.json').read_text())
    summary=json.loads((root/'summary.json').read_text())
    if summary['status']!='complete':raise ValueError('Cannot decide from an incomplete run')
    if plan['smoke'] or plan['queries']!=16:
        return dict(verdict='CALIBRATION_ONLY',reason='Requires full 16-query development screen')
    if not plan['upstream']:
        return dict(verdict='INCOMPLETE_CONTROLS',reason='Published upstream controls are required')
    if plan['budgets']!=list(BUDGETS) or set(plan['ks'])!={10,100} or plan['repeats']<3:
        raise ValueError('Run does not follow frozen development settings')
    files=sorted((root/'cells').glob('*.json'))
    expected_rows=2+2*len(plan['grid'])+4*len(plan['budgets'])+6*len(plan['upstream_grid'])
    if len(files)!=16*4*2:raise ValueError('Incomplete condition/query cells')
    rows=[]
    for p in files:
        cell=json.loads(p.read_text())
        if len(cell['rows'])!=expected_rows:raise ValueError('Incomplete method coverage')
        if cell['plan']!=identity(plan) or cell['checksum']!=identity(cell['rows']):
            raise ValueError('Cell integrity failure')
        rows.extend(cell['rows'])
    reports=[]
    for budget in BUDGETS:
        method=f'hnsw0_alternate_before/D{budget}';wins=[];details=[]
        for regime in REGIMES:
            subset=[r for r in rows if r['condition']==regime and r['k']==100]
            candidate=[r for r in subset if r['method']==method]
            if len(candidate)!=16 or {r['query_id'] for r in candidate}!=set(plan['query_ids']):
                raise ValueError('Incomplete candidate cells; not a valid negative result')
            baselines=[]
            for name in sorted({r['method'] for r in subset}):
                if not (name=='exact_simd' or name.startswith(('faiss_','upstream_','hnsw0_ordinary_before/'))):continue
                rs=[r for r in subset if r['method']==name]
                if quality(rs):baselines.append((name,rs))
            if not baselines:raise ValueError('Missing qualified baseline (exact must qualify)')
            name,baseline=min(baselines,key=lambda pair:np.mean(np.log([r['ms'] for r in pair[1]])))
            q=quality(candidate);comparison=paired(candidate,baseline) if q else None
            passed=bool(q and comparison['geomean_speedup']>=1.25-1e-12 and comparison['faster_queries']>=12
                        and comparison['bootstrap_lower95']>1.)
            if passed:wins.append(regime)
            details.append(dict(regime=regime,quality_pass=bool(q),baseline=name,
                baseline_timing_scope=baseline[0]['timing_scope'],comparison=comparison,pass_gate=passed))
        reports.append(dict(method=method,winning_regimes=wins,details=details))
    eligible=[r for r in reports if len(r['winning_regimes'])>=2]
    # Fixed tie-break: more passing regimes, then smaller distance budget.
    choice=max(eligible,key=lambda r:(len(r['winning_regimes']),-int(r['method'].split('/D')[1]))) if eligible else None
    return dict(verdict='CONFIRM_ONE_FROZEN_POLICY' if choice else 'STOP_D_AND_SWITCH_TO_A_B',
                selected=choice,reports=reports,
                note='Development screening only. Bootstrap does not correct configuration selection; untouched confirmation is mandatory.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);a=p.parse_args()
    report=decide(a.out);atomic(Path(a.out)/'DECISION.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
