"""Small mechanism probes. These are NOT the published CN or MultiVerS models."""
from collections import Counter
from math import isfinite


def history_cap_weights(history):
    """One unit of mass per exactly identical historical profile.

    Features contain historical (item, rating) pairs only. Target ratings, true
    identities and attacker labels must never enter this function. Missing
    histories get individual weight one: this toy has NO cold-start protection.
    Exact duplicates are a diagnostic; benign specialists can be discounted too.
    """
    signatures = {}
    for uid, events in history.items():
        pairs = sorted(events)
        if len({item for item, _ in pairs}) != len(pairs):
            raise ValueError("Repeated historical item for one user")
        signatures[uid] = tuple(pairs) if pairs else ('unseen', uid)
    sizes = Counter(signatures.values())
    return {uid: 1/sizes[sig] for uid, sig in signatures.items()}


def aggregate(ratings, weights=None):
    if len({u for u, _ in ratings}) != len(ratings):
        raise ValueError("At most one target rating per account")
    weighted = [(float(r), float(weights[u] if weights is not None else 1))
                for u, r in ratings]
    if any(not isfinite(r) or not 0 <= r <= 1 or not isfinite(w) or w < 0
           for r, w in weighted):
        raise ValueError("Ratings in [0,1] and finite nonnegative weights required")
    den = sum(w for _, w in weighted)
    return sum(r*w for r, w in weighted)/den if den else None


def select_evidence(candidates, k, diversity_bonus=0.0):
    """Greedy score plus predicted-stance coverage; one record per origin.

    Candidate keys: id, origin, score, predicted_stance. No gold label is read.
    'origin' is known only for controlled duplication experiments or explicit
    provenance. It must not be inferred from SciFact annotation provenance.
    """
    if k < 1 or diversity_bonus < 0:
        raise ValueError("Invalid selection budget or bonus")
    if len({str(c['id']) for c in candidates}) != len(candidates):
        raise ValueError("Duplicate candidate IDs")
    if any(not isfinite(float(c['score'])) for c in candidates):
        raise ValueError("Scores must be finite")
    remaining = list(candidates); selected = []; origins = set(); stances = set()
    while remaining and len(selected) < k:
        choices = [c for c in remaining if c['origin'] not in origins]
        if not choices:
            break
        def utility(c):
            s = c['predicted_stance']
            bonus = diversity_bonus if s in {'SUPPORT','CONTRADICT'} and s not in stances else 0
            return (-(float(c['score'])+bonus), str(c['id']))
        best = min(choices,key=utility)
        selected.append(best); origins.add(best['origin']);stances.add(best['predicted_stance'])
        remaining.remove(best)
    return selected


def smoke():
    history = {'a':[(1,1)], 'b':[(2,1)], **{f'x{i}':[(3,0),(4,1)] for i in range(8)}}
    ratings = [('a',1),('b',1),*[(f'x{i}',0) for i in range(8)]]
    candidates = [dict(id=f'copy{i}',origin='same-source',score=1-i*.01,predicted_stance='SUPPORT') for i in range(5)]
    candidates += [dict(id='counter',origin='other-source',score=.8,predicted_stance='CONTRADICT')]
    return {'status':'synthetic mechanism check only; not benchmark evidence',
            'A_notes':{'uniform':aggregate(ratings), 'history_cap':aggregate(ratings,history_cap_weights(history)),
                       'failure_case':'The same cap applies if the eight identical histories are independent honest specialists.'},
            'B_evidence':{'ordinary_top2':[c['id'] for c in candidates[:2]],
                          'origin_unique_top2':[c['id'] for c in select_evidence(candidates,2)]}}
