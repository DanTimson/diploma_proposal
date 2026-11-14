"""Explicit metric denominators, separate from training and feature construction."""
from math import isfinite


def divide(a, b):
    return a / b if b else None


def note_metrics(published, helpful):
    if len(published) != len(helpful):
        raise ValueError("Publication decisions and truth labels must align")
    if any(x not in (0, 1, False, True) for x in [*published, *helpful]):
        raise ValueError("Expected binary decisions and labels")
    counts = dict(n_H=sum(published), n_U=len(published)-sum(published),
                  n_h=sum(helpful), n_u=len(helpful)-sum(helpful),
                  n_uH=sum(p and not h for p, h in zip(published, helpful)),
                  n_hU=sum(h and not p for p, h in zip(published, helpful)))
    return {**counts, **note_rates(counts)}


def note_rates(c):
    return dict(pollution=divide(c['n_uH'], c['n_H']),
                suppression=divide(c['n_hU'], c['n_h']),
                infiltration=divide(c['n_uH'], c['n_u']),
                waste=divide(c['n_hU'], c['n_U']))


def pair_metrics(gold, predicted):
    """Micro signed claim-document P/R/F1. Missing pairs mean NEI.

    This matches the SciFact-Open paper's label-aware pair task, not claim accuracy.
    Unjudged predictions are negatives under that benchmark convention; this does
    not prove the underlying scientific document contains no evidence.
    """
    labels = {'SUPPORT', 'CONTRADICT', 'NEI'}
    if not set(gold.values()) <= labels or not set(predicted.values()) <= labels:
        raise ValueError("Unrecognized evidence label")
    actual = {k: v for k, v in gold.items() if v != 'NEI'}
    pos = {k: v for k, v in predicted.items() if v != 'NEI'}
    tp = sum(actual.get(k) == v for k, v in pos.items())
    return dict(tp=tp, predicted=len(pos), relevant=len(actual),
                precision=divide(tp, len(pos)), recall=divide(tp, len(actual)),
                f1=divide(2*tp, len(pos)+len(actual)))


def target_hit_rate(rankings, target, train_seen, k):
    """Exclude users who already consumed the target; mask all train items."""
    if k < 1:
        raise ValueError("k must be positive")
    values = []
    for uid, ranking in rankings.items():
        seen = set(train_seen.get(uid, []))
        if target in seen:
            continue
        unique = list(dict.fromkeys(i for i in ranking if i not in seen))
        values.append(target in unique[:k])
    return {'eligible_users': len(values), 'target_hr': divide(sum(values),len(values))}


def score_recall(ranked, relevant, k):
    if k < 1:
        raise ValueError("k must be positive")
    return divide(len(set(ranked[:k]) & set(relevant)), len(set(relevant)))
