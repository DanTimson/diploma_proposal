#!/usr/bin/env python3
"""Audit natural grouping semantics before buying an ANN benchmark with them."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from diploma_starters.assets import fetch
from diploma_starters.group_screen import digest,write_json

def audit(rows):
    def counts(values):
        c=Counter(values);n=len(rows)
        return dict(groups=len(c),largest_group=max(c.values()),largest_fraction=max(c.values())/n,
                    fraction_items_in_nonsingleton_groups=sum(v for v in c.values() if v>1)/n)
    signatures=[tuple(sorted(set(r['sub_categories']))) for r in rows]
    author_sets=[tuple(sorted(set(r['authors']))) for r in rows]
    author_counts=Counter(a for authors in author_sets for a in authors)
    return dict(items=len(rows),exact_subcategory_set=counts(signatures),exact_author_set=counts(author_sets),
        multi_author_items=sum(len(a)>1 for a in author_sets),distinct_author_strings=len(author_counts),
        repeated_author_strings=sum(v>1 for v in author_counts.values()),
        by_filter_value=[dict(number_of_sub_categories=value,eligible=sum(r['number_of_sub_categories']==value for r in rows),
            category_groups=len({s for r,s in zip(rows,signatures) if r['number_of_sub_categories']==value}))
            for value in sorted({r['number_of_sub_categories'] for r in rows})],
        interpretation='Category-set groups are a valid partition but do not imply source independence. One-per-author with all coauthors is overlapping set exclusion, not this objective. Author strings are not disambiguated identities.')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-dir',default='external/fanns-medium');p.add_argument('--out',required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    lock=json.loads((root/'evidence/sources.lock.json').read_text())['assets']
    spec=lock['fanns_medium_database_attributes'];path=Path(a.data_dir)/spec['filename'];fetch(spec,path)
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    write_json(a.out,dict(sha256=digest(path),source_sha256=digest(__file__),**audit(rows)))
    print('COMPLETE',a.out)
if __name__=='__main__':main()
