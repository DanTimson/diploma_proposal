#!/usr/bin/env python3
"""Bundle completed run receipts; exclude datasets, indices, and native libraries."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='D-004-review.zip');p.add_argument('runs',nargs='+');a=p.parse_args()
    roots=[Path(r) for r in a.runs]
    if len({r.name for r in roots})!=len(roots):p.error('Run directory names must be unique')
    for root in roots:
        if not (root/'summary.json').is_file():p.error(f'Incomplete run: {root}')
    hashes={}
    with zipfile.ZipFile(a.out,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for root in roots:
            for path in sorted(root.iterdir()):
                if not path.is_file() or path.suffix not in {'.json','.log','.py'}:continue
                name=f'{root.name}/{path.name}';payload=path.read_bytes()
                z.writestr(name,payload);hashes[name]=hashlib.sha256(payload).hexdigest()
        z.writestr('SHA256.json',json.dumps(hashes,indent=2))
    print(a.out)

if __name__=='__main__':main()
