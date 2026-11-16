#!/usr/bin/env python3
"""Package receipts without datasets or graph indexes; refuse overwrites."""
import argparse
from pathlib import Path
import zipfile
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);p.add_argument('runs',nargs='+');a=p.parse_args()
runs=[Path(x) for x in a.runs]
if len({r.name for r in runs})!=len(runs):p.error('Run basenames must be distinct')
files=[]
for run in runs:
 if not (run/'summary.json').exists():p.error(f'Incomplete run: {run}')
 for f in sorted(run.iterdir()):
  if f.is_file() and (f.suffix in ['.json','.log'] or f.name.endswith(('_idx_uint32.bin','_dists_float.bin'))):files.append((f,run.name+'/'+f.name))
out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED) as z:
 for f,name in files:z.write(f,name)
print(out, out.stat().st_size,'bytes;',len(files),'receipt files')
