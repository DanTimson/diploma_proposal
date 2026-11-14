import argparse
import json
from pathlib import Path
from . import audit, assets, prototypes


def main():
    p=argparse.ArgumentParser(description='Research starters: smoke, source audits and workload plans')
    p.add_argument('--out',help='Write JSON to a new output file')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('smoke')
    q=sub.add_parser('audit-notes');q.add_argument('csv')
    q=sub.add_parser('audit-recad');q.add_argument('zip')
    q=sub.add_parser('audit-scifact');q.add_argument('claims');q.add_argument('predictions')
    q=sub.add_parser('plan');q.add_argument('config')
    q=sub.add_parser('fetch');q.add_argument('asset');q.add_argument('--lock',default='evidence/sources.lock.json');q.add_argument('--destination',default='external')
    q=sub.add_parser('extract-scifact');q.add_argument('archive');q.add_argument('destination')
    a=p.parse_args()
    if a.command=='smoke':result=prototypes.smoke()
    elif a.command=='audit-notes':result=audit.notes(a.csv)
    elif a.command=='audit-recad':result=audit.recad(a.zip)
    elif a.command=='audit-scifact':result=audit.scifact(a.claims,a.predictions)
    elif a.command=='plan':
        result=json.loads(Path(a.config).read_text())
        result['execution']='NOT RUN: declaration of planned work, not an implemented experiment executor'
    elif a.command=='fetch':
        spec=json.loads(Path(a.lock).read_text())['assets'][a.asset]
        result={'downloaded':assets.fetch(spec,Path(a.destination)/spec['filename'])}
    else:result={'extracted':assets.extract_scifact(a.archive,a.destination)}
    text=json.dumps(result,indent=2,allow_nan=False,ensure_ascii=False)
    if a.out:
        dest=Path(a.out);dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('x',encoding='utf-8') as f:f.write(text+'\n')
    print(text)


if __name__=='__main__':main()
