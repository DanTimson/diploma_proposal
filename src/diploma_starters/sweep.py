"""Sequential, bounded and resumable Linux/WSL experiment orchestration."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from .audit import sha256


def stamp():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,delete=False) as f:
        temporary=Path(f.name)
        json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)


def process_start(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    except (FileNotFoundError,IndexError,ProcessLookupError):return None


def stop_run(directory):
    info=json.loads((Path(directory)/'pid.json').read_text())
    if process_start(info['pid'])!=info['start_ticks']:
        return 'No matching active runner; nothing signalled.'
    os.kill(info['pid'],signal.SIGTERM)
    return 'Stop requested; current child will be terminated and state retained.'


def validate_plan(plan):
    if plan.get('schema')!=1:raise ValueError('Unknown sweep schema')
    if not 0<plan['hours']<=12 or not 0<plan['job_minutes']<=120:
        raise ValueError('Invalid time limits')
    if not 2<=plan['memory_gib']<=24:raise ValueError('Invalid address-space limit')
    seen=set()
    for job in plan['jobs']:
        name=job['id']
        if not re.fullmatch(r'[a-zA-Z0-9_-]+',name) or name in seen:
            raise ValueError('Invalid/duplicate job ID')
        if not set(job.get('requires',[]))<=seen:raise ValueError('Dependencies must precede job')
        if job['kind'] not in ('fetch','extract','benchmark'):raise ValueError('Unknown job kind')
        if job['kind']=='benchmark' and not isinstance(job['args'],list):raise ValueError('args must be a list')
        seen.add(name)


def environment_signature(repo, plan):
    packages={}
    for name in ['numpy','scipy','scikit-learn','faiss-cpu','threadpoolctl','pandas','pyarrow']:
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    sources={str(p.relative_to(repo)):sha256(p) for p in sorted((repo/'src/diploma_starters').glob('*.py'))}
    sources['evidence/sources.lock.json']=sha256(repo/'evidence/sources.lock.json')
    record={'plan':plan,'python':sys.version,'executable':sys.executable,'packages':packages,
            'sources':sources,'repo':str(repo),'threads':1,'nice_increment':5}
    digest=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()
    return digest,record


def receipt_ok(path, token, digest=None):
    try:
        if digest is not None and sha256(path)!=digest:return False
        result=json.loads(Path(path).read_text())
        return result.get('receipt_token')==token and bool(result.get('summary'))
    except (OSError,ValueError,TypeError):return False


def prepare_scifact(archive, destination):
    from .assets import extract_scifact
    dest=Path(destination)
    checks={'data/claims.jsonl':'ecd7a1bec2fc1b483f9127f6d0927ff17ebeb08e120088c6e037dfc85ce2b63b',
            'data/corpus_candidates.jsonl':'664801e5088c177cd09c01bb410b61e73312bf393f88726f2b9ffb8416f44e0f'}
    if dest.exists():
        if all((dest/n).is_file() and sha256(dest/n)==h for n,h in checks.items()):return
        raise ValueError('Existing extraction incomplete/different; preserve it and choose another destination')
    dest.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='scifact-stage-',dir=dest.parent))
    try:
        extract_scifact(archive,stage)
        if not all(sha256(stage/n)==h for n,h in checks.items()):raise ValueError('SciFact content checksum mismatch')
        stage.rename(dest)
    finally:
        if stage.exists():shutil.rmtree(stage)


def job_command(job, output, token):
    prefix=[sys.executable,'-m']
    if job['kind']=='benchmark':
        return prefix+['diploma_starters.ann_cli','--out',str(output),'--receipt-token',token]+job['args']
    if job['kind']=='fetch':
        return prefix+['diploma_starters.cli','fetch',job['asset'],'--destination',job['destination']]
    return prefix+['diploma_starters.sweep','--extract-scifact',job['archive'],job['destination']]


def terminate(proc):
    if proc.poll() is not None:return
    try:os.killpg(proc.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:os.killpg(proc.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        proc.wait()


def run_child(command, log, cwd, timeout, memory_gib, cancelled):
    def limits():
        memory=int(memory_gib*1024**3)
        resource.setrlimit(resource.RLIMIT_AS,(memory,memory))
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        os.nice(5)
    env=dict(os.environ)
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
        env[key]='1'
    env['PYTHONUNBUFFERED']='1'
    env['PYTHONPATH']=str(Path(cwd)/'src')+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    started=time.monotonic()
    with Path(log).open('a') as f:
        f.write('\n'+stamp()+' '+json.dumps(command)+'\n');f.flush()
        proc=subprocess.Popen(command,cwd=cwd,stdout=f,stderr=subprocess.STDOUT,env=env,
                              start_new_session=True,preexec_fn=limits)
        try:
            while proc.poll() is None:
                if cancelled():terminate(proc);return 'interrupted',None,time.monotonic()-started
                if time.monotonic()-started>=timeout:
                    terminate(proc);return 'timeout',None,time.monotonic()-started
                try:proc.wait(timeout=min(1,max(.01,timeout-(time.monotonic()-started))))
                except subprocess.TimeoutExpired:pass
            return ('complete' if proc.returncode==0 else 'failed'),proc.returncode,time.monotonic()-started
        finally:terminate(proc)


def summary(directory, plan, state):
    directory=Path(directory)
    counts=Counter(v['status'] for v in state['jobs'].values())
    rows=['# Research sweep status', '',f"Updated: {stamp()}",
          f"Runner: {state['status']}; jobs: {dict(counts)}",'',
          'Exploratory screening only. Fastest eligible setting is selected from this run, not validated on a separate tuning set.',
          'Qualifying here means group recall >=99%, fill >=99%, zero eligibility/group violations. Task metrics still require review.', '',
          '| Case | Best exact p50 ms | Qualifying ANN | ANN p50 / p95 ms | Group recall | Speed ratio |',
          '|---|---:|---|---:|---:|---:|']
    for job in plan['jobs']:
        status=state['jobs'].get(job['id'],{})
        if job['kind']!='benchmark' or status.get('status')!='complete':continue
        try:r=json.loads((directory/'results'/f"{job['id']}.json").read_text())
        except (OSError,ValueError):continue
        exact=min(v['p50_ms'] for k,v in r['summary'].items() if k.startswith('exact_'))
        eligible=[(k,v) for k,v in r['summary'].items() if k.startswith('hnsw_') and
                  v.get('group_recall',0)>=.99 and v.get('fill',0)>=.99 and
                  v['eligibility_violations']==v['group_violations']==0]
        if not eligible:
            rows.append(f"| {job['id']} | {exact:.4f} | none | — | — | — |")
        else:
            name,v=min(eligible,key=lambda pair:pair[1]['p50_ms'])
            rows.append(f"| {job['id']} | {exact:.4f} | {name} | {v['p50_ms']:.4f} / {v['p95_ms']:.4f} | {v['group_recall']:.3%} | {exact/v['p50_ms']:.2f}x |")
    rows+=['','## Jobs','','| Job | State | Seconds | Detail |','|---|---|---:|---|']
    for job in plan['jobs']:
        s=state['jobs'].get(job['id'],{'status':'pending'})
        rows.append(f"| {job['id']} | {s['status']} | {s.get('seconds',0):.1f} | {s.get('detail','')} |")
    rows+=['','Keep full results and logs; do not infer scientific novelty from this summary. If a case failed, inspect logs/<job>.log.']
    tmp=directory/'SUMMARY.md.tmp';tmp.write_text('\n'.join(rows)+'\n');os.replace(tmp,directory/'SUMMARY.md')


def execute(repo, plan, directory, retry_failed=False):
    validate_plan(plan)
    repo=Path(repo).resolve();directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(repo).free < plan.get('minimum_free_gib',0)*1024**3:
        raise ValueError('Insufficient free disk space for this plan')
    with (directory/'.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('A runner already owns this directory')
        return execute_locked(repo,plan,directory,retry_failed)


def execute_locked(repo, plan, directory, retry_failed):
    signature,environment=environment_signature(repo,plan)
    manifest=directory/'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text())['signature']!=signature:
            raise ValueError('Code/plan/environment changed: use a new run directory')
    else:
        if (directory/'state.json').exists():raise ValueError('State exists without manifest')
        atomic_json(manifest,{'signature':signature,'created':stamp(),**environment})
    for part in ['results','logs']:(directory/part).mkdir(exist_ok=True)
    state_path=directory/'state.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {'jobs':{}}
    state['status']='running';state['launch_started']=stamp()
    atomic_json(directory/'pid.json',{'pid':os.getpid(),'start_ticks':process_start(os.getpid())})
    cancelled=[False]
    previous={sig:signal.signal(sig,lambda *_:cancelled.__setitem__(0,True)) for sig in [signal.SIGINT,signal.SIGTERM]}
    deadline=time.monotonic()+plan['hours']*3600
    def save():
        state['updated']=stamp();atomic_json(state_path,state);summary(directory,plan,state)
    try:
        save()
        for job in plan['jobs']:
            name=job['id'];token=signature+':'+name;output=directory/'results'/f'{name}.json'
            old=state['jobs'].get(name,{})
            if job['kind']=='benchmark' and receipt_ok(output,token,old.get('sha256')):
                state['jobs'][name]={**old,'status':'complete','sha256':sha256(output),'detail':'verified/recovered receipt'}
                save();continue
            if old.get('status') in ('failed','timeout') and not retry_failed:
                continue
            if any(state['jobs'].get(dep,{}).get('status')!='complete' for dep in job.get('requires',[])):
                state['jobs'][name]={'status':'blocked','detail':'dependency incomplete'};save();continue
            if cancelled[0] or time.monotonic()>=deadline:break
            if job['kind']=='benchmark' and environment_signature(repo,plan)[0]!=signature:
                raise ValueError('Code/environment changed during sweep; use a new run directory')
            # Damaged/incomplete receipts remain available for diagnosis.
            if output.exists():output.rename(output.with_name(output.name+f'.invalid-{time.time_ns()}'))
            state['jobs'][name]={'status':'running','started':stamp()};save()
            print(f'{stamp()} START {name}',flush=True)
            timeout=min(plan['job_minutes']*60,deadline-time.monotonic())
            status,code,seconds=run_child(job_command(job,output,token),directory/'logs'/f'{name}.log',
                                          repo,timeout,plan['memory_gib'],lambda:cancelled[0])
            if status=='complete' and job['kind']=='benchmark' and not receipt_ok(output,token):
                status='failed'
            result={'status':status,'seconds':seconds,'exit_code':code,'finished':stamp()}
            if status=='complete' and job['kind']=='benchmark':result['sha256']=sha256(output)
            state['jobs'][name]=result;save();print(f'{stamp()} {status.upper()} {name}',flush=True)
            if cancelled[0]:break
        if cancelled[0]:state['status']='interrupted'
        elif time.monotonic()>=deadline:state['status']='budget_exhausted'
        elif all(state['jobs'].get(j['id'],{}).get('status')=='complete' for j in plan['jobs']):state['status']='complete'
        else:state['status']='finished_with_failures'
    finally:
        if state.get('status')=='running':state['status']='interrupted'
        save()
        for sig,handler in previous.items():signal.signal(sig,handler)
    return state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',default='topics/D_hnsw/sweep.json')
    p.add_argument('--run-dir',default='local-runs/D-002-sweep')
    p.add_argument('--retry-failed',action='store_true')
    p.add_argument('--status',action='store_true')
    p.add_argument('--stop',action='store_true')
    p.add_argument('--extract-scifact',nargs=2,metavar=('ARCHIVE','DESTINATION'),help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.extract_scifact:prepare_scifact(*a.extract_scifact);return
    if a.status:print((Path(a.run_dir)/'SUMMARY.md').read_text());return
    if a.stop:print(stop_run(a.run_dir));return
    for dependency in ['numpy','scipy','scikit-learn','faiss-cpu','threadpoolctl']:
        importlib.metadata.version(dependency)
    plan=json.loads(Path(a.plan).read_text())
    state=execute(Path.cwd(),plan,a.run_dir,a.retry_failed)
    print('Final status:',state['status'])
    if state['status']=='finished_with_failures':raise SystemExit(1)


if __name__=='__main__':main()
