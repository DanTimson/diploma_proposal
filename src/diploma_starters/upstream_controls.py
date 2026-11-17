"""Pinned upstream search with a logging-only instrumentation copy.

No upstream source or build output is edited. Reuses its Unix Makefiles recipe.
"""
import json
import os
from pathlib import Path
import shlex
import subprocess
import numpy as np
from .explore import REV, atomic, identity
from .group_screen import digest, write_bin


def instrument(source):
    anchor = '        std::sort(latency_stats.begin(), latency_stats.end());'
    if source.count(anchor) != 1:
        raise ValueError('Pinned upstream instrumentation anchor differs')
    addition = '''        // D-007: export per-query counters BEFORE sorting latency_stats.
        // Export is outside timed search; search logic is unchanged.
        {
            std::ofstream receipt(result_path_prefix + "_" + std::to_string(L) + "_per_query.tsv");
            if (!receipt) throw std::runtime_error("Cannot write D-007 timings");
            receipt << std::setprecision(12);
            for (size_t qi = 0; qi < query_num; ++qi)
                receipt << qi << "\\t" << latency_stats[qi] << "\\t" << cmp_stats[qi] << "\\n";
        }
'''
    # high_resolution_clock may alias wall time and can jump backwards.
    # Use the same monotonic clock for every start/end in this private copy.
    source = source.replace("std::chrono::high_resolution_clock", "std::chrono::steady_clock")
    return source.replace(anchor,addition+anchor), addition


def recipes(flags_text, link_text, source, obj, binary):
    flags = {}
    for line in flags_text.splitlines():
        if line.startswith(('CXX_DEFINES =','CXX_INCLUDES =','CXX_FLAGS =')):
            name,value = line.split('=',1); flags[name.strip()] = shlex.split(value)
    if set(flags) != {'CXX_DEFINES','CXX_INCLUDES','CXX_FLAGS'}:
        raise ValueError('Unsupported upstream flags.make')
    link = shlex.split(link_text)
    if not link or any(token in ('&&',';','|','>') for token in link):
        raise ValueError('Unsupported upstream link recipe')
    objects = [i for i,v in enumerate(link) if v.endswith('search_memory_index.cpp.o')]
    if len(objects)!=1 or link.count('-o')!=1:
        raise ValueError('Unsupported upstream object/link layout')
    link[objects[0]] = str(obj)
    link[link.index('-o')+1] = str(binary)
    compile_command = [link[0]] + flags['CXX_DEFINES']+flags['CXX_INCLUDES']+flags['CXX_FLAGS']
    compile_command += ['-c',str(source),'-o',str(obj)]
    return compile_command,link


def build_probe(upstream, cache):
    up=Path(upstream).resolve(); cache=Path(cache).resolve(); cache.mkdir(parents=True,exist_ok=True)
    rev=subprocess.check_output(['git','-C',str(up),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(up),'diff','HEAD','--'],text=True).strip()
    if rev!=REV or dirty:raise ValueError('Require clean pinned Diverse DiskANN checkout')
    if not (up/'build/apps/search_memory_index').is_file():
        raise ValueError('Missing original search_memory_index; run the existing setup_diverse.sh first')
    target=up/'build/apps/CMakeFiles/search_memory_index.dir'
    try:
        flags=(target/'flags.make').read_text(); link=(target/'link.txt').read_text()
    except FileNotFoundError as exc:
        raise ValueError('Build search_memory_index using the existing setup_diverse.sh (Unix Makefiles required)') from exc
    original=up/'apps/search_memory_index.cpp'
    source,addition=instrument(original.read_text())
    libraries=sorted((up/'build').rglob('libdiskann.so*'))+sorted((up/'build').rglob('libdiskann*.a'))
    hashes={str(p):digest(p) for p in libraries if p.is_file()}
    if not hashes:raise ValueError('Missing built DiskANN library')
    compiler=shlex.split(link)[0]
    version=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0]
    key=identity([source,flags,link,hashes,version])[:20]
    root=cache/key;root.mkdir(exist_ok=True)
    cpp=root/'search_memory_index_instrumented.cpp';obj=root/'probe.o';binary=root/'search_probe'
    compile_command,link_command=recipes(flags,link,cpp,obj,binary)
    receipt=root/'build.json'
    if not receipt.exists():
        cpp.write_text(source);(root/'logging-addition.txt').write_text(addition)
        try:
            with (root/'build.log').open('w') as log:
                subprocess.run(compile_command,cwd=up/'build/apps',stdout=log,stderr=subprocess.STDOUT,check=True)
                subprocess.run(link_command,cwd=up/'build/apps',stdout=log,stderr=subprocess.STDOUT,check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f'Upstream probe build failed; see {root / "build.log"}') from exc
        atomic(receipt,dict(revision=rev,original_sha256=digest(original),source_sha256=digest(cpp),
            binary_sha256=digest(binary),compiler=version,libraries=hashes,timing_clock="std::chrono::steady_clock",
            compile=compile_command,link=link_command,logging_addition=addition))
    info=json.loads(receipt.read_text())
    if not binary.exists() or digest(binary)!=info['binary_sha256']:
        raise ValueError('Cached upstream probe changed')
    return binary,info


def read_probe(prefix, L, nq, pool, n):
    path=Path(str(prefix)+f'_{L}_idx_uint32.bin')
    with path.open('rb') as f:
        shape=np.fromfile(f,dtype='<u4',count=2);ids=np.fromfile(f,dtype='<u4')
    if shape.tolist()!=[nq,pool] or ids.size!=nq*pool:
        raise ValueError('Upstream result dimensions differ')
    ids=ids.reshape(nq,pool).astype('i8')
    if np.any((ids>=n)&(ids!=2**32-1)):raise ValueError('Invalid upstream result ID')
    ids[ids==2**32-1]=-1
    timing=np.loadtxt(str(prefix)+f'_{L}_per_query.tsv',ndmin=2)
    if timing.shape!=(nq,3) or not np.array_equal(timing[:,0],np.arange(nq)):
        raise ValueError('Upstream per-query timing schema differs')
    if not np.isfinite(timing).all() or np.any(timing[:,1:]<0):
        raise ValueError('Invalid upstream timing/counters')
    return ids,timing[:,1]/1000,timing[:,2].astype('i8')


def verify_stored_vectors(path, vectors):
    with Path(path).open('rb') as f:shape=np.fromfile(f,dtype='<u4',count=2)
    if shape.tolist()!=list(vectors.shape) or Path(path).stat().st_size!=8+vectors.size*4:
        raise ValueError('Upstream stored-vector shape differs (padded/dynamic formats unsupported)')
    stored=np.memmap(path,dtype='<f4',mode='r',offset=8,shape=vectors.shape)
    for start in range(0,len(vectors),1024):
        if not np.array_equal(stored[start:start+1024],vectors[start:start+1024]):
            raise ValueError('Upstream stored vectors differ from prepared vectors')


def prepare_candidates(prepared, out, data, nq, repeats, probe, grid, timeout, original_probe):
    """Return timed candidate batches; filtering/group reduction is downstream."""
    root=out/'upstream';root.mkdir(parents=True,exist_ok=True)
    queries=np.tile(data['queries'][:nq],(repeats+1,1))
    query_path=root/'queries.bin';write_bin(query_path,queries)
    n=len(data['groups']); result={};query_sha=digest(query_path)
    vectors=np.load(prepared/'vectors.npy',mmap_mode='r')
    for graph in ('diskann','diverse_diskann'):
        staging=root/graph;staging.mkdir(exist_ok=True);prefix=staging/'index'
        # Static indexes: preserve IDs; do not rewrite the original graph/data.
        for suffix in ('','.data'):
            source=(prepared/(graph+suffix)).resolve()
            if not source.is_file():raise ValueError('Missing D-006 upstream artifact: '+str(source))
            target=Path(str(prefix)+suffix)
            if target.is_symlink():
                if target.resolve()!=source:raise ValueError('Staging symlink differs')
            elif target.exists():raise ValueError('Refusing to replace upstream staging file')
            else:target.symlink_to(source)
        np.savetxt(str(prefix)+'_sellers.txt',data['groups'],fmt='%d')
        verify_stored_vectors(prepared/(graph+'.data'),vectors)
        inputs={str(prepared/(graph+s)):digest(prepared/(graph+s)) for s in ('','.data')}
        inputs['queries']=query_sha;inputs['groups']=digest(str(prefix)+'_sellers.txt')
        for mode,cap in [('ordinary',0),('diverse1',1),('diverse4',4)]:
            for ef,pool in grid:
                name=f'{graph}-{mode}-L{ef}-P{pool}';destination=root/name
                receipt=root/(name+'.json')
                command=[str(probe),'--data_type','float','--dist_fn','l2',
                    '--index_path_prefix',str(prefix),'--query_file',str(query_path),
                    '--result_path',str(destination),'-K',str(pool),'-L',str(ef),'-T','1',
                    '--diverse_search',str(cap>0).lower(),'--max_K_per_seller',str(cap),
                    '--scale_seller_limits','false','--post_process','false']
                # Verify both graph and stored-vector artifacts on every invocation.
                plan=identity([command,inputs,digest(probe)])
                if receipt.exists():
                    saved=json.loads(receipt.read_text())
                    if saved['plan']!=plan:raise ValueError('Upstream cached plan differs')
                    for file,sha in saved['outputs'].items():
                        if digest(file)!=sha:raise ValueError('Upstream cached output changed')
                else:
                    print('UPSTREAM',name,flush=True)
                    with (root/(name+'.log')).open('w') as log:
                        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)
                    files=[Path(str(destination)+f'_{ef}_idx_uint32.bin'),
                           Path(str(destination)+f'_{ef}_per_query.tsv')]
                    atomic(receipt,dict(plan=plan,command=command,inputs=inputs,
                        outputs={str(p):digest(p) for p in files}))
                ids,ms,cmps=read_probe(destination,ef,len(queries),pool,n)
                if (ef,pool)==grid[0]:
                    # Check the logging copy against the original binary in every
                    # construction/search combination before using its measurements.
                    check=root/(name+'-original-check.json')
                    check_plan=identity([plan,digest(original_probe)])
                    if check.exists():
                        if json.loads(check.read_text())['plan']!=check_plan:
                            raise ValueError('Original-binary validation receipt differs')
                    else:
                        original_destination=root/(name+'-original')
                        argv=command.copy();argv[0]=str(original_probe)
                        argv[argv.index('--result_path')+1]=str(original_destination)
                        with (root/(name+'-original.log')).open('w') as log:
                            subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)
                        original_file=Path(str(original_destination)+f'_{ef}_idx_uint32.bin')
                        instrumented_file=Path(str(destination)+f'_{ef}_idx_uint32.bin')
                        if digest(original_file)!=digest(instrumented_file):
                            raise ValueError('Logging copy differs from original upstream search')
                        atomic(check,dict(plan=check_plan,ids_byte_identical=True,command=argv,
                                          result_sha256=digest(original_file)))
                ids=ids.reshape(repeats+1,nq,pool)
                if not all(np.array_equal(ids[0],ids[r]) for r in range(1,repeats+1)):
                    raise ValueError('Upstream search changed across repeated identical queries')
                result[(graph,mode,ef,pool)]=(ids[0],ms.reshape(repeats+1,nq)[1:],
                                            cmps.reshape(repeats+1,nq)[1:])
    return result
