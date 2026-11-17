"""Compiled selection and one fixed, predicate-aware routing hypothesis."""
import ctypes
import os
from pathlib import Path
import platform
import subprocess
import numpy as np
from .baseline_controls import SIMDExact
from .explore import Kernel,ROOT,identity
from .group_screen import digest


class FastSelector(SIMDExact):
    def select(self,q,mask,ids,k):
        q=np.ascontiguousarray(q,dtype='f4');mask=np.asarray(mask,dtype=bool)
        ids=np.asarray(ids,dtype='i8')
        if q.shape!=(self.x.shape[1],) or not np.isfinite(q).all() or mask.shape!=(len(self.x),) or ids.ndim!=1 or k<1:
            raise ValueError('Invalid candidate selection arguments')
        missing=(ids==-1)|(ids==2**32-1)
        if np.any((~missing)&((ids<0)|(ids>=len(self.x)))):raise ValueError('Invalid candidate ID')
        ids=np.unique(ids[~missing]);ids=np.ascontiguousarray(ids[mask[ids]],dtype='i8')
        out=np.empty(k,dtype='i8')
        count=self.fn(self.x,q,ids,self.groups,len(ids),len(self.x),self.x.shape[1],k,1,out)
        if count<0:raise ValueError('Native candidate reduction failed')
        return out[:count]


class RetentionKernel(Kernel):
    def __init__(self,x,edges,cache):
        super().__init__(x,edges,cache)
        source=ROOT/'native/retention.cpp';compiler=os.environ.get('CXX','g++')
        version=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0]
        flags=['-O3','-std=c++17','-shared','-fPIC','-fopenmp-simd','-march=native']
        key=identity([digest(source),version,flags,platform.machine()])[:20]
        binary=Path(cache)/f'retention-{key}.so'
        if not binary.exists():
            temp=binary.with_suffix(f'.{os.getpid()}.so')
            try:
                subprocess.run([compiler,*flags,str(source),'-o',str(temp)],check=True)
                temp.replace(binary)
            finally:temp.unlink(missing_ok=True)
        argtypes=self.fn.argtypes
        self.lib=ctypes.CDLL(str(binary));self.fn=self.lib.retention
        self.fn.argtypes=argtypes;self.fn.restype=ctypes.c_int
        self.receipt=dict(compiler=version,flags=flags,source_sha256=digest(source),
                          binary_sha256=digest(binary),accumulation='float64 SIMD reduction')

    def search(self,q,groups,mask,starts,budget,beam,cap,reserve,k):
        if cap not in (0,1,2,3) or reserve!=0:raise ValueError('Unsupported retention policy')
        return super().search(q,groups,mask,starts,budget,beam,cap,reserve,k)
