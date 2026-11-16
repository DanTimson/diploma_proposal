"""Opt-in boundary test against the compiled, pinned external implementation."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np
from diploma_starters.group_screen import NativeExact, write_bin


@unittest.skipUnless(os.environ.get('D004_ACORN_BINARY'), 'Set D004_ACORN_BINARY after setup_acorn.sh')
class ACORNIntegration(unittest.TestCase):
    def test_empty_singleton_and_all_masks(self):
        binary=Path(os.environ['D004_ACORN_BINARY']).resolve()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);n=100
            x=np.random.default_rng(12).normal(size=(n,8)).astype('f4')
            q=np.repeat(x[37:38],3,axis=0);g=np.arange(n,dtype='i8')
            masks=np.zeros((3,n),dtype='u1');masks[1,37]=1;masks[2]=1
            write_bin(root/'base.bin',x);write_bin(root/'query.bin',q)
            masks.tofile(root/'masks');g.tofile(root/'groups');g.astype('i4').tofile(root/'metadata')
            subprocess.run([str(binary),str(root/'base.bin'),str(root/'query.bin'),str(root/'masks'),
                str(root/'groups'),str(root/'metadata'),str(root/'result'), '10','100','512','1','11','1'],
                check=True,timeout=60,stdout=subprocess.DEVNULL)
            with (root/'result').open('rb') as f:
                np.testing.assert_array_equal(np.fromfile(f,'<u4',2),[3,10])
                ids=np.fromfile(f,'<i8',30).reshape(3,10)
                times=np.fromfile(f,'<f8',3);rejected=np.fromfile(f,'<i8',3)
            exact=NativeExact(x,g,root/'cache')
            for i in range(3):
                np.testing.assert_array_equal(ids[i][ids[i]>=0],exact.search(q[i],masks[i],10,True))
            self.assertTrue(np.all(times>=0))
            self.assertGreater(rejected[0],0)


if __name__=='__main__':unittest.main()
