from fire import Fire
from time import time

from importlib import import_module
import os
import pathlib
import shutil
import sys

from qmc.analysis import get_stat

def chk_dir(name):
    if not os.path.exists(name):
        os.mkdir(name)
    if not os.path.isdir(name):
        raise ValueError(f"{name} is not a directory")

class App:

    @property
    def path(self):
        return f"out/{self.name}"

    def _get_data_file(self,n=None):
        if n==None:
            return f"{self.path}/data.npz"
        else:
            return f"{self.path}/data.{n}.npz"

    @property
    def model_file(self):
        return f"{self.path}/model.py"

    @property
    def hist_file(self):
        return f"{self.path}/history"

    def _log(self,*x,silent=False):
        if not silent:
            print(*x, flush=True)
        with open(f"{self.path}/history", "at") as f:
            print(*x, file=f, flush=True)
            

    def create(self, name, from_name=None):
        '''
        Create a QMC task. model_file.py should contain qmc
        which is a QMC object. Output files are at out/name
        '''
        self.name = name
        if not os.path.exists("out"):
            os.mkdir("out")

        if not os.path.exists(self.path):
            if from_name != None:
                shutil.copytree(f"out/{from_name}", f"out/{name}")
            else:
                os.mkdir(self.path)
                shutil.copyfile("model.py", self.model_file)

    def run(self, name, alp, metric=None, silent=False, data=None, bound=None):
        '''
        Run a QMC task
        '''
        self.name = name
        sys.path.insert(0, self.path)
        qmc = import_module("model").qmc
        if os.path.exists(self._get_data_file(data)):
            qmc.load(self._get_data_file(data))

        last = time()

        save_number = 0
        while os.path.exists(self._get_data_file(save_number)):
            save_number += 1

        self._log(f'#{{"alp":{alp}, "metric":{metric!r}, "start":{data}, "save":{save_number}}}', silent=silent)

        last_E = None

        while True:
            if (E := qmc.update(alp,metric)) is None:
                continue
            self._log(time()-last, E, silent=silent)
            last = time()
            if last_E != None and bound != None and abs(E-last_E)>bound:
                print(f"E={E} skipping")
                qmc.load(self._get_data_file(save_number))
                qmc.sample(qmc.n_step)
                continue
            last_E = E
            qmc.save(self._get_data_file())
            qmc.save(self._get_data_file(save_number))

    def list(self):
        stat = get_stat()
        print('name','energy','desc',sep='\t\t')
        for k,v in stat.items():
            E = None if len(v['energy']) == 0 else v['energy'][-1]
            print(k,E,v['desc'],sep='\t\t')


Fire(App)
