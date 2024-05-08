'''Complex Pfaffian network with permutation and translation symmetry'''
from qmc.qmc import QMC, make_potential
from qmc.model import (Linear, Sequential, Complex_Attention,
                       periodic_embed_complex, Complex_Tail)

from flax.linen import Module, compact
from jax.numpy.linalg import slogdet, det
from jax.numpy import max,abs, sqrt
from jax.tree_util import tree_map

from jax.random import PRNGKey

from typing import Callable, List

n_e = 8
n_up = 4
head = 8
out_dims = 8
dense = 64
latvec = 1.

scale = lambda s: lambda x: x*s

class resnet(Module):
    f: Callable
    @compact
    def __call__(self, x):
        return x+self.f(x)


model = Sequential((
    periodic_embed_complex(latvec),

    resnet(Sequential((scale(.2), Complex_Attention(out_dims, head, n_up)))),
    resnet(Sequential((scale(.07), Complex_Attention(out_dims, head, n_up)))),

    Complex_Tail(
        Sequential((Linear(dense), scale(.2), Linear(1))),
        Sequential((Linear(dense), scale(.2), Linear(1))),
        Sequential((Linear(dense), scale(.2), Linear(1))),
        n_up)
    ))

qmc = QMC(model, make_potential(latvec=latvec), n_e, 320,
          mc_step_size=.001, mc_show_acc=False, mc_n_step=100,
          mc_warmup=10000, key=PRNGKey(1))

#print("qmc.r.shape=",qmc.r.shape)

