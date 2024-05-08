from flax.linen import Module, compact, initializers

from typing import Callable, List

from jax.numpy import (einsum, tanh, mean, indices, concatenate, zeros,
                       arange, cos, sin, pi, stack, sqrt, indices, array,
                       exp, eye)
from jax.numpy.linalg import slogdet

from jax.nn import softmax

from jax.random import PRNGKey, split, normal

import numpy as np

class Quadratic(Module):
    out_dims   : int
    init_scale : float = 1.
    activation : Callable = tanh

    @compact
    def __call__(self, x):
        a = Linear(self.out_dims, self.init_scale, self.activation)(x)
        b = Linear(self.out_dims, self.init_scale, self.activation)(x)
        return a*b

class Linear(Module):
    out_dims   : int
    init_scale : float = 1.
    activation : Callable = tanh

    @compact
    def __call__(self, x):
        self.sow("intermediate", "a", x)
        k = sqrt(x.shape[-1])
        if self.out_dims == 1:
            W = self.param('W', initializers.normal(self.init_scale),
                           (x.shape[-1], ))
            b = self.param('b', initializers.zeros_init(), tuple())
            y = einsum("i,...i->...", W, x) / k + b *1e-1
        else:
            W = self.param('W', initializers.normal(self.init_scale),
                           (x.shape[-1], self.out_dims))
            b = self.param('b', initializers.zeros_init(),
                           (self.out_dims,))
            y = einsum("ij,...i->...j", W, x) / k + b*1e-1
        y = self.perturb('e', y)
        self.sow("debug","x",x)
        self.sow("debug","y",y)
        y = self.activation(y)
        self.sow("debug","fy",y)
        return y

class ResNet(Module):
    layer: Callable

    @compact
    def __call__(self, x):
        n_e, n_in = x.shape
        y = self.layer(x)
        if y.shape[1] >= n_in:
            y = y.at[:, :n_in].add(x)
        return y

@jit
def slogpf(M):
    N = M.shape[0]
    for i in range(N-2):
        u = M[:,i].at[:i+1].set(0)
        v = zeros(N,dtype=M.dtype).at[i+1].set(sqrt((u*u).sum()))
        n = u - v
        A = eye(N) - 2*n[:,None]*n[None,:]/(n*n).sum()
        M = A @ M @ A
    M = M.diagonal(1)[::2]
    _M = abs(M)
    return (M/_M).prod(), log(_M).sum()


class Tail(Module):
    uu: Callable
    dd: Callable | None = None
    ud: Callable | None = None
    n_up: int = 0

    @compact
    def __call__(self, x):
        n_e,feat,_,_ = x.shape
        A = array([[[1.,0.],[0.,1.]],[[0.,-1.],[1.,0.]]])
        x  = einsum("ifza,jgzb,cab->ijfgzc",x,x,A)
        x = x.reshape((n_e,n_e,feat**2 * 6))

        n_up = self.n_up
        u,d = slice(None,n_up), slice(n_up,None)
        transpose = lambda x: einsum("ij...->ji...", x)
        if n_up == 0:
            x = self.uu(x)
            x -= transpose(x)
        else:
            uu = self.uu(x[u,u])
            dd = self.dd(x[d,d])
            ud = self.ud(x[u,d])
            x = (zeros((n_e,n_e))
                 .at[u,u].set(uu - transpose(uu))
                 .at[d,d].set(dd - transpose(dd))
                 .at[u,d].set(ud)
                 .at[d,u].set(-transpose(ud)))

        return  slogdet(x)[1] / 2

class Complex_Tail(Module):
    uu: Callable
    dd: Callable | None = None
    ud: Callable | None = None
    n_up: int = 0

    @compact
    def __call__(self, x):

        eps = (zeros((2,2,2))
                .at[0,0,0].set(1)
                .at[1,1,0].set(1)
                .at[0,1,1].set(1)
                .at[1,0,1].set(-1))

        n_e,feat,_,_ = x.shape
        x  = einsum("ifza,jgzb,abc->ijfgzc",x,x,eps)
        x = x.reshape((n_e,n_e,feat**2 * 6))

        n_up = self.n_up
        u,d = slice(None,n_up), slice(n_up,None)
        transpose = lambda x: einsum("ij...->ji...", x)
        if n_up == 0:
            x = self.uu(x)
            x -= transpose(x)
        else:
            uu = self.uu(x[u,u])
            dd = self.dd(x[d,d])
            ud = self.ud(x[u,d])
            x = (zeros((n_e,n_e))
                 .at[u,u].set(uu - transpose(uu))
                 .at[d,d].set(dd - transpose(dd))
                 .at[u,d].set(ud)
                 .at[d,u].set(-transpose(ud)))

        return  slogdet(x)[1] / 2



class Complex_Attention(Module):
    out_dims: int
    head: int
    n_up: int = 0

    @compact
    def __call__(self, x):
        out_dims, head, n_up = self.out_dims, self.head, self.n_up
        n_e,feat,_,_ = x.shape

        init_fn = initializers.normal(1.)

        M_uu = self.param("M_uu", init_fn, (head,feat,feat,3,2))
        A_uu = self.param("A_uu", init_fn, (head,out_dims,feat,3,2))
        if n_up > 0:
            M_dd = self.param("M_dd", init_fn, (head,feat,feat,3,2))
            M_ud = self.param("M_ud", init_fn, (head,feat,feat,3,2))
            M_du = self.param("M_du", init_fn, (head,feat,feat,3,2))
            A_dd = self.param("A_dd", init_fn, (head,out_dims,feat,3,2))
            A_ud = self.param("A_ud", init_fn, (head,out_dims,feat,3,2))
            A_du = self.param("A_du", init_fn, (head,out_dims,feat,3,2))
            u,d = x[:n_up], x[n_up:]
        else:
            u = x

        ksi = (zeros((2,2,2))
                .at[0,0,0].set(1)
                .at[1,1,0].set(-1)
                .at[0,1,1].set(1)
                .at[1,0,1].set(1))

        eps = (zeros((2,2,2))
                .at[0,0,0].set(1)
                .at[1,1,0].set(1)
                .at[0,1,1].set(1)
                .at[1,0,1].set(-1))

        _uu = einsum("nfza,mgzb,hfgzc,abc->hnm", u, u, M_uu, eps)
        #uu = softmax(einsum("nfza,mgzb,hfgzc,abc->hnm", u, u, M_uu/feat, eps))
        self.sow("debug", "uu", _uu)
        uu = softmax(_uu)

        if n_up > 0:
            dd = softmax(einsum("nfza,mgzb,hfgzc,abc->hnm", d, d, M_dd, eps))
            ud = softmax(einsum("nfza,mgzb,hfgzc,abc->hnm", u, d, M_ud, eps))
            du = softmax(einsum("nfza,mgzb,hfgzc,abc->hnm", d, u, M_du, eps))
            y = (zeros((n_e, out_dims, 3, 2))
                    .at[:n_up].set(   einsum("hnm,hgfza,mfzb,abc->ngzc",uu,A_uu,u,ksi)
                                    + einsum("hnm,hgfza,mfzb,abc->ngzc",ud,A_ud,d,ksi))
                    .at[n_up:].set(   einsum("hnm,hgfza,mfzb,abc->ngzc",dd,A_dd,d,ksi)
                                    + einsum("hnm,hgfza,mfzb,abc->ngzc",du,A_du,u,ksi))
                    )
        else:
            y = einsum("hnm,hgfza,mfzb,abc->ngzc",uu,A_uu,u,ksi)

        #y /= sqrt((y*y).sum(axis=-1,keepdims=True))

        return y




class Attention(Module):
    length: int
    head: int = 1
    n_up: int = 0

    @compact
    def __call__(self, x):
        n_e, feat, *rest = x.shape
        head, length, n_up = self.head, self.length, self.n_up

        qkd_shape = (head,length,feat)
        if n_up > 0:
            qkd_shape = (2,)+qkd_shape

        Q = self.param("Q",initializers.normal(1.), qkd_shape)
        K = self.param("K",initializers.normal(1.), qkd_shape)
        D = self.param("D",initializers.normal(1.), qkd_shape)

        normalize = lambda x: x/sqrt((x*x).sum((-1,-2,-3),keepdims=True))
        linear = lambda a,b: normalize(einsum("nf...,hqf->nhq...", a,b))

        if n_up == 0:
            q = linear(x,Q)
            k = linear(x,K)
            d = linear(x,D)
        else:
            q = concatenate((linear(x[:n_up],Q[0]),linear(x[n_up:],Q[1])))
            k = concatenate((linear(x[:n_up],K[0]),linear(x[n_up:],K[1])))
            d = concatenate((linear(x[:n_up],D[0]),linear(x[n_up:],D[1])))

        w = softmax(einsum("qh...,kh...->hqk", q,k), axis=-1)
        v = einsum("hqk,khd...->qhd...",w,d)
        v = v.reshape((n_e,head*length)+tuple(rest))

        return v


class Backflow(Module):
    @compact
    def __call__(self, x):
        '''
        x.shape = (n_e, n_feat, 3)
        '''
        pass


class Pairwise(Module):
    update: Callable

    @compact
    def __call__(self, x):
        n_e, n_in = x.shape
        pair_idx = (arange(n_e)[:,None] + arange(1,n_e)[None,:]) % n_e
        pairs = zeros((n_e, n_e-1, 2*n_in))\
                .at[:, :, :n_in].set(x[:,None,:])\
                .at[:, :, n_in:].set(x[pair_idx])
        y = mean(self.update(pairs), axis=1)
        return y

def slatter(n_up):
    f = lambda x:\
            slogdet(x[:n_up, :n_up])[1] + slogdet(x[n_up:, n_up:])[1]
    return f


class Sequential(Module):
    layers : List[Callable]

    @compact
    def __call__(self, x):
        for n, layer in enumerate(self.layers):
            x = layer(x)
            self.sow("sequential", f"{n}", x)
        return x

def periodic_embed(latvec, n_up):
    embed = lambda x: (zeros((x.shape[0], 7))\
        .at[:,0:3].set(cos(2*pi*x / latvec))\
        .at[:,3:6].set(sin(2*pi*x / latvec))\
        .at[:,  6].set(2*(arange(x.shape[0]) < n_up)-1))
    return embed

def periodic_embed_v2(latvec,n):
    r = (indices((n,n,n)) / n).reshape(3,-1).T * latvec
    def embed(x):
        x = x[:,None,:] + r
        return stack((cos(2*pi*x/latvec),sin(2*pi*x/latvec)),axis=-1)
    return embed

periodic_embed_complex = lambda latvec: lambda x: (
        stack((cos(2*pi*x/latvec), sin(2*pi*x/latvec)), axis=-1)
        .reshape(x.shape[0],1,3,2))


