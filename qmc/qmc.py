from jax import (jit, vmap, grad, pmap, linearize, local_device_count,
         value_and_grad)
from jax.numpy import (exp, where, eye, mean, std, einsum, indices, pi,
        triu_indices, array, cos, zeros, abs, savez, load, isfinite, sqrt)
from jax.tree_util import  (tree_structure, tree_map, tree_leaves,
                            tree_all)
from jax.lax import fori_loop, pmean
from jax.debug import print as jprint
from jax.random import split, normal, uniform, PRNGKey
from jax.numpy.linalg import norm, det, inv
from jax.scipy.special import erfc

from folx import forward_laplacian

from functools import partial, cache

_pmap  = partial(pmap , axis_name='device')
_pmean = partial(pmean, axis_name='device')

class QMC:
    def __init__(self, model, potential, n_e, batch,
                 key=PRNGKey(0), n_device=None, mc_step_size=.05,
                 mc_n_step=30, mc_show_acc=False, mc_warmup=300):
        """
        Arguments:
            model  : flax module instance representing log psi.
            potential: callable, potential(r) where r.shape=(n_e, 3)
            n_e    : number of electron
            batch  : batch size
            key    : the PRNGKey
            n_device : number of device
            mc_*   : arguments for monte carlo samplers.
        """
        self.key = key
        self.n_e = n_e
        self.warmup = mc_warmup
        self.n_step = mc_n_step
        self.step_size = mc_step_size
        self.n_device = n_device if n_device else local_device_count()

        self.param, self.log_psi, self.log_psi_kfac = handle_model(
                model, n_e, self.get_key()[0])

        self.E_local = make_E_local(self.log_psi, potential)
        self.sampler = make_sampler(self.log_psi, mc_show_acc)

        self.get_trainer = cache(
                partial(make_trainer, self.log_psi, self.E_local))

        self.E_mean = _pmap(
                jit(lambda p,r:_pmean(mean(self.E_local(p,r)))),
                in_axes=(None,0), out_axes=None)

        self.r = normal( self.get_key()[0],
                (self.n_device, batch // self.n_device, self.n_e, 3))

        self.is_warmed_up = False
        

    def get_key(self, n=1):
        key = split(self.key, n+1)
        self.key = key[0]
        return key[1:]

    def sample(self, n_step, step_size=None):
        key = self.get_key(self.n_device)
        step_size = self.step_size if not step_size else step_size
        self.r = self.sampler(self.param, self.r, key, step_size, n_step)

    @property
    def energy(self):
        if not self.is_warmed_up:
            self.sample(self.warmup)
            self.is_warmed_up = True
        return self.E_mean(self.param, self.r)

    def update(self, alp, metric=None):
        if not self.is_warmed_up:
            self.sample(self.warmup)
            self.is_warmed_up = True
        else:
            self.sample(self.n_step)
        delta, E_mean = self.get_trainer(metric)(self.param, self.r)
        if not tree_all(tree_map(lambda x: isfinite(x).all(), delta)):
            return None
        self.param = tree_map(lambda x,dx: x-dx*alp, self.param, delta)
        return E_mean

    def save(self, filename):
        savez(filename, *tree_leaves(self.param))

    def load(self, filename):
        param = list(load(filename).values())
        self.param = tree_structure(self.param).unflatten(param)
        self.is_warmed_up = False


def make_trainer(log_psi, E_local, metric=None):
    grad_log_psi = vmap(grad(log_psi), in_axes=(None,0))
    _mean = lambda x: _pmean(mean(x,axis=0))

    if metric == None:
        func = lambda coef: lambda x: _pmean(einsum('i,i...',coef,x)) 
    elif metric == 'diag':
        func = lambda coef: lambda x: (
            _pmean(einsum('i,i...',coef,x)) / _mean((x-_mean(x))**2))
    else:
        raise ValueError(f"unknown metric approximation {metric}")

    @partial(_pmap, in_axes=(None,0), out_axes=None)
    @jit
    def update(param, r):
        E_l = E_local(param, r)
        E_mean = _pmean(mean(E_l))
        coef = (E_l-E_mean) / r.shape[0]
        delta = tree_map(func(coef), grad_log_psi(param,r))
        return delta, E_mean

    return update

def make_potential(R=None,Z=None,latvec=None):
    """
    latvec is None if not periodic, float if cubic lattice
            or [a1,a2,a3] be three lattice vectors.
    """
    if latvec==None:
        V = lambda r: 1/norm(r, axis=-1)
        E_0 = 0.
    else:
        if type(latvec)==float:
            latvec = eye(3) * latvec
        latvec = array(latvec)
        invlat = inv(latvec)
        recvec = inv(latvec).T * 2*pi
        latvol = abs(det(latvec))
        kappa = pi**.5 / latvol**(1/3)
        idx = indices((3,3,3)).reshape(3,-1).T - 1
        idx_nonzero = idx[~(idx==0).all(axis=-1)]
        lattice = idx_nonzero @ latvec
        k       = idx_nonzero @ recvec
        k2 = (k**2).sum(axis=-1)
        coef = exp(-k2/(4*kappa**2)) / k2
        def V(r):
            r = ((r @ invlat + .5) % 1 - .5) @ latvec
            r = r[..., None, :]
            d = norm(r - lattice, axis=-1)
            v = ( (erfc(kappa*d) / d).sum(axis=-1) 
             + (cos((k*r).sum(axis=-1)) * coef).sum(axis=-1) *4*pi/latvol
             - pi / (latvol*kappa**2) )
            return v
        E_0 = (V(array([.0,.0,.0])) - 2*kappa/pi**.5) / 2
        lattice = idx @ latvec
    if R==None:
        VrR = lambda r: 0
    else:
        R,Z = map(array, (R,Z))
        VrR = lambda r: -(V(r[...,None,:]-R)*Z).sum(axis=(-1,-2))
    def potential(r):
        n_e = r.shape[-2]
        i,j = triu_indices(n_e, 1)
        rr = r[..., i, :] - r[..., j, :]
        return V(rr).sum(axis=-1) + VrR(r) + n_e * E_0
    return jit(potential)


def make_E_local(log_psi, potential):
    potential = vmap(potential)
    #lapl = ForwardLaplacianOperator(6)(lambda r: log_psi(param, r))
    #def E_local(param, r):
    #    fwd_lapl = forward_laplacian(lambda r: log_psi(param, r), 6)
    #    @vmap
    #    def kinetic(r):
    #        lapl = fwd_lapl(r)
    #        return -.5*(lapl.laplacian + (lapl.jacobian.data**2).sum())
    #    return kinetic(r) + potential(r)
    #return jit(E_local)

    #    #n_e = r.shape[-2]
    #    #grad_log_psi = grad(lambda r: log_psi(param, r.reshape(n_e,3)))
    #    f = lambda r: log_psi(param, r)
    #    grad_log_psi = grad(f)
    #    lapl_log_psi = lapl(6)(f)
        #lapl_log_psi = forward_laplacian(lambda r: log_psi(param, r))
        #I = eye(n_e*3)
    #    kinetic = vmap(lambda r: -.5*(
    #            lapl_log_psi(r).laplacian + (grad_log_psi(r)**2).sum()))
        #@vmap
        #def kinetic(r):
        #    lapl = lapl_log_psi(r).laplacian
        #    return -.5 * (lapl + grad_log_psi(r)
        #    #val, hess = linearize(grad_log_psi, r.flatten())
        #    #def acc(i, res):
        #    #    return res + hess(I[i])[i]
        #    #return -.5 * (fori_loop(0, n_e*3, acc, 0.) + (val**2).sum())
    #    return kinetic(r) + potential(r)
    #return jit(E_local)
    def E_local(param, r):
        n_e = r.shape[-2]
        grad_log_psi = grad(lambda r: log_psi(param, r.reshape(n_e,3)))
        I = eye(n_e*3)
        @vmap
        def kinetic(r):
            val, hess = linearize(grad_log_psi, r.flatten())
            def acc(i, res):
                return res + hess(I[i])[i]
            return -.5 * (fori_loop(0, n_e*3, acc, 0.) + (val**2).sum())
        return kinetic(r) + potential(r)
    return jit(E_local)


def make_sampler(log_psi, show_acc=False):
    log_psi = vmap(log_psi, in_axes=(None,0))
    def sampler(param, r, key, step_size, n_step):
        batch = r.shape[:-2]
        log_p = log_psi(param, r)*2
        def step(i,rpk):
            r,p,k = rpk
            k,k1,k2 = split(k,3)
            _r = r + normal(k1, r.shape) * step_size
            _log_p = log_psi(param, _r)*2
            chi = uniform(k2, batch)
            acc = exp(_log_p - p) > chi
            if show_acc:
                jprint("acc rate {}", mean(acc))
            r = where(acc[...,None,None], _r, r)
            p = where(acc, _log_p, p)
            return r,p,k
        r,_,_ = fori_loop(0, n_step, step, (r,log_p,key))
        return r
    return pmap(jit(sampler), in_axes=(None,0,0,None,None))

def handle_model(model, n_e, key):
    param = model.init(key, zeros((n_e, 3)))
    perturb = param['perturbations']
    def func(params, r, perturb, mut):
        p = {'params': params, 'perturbations': perturb}
        return model.apply(p,r,mutable=mut)
    log_psi = lambda p,r: model.apply({'params':p},r)
    _log_psi_kfac = value_and_grad(func, argnums=2, has_aux=True)
    def log_psi_kfac(p,r):
        ((_log_psi, a), e) = _log_psi_kfac(p,r,perturb,'intermediate')
        a = a['intermediate']
        a = tree_map(lambda x:x['a'][0], a,
                     is_leaf=lambda x:x.keys()=={'a'})
        e = tree_map(lambda x:x['e'], e,
                     is_leaf=lambda x:x.keys()=={'e'})
        return _log_psi, (a,e)
    return param['params'], jit(log_psi), jit(log_psi_kfac)





