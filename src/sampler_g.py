"""VMC sampler using Gibbs Samplings"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from src.nqs import DSE

@dataclass(frozen=True)
class GibbsSampler:
    """Class to define the Gibbs sampler for the VMC algorithm for it's use in the NQS"""
    n_chains: int  #Number of Markov chains to run in parallel
    step_size: float = 0.4  #Step size for the Metropolis updates
    n_sweep: int = 10  #Gibbs steps per sweep (between two recorded samples)
    n_metro: int = 1 # Metropolis steps inside each Gibbs step

    def init_walkers(self, key, n_particles, dim):
        """
        Initialize the walkers for the Gibbs sampler
        all walkers are initialized with a gaussian distribution with mean 0 and covariance I.
    
        """
    
        shape = (self.n_chains, n_particles, dim)
    
        walkers = jax.random.normal(key, shape = shape)
    
        return walkers

    def hidden_sample(self, params, walkers, key):
        """Sample the hidden units h ~ p(h|x) for all chains at once
        p(h_j=+1|x)=sigmoid(2 theta_j). independently for every j
        theta=b+WH_global
        returns h: shape (n_chains, M) with values +-1"""

        params_rbm=params["params_rbm"]

        #The visibles for the RBM is the H vector so we need to encode the positions
        F, dim = params["params_dse"]["W"].shape
        dse = DSE(n_neurons=F, dim=dim)
        H_global=jax.vmap(lambda x: dse.apply(params["params_dse"], x))(walkers)

        #Now we can calculate theta:

        theta=params_rbm["b"]+H_global @ params_rbm["W"].T

        #Now we need to sample h.Each hidden unit is +1 with probability sigmoid(2theta) and -1 otherwise

        u=jax.random.uniform(key, shape=theta.shape)

        h=jnp.where(u <jax.nn.sigmoid(2*theta), 1.0, -1.0)

        return h

    def log_p1(self, params, c, walkers):
        """Log of the one particle conditional p1(y|c), without normalization, for every particle
        of every chain: log p1(y) = c·swish(W_d y + b_d) - 2 alpha |y|^2
        Given h the particles are independent and all follow p1. Metropolis only needs differences
        of log p1, so the normalization is not needed.
        c: shape (n_chains, F), c = a + h W, computed in step
        walkers: shape (n_chains, N, dim), current or proposed positions
        returns: shape (n_chains, N)"""
        alpha=jax.nn.softplus(params["alpha_tilde"])

        #We need to calculate the Deep set features of every particle
        params_dse=params["params_dse"]

        features = jax.nn.swish(walkers @ params_dse["W"].T + params_dse["b"])

        #c·features for every particle. c[:, None, :] repeats the c of each chain for all its particles

        rbm_term = jnp.sum(c[:, None, :] * features, axis=-1)

        #Gaussian envelope of every particle: (n_chains, N)
        envelope = 2 * alpha * jnp.sum(walkers**2, axis=-1)

        return rbm_term - envelope

    def metro_step(self, params, c, carry, key):
        """One Metropolis step of every particle of every chain at once, sampling p(x|h) with c fixed.
        Proposal y' = y + step_size*xi with xi ~ N(0, I) for every particle, accepted with probability
        min(1, p1(y')/p1(y)). Given h the particles are independent, so each particle is accepted or
        rejected on its own (accept has shape (n_chains, N), not (n_chains,) as in MetroSampler).
        carry = (walkers, log_p) with log_p = log_p1 of the current walkers, shape (n_chains, N),
        so it has the (carry, key) -> (carry, accept) form that lax.scan needs.
        returns: carry=(walkers, log_p), accept"""

        walkers, log_prob = carry

        #we propose the next step, we need a key for the move proposition and other for the acceptance

        key_move, key_accept = jax.random.split(key) 

        xi=jax.random.normal(key=key_move, shape=walkers.shape)
        #Move proposal
        proposal= walkers + self.step_size*xi
        #We need to evaluate log_p1 in the new position
        log_prob_new=self.log_p1(params, c, proposal)

        #We need to check for acceptance for each individual particle
        u=jax.random.uniform(key_accept, shape=log_prob_new.shape)
        #Accept criterion

        accept=jnp.log(u) < log_prob_new - log_prob

        #we only accept in those particles in which accept=true
        walkers = jnp.where(accept[:, :, None], proposal, walkers)
        log_prob = jnp.where(accept, log_prob_new, log_prob)
        
        return (walkers, log_prob), accept

    def step(self, params, walkers, key):
        """One Gibbs step for all chains:
        1. sample the hidden units h ~ p(h|x) with hidden_sample
        2. compute the effective vector c = a + h W, shape (n_chains, F)
        3. compute log_p1 of the current walkers with this c (c changed, so the old value is not valid)
        4. do n_metro metro_step with c fixed (lax.scan), which samples x ~ p(x|h)
        returns: walkers, mean acceptance of the Metropolis moves"""

        #Split key
        key_h, key_m =jax.random.split(key)
        #First we sample h with hidden_sample

        h = self.hidden_sample(params=params, walkers=walkers, key=key_h)

        #Now we need to calculate the effective vector c

        params_rbm=params["params_rbm"]

        c=params_rbm["a"]+ h @ params_rbm["W"] 

        #log_p1 of the current walkers with the new c (the one from the previous Gibbs step used another c)
        log_prob=self.log_p1(params=params, c=c, walkers=walkers)

        #n_metro Metropolis steps with params and c fixed, which samples x ~ p(x|h)
        keys = jax.random.split(key_m, self.n_metro)
        (walkers, log_prob), accepts = jax.lax.scan(lambda carry, k: self.metro_step(params=params, c=c, carry=carry, key=k), (walkers, log_prob), keys)

        #accepts has shape (n_metro, n_chains, N): one decision per particle and Metropolis step
        return walkers, jnp.mean(accepts)

    def sample(self, params, walkers, n_samples, key):
        """Run the chains and record n_samples configurations per chain, as MetroSampler.sample.
        Each recorded sample is taken after n_sweep Gibbs steps (lax.scan of step).
        It takes params (the dictionary unravel(theta)) instead of log_psi, because it needs the
        encoder, the RBM and alpha separately.
        returns: samples of shape (n_samples*n_chains, N, dim) in sweep-major order (check_therm
        relies on it), the final walkers and the mean acceptance"""

        #one_sweep (like in metrosampler): n_sweep Gibbs steps, then we record the configuration.
        #It has the (carry, key) -> (carry, output) form of lax.scan, with params taken from sample
        def one_sweep(walkers, key):
            keys = jax.random.split(key, self.n_sweep)
            walkers, accepts = jax.lax.scan(lambda w, k: self.step(params, w, k), walkers, keys)
            return walkers, (walkers, accepts)

        #Key split for the n samples
        keys= jax.random.split(key, n_samples)
        walkers, (samples, accepts) = jax.lax.scan(one_sweep, walkers, keys)

        #samples is (n_samples, n_chains, N, dim); we merge the first two axes into one batch of
        #configurations, keeping the sweep-major order that check_therm needs
        samples = samples.reshape(-1, *walkers.shape[1:])

        return samples, walkers, jnp.mean(accepts)




    def check_therm(self, samples, z_max=3.0):
        """Check if the chains have thermalized: the mean of sum_i |x_i|^2 must not drift between the
        third and the last quarter of the samples. Since the chains are independent, the spread of the
        per-chain drifts gives its error bar. Returns (thermalized, z) with z the drift in sigmas."""

        #sum_i |x_i|^2 per sample, splitting again sweeps and chains: (n_samples, n_chains)
        r2 = jnp.sum(samples**2, axis=(1, 2)).reshape(-1, self.n_chains)

        if r2.shape[0] < 4:
            raise ValueError(f"check_therm needs at least 4 samples per chain, got {r2.shape[0]}: increase n_thermalization")

        q = r2.shape[0] // 4 #We ignore the first 2 quarters of the samples,
        d = jnp.mean(r2[3*q:], axis=0) - jnp.mean(r2[2*q:3*q], axis=0)  #drift of every chain
        z = jnp.mean(d) / (jnp.std(d) / jnp.sqrt(self.n_chains))

        #We aim for a z smaller than 3, which means that the drift is smaller than 3 sigmas. 
        
        return jnp.abs(z) < z_max, z    
