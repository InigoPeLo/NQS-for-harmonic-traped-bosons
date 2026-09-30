"""VMC sampler module"""

import jax 
import jax.numpy as jnp
from dataclasses import dataclass

@dataclass(frozen=True)
class MetroSampler:
    """Class to define the Metropolis sampler for the VMC algorithm"""

    n_chains: int  #Number of Markov chains to run in parallel
    step_size: float = 0.4  #Step size for the Metropolis updates
    n_sweep: int = 10  #Metropolis steps per sweep (between two recorded samples)

    def init_walkers(self, key, n_particles, dim):
        """
        Initialize the walkers for the Metropolis sampler
        all walkers are initialized with a gaussian distribution with mean 0 and covariance I.

        """

        shape = (self.n_chains, n_particles, dim)

        walkers = jax.random.normal(key, shape = shape)

        return walkers

    def step(self, log_psi, carry, key):
        """One step of montecarlo for all chains at once.
        carry = (walkers, log_prob), with the form scan needs: (carry, key) -> (carry, accept)
        """

        walkers, log_prob = carry

        #First we propose the next step

        key_move, key_accept = jax.random.split(key)   #One key for the movement, the other for the acceptance 

        #Gaussian displacement for every coordinate of every chain
        xi = jax.random.normal(key_move, shape=walkers.shape)  # ξ ~ N(0, I)

        #This way we update all walkers at once
        proposal = walkers + self.step_size * xi # X' = X + δ ξ

        #Now we have to check if we accept it or not, first we calculate the new log_prob (log|psi|, without the 2)
        log_prob_new=jnp.real(jax.vmap(log_psi)(proposal))

        #Acceptance criterion

        #Random number between 0 and 1
        u = jax.random.uniform(key_accept, shape=(self.n_chains,))

        #Metropolis-Hastings criterion: accept with probability min(1, |psi'|^2/|psi|^2)

        accept= jnp.log(u) < 2*(log_prob_new-log_prob)

        #If the change is accepted we redefine the variables
        #accept[:, None, None] broadcasts each chain's decision to all its particles and coordinates
        walkers= jnp.where(accept[:, None, None], proposal, walkers)
        log_prob= jnp.where(accept, log_prob_new, log_prob)

        #We need to give back the arguments this way so scan can use them.
        return (walkers, log_prob), accept

    def sample(self, log_psi, walkers, n_samples, key):
        """Run the chains and record n_samples configurations per chain.
        """

        #First we calculate the starting log_prob

        log_prob = jnp.real(jax.vmap(log_psi)(walkers))

        #We do n_sweep mc steps and save the final configurations
        def one_sweep(carry, key):
            keys = jax.random.split(key, self.n_sweep)
            carry, accepts = jax.lax.scan(lambda c, k: self.step(log_psi, c, k), carry, keys)
            return carry, (carry[0], accepts)

        #We make as many samples as we desire

        keys=jax.random.split(key, n_samples)

        (walkers, log_prob), (samples, accepts)=jax.lax.scan(one_sweep, (walkers, log_prob), keys)

        #samples is (n_samples, n_chains, N, dim); we merge the first two axes
        #into one batch of configurations

        samples = samples.reshape(-1, *walkers.shape[1:])

        #We return mean accepts to check if the sampler is working as intended, we aim for values 
        #close to 50%
        return samples, walkers, jnp.mean(accepts)



       