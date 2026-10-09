#Here we are going to define the problem we aim to solve with our NQS.

import jax.numpy as jnp
import jax
from dataclasses import dataclass

@dataclass(frozen=True)
class boson_trap:
    """Class to define the boson trap without interactions."""
    n_particles: int   #Number of particles in the system
    dim: int=2 #Dimension of the system (We use 2D for now)
    omega: float = 1.0 #Frequency of the harmonic trap
    e_loc_batch: int = 256   #samples whose local energy is computed at the same time
    lap_batch: int = 32      #coordinates whose second derivative is computed at the same time, per sample

    @property
    #We define this property to use it as a sanity check for our NQS.
    def exact_energy(self):
        """
        Exact ground state energy of the non-interacting boson trap.
        E_0 = N * dim * omega / 2
        """
        return self.n_particles * self.dim * self.omega / 2    

    def trap_potential(self, x):
        """
        Potential energy of the harmonic trap. 
        V(x) = 1/2 omega^2 sum_i |x_i|^2, x of shape (N, dim).
        """
        return 0.5 * self.omega**2 * jnp.sum(x**2)

    def kinetic_local(self, log_psi, x):
        """
        Local kinetic energy of the system. 
        T(x) = -1/2 sum_i (nabla_i^2 log(psi) + (nabla_i log(psi))(nabla_i log(psi)))
        """
        #We get the shape of x and we define a flat version of it, with the n = N*dim coordinates
        shape=x.shape
        x_flat = x.reshape(-1)
        n = x_flat.shape[0]

        #Both parts of log_psi = u + i phi in one real vector g = [u, phi], shape (2,). jax.grad needs a real
        #scalar, but jax.jacrev accepts a real vector, so the derivatives of u and phi come out of the same pass
        #through the network instead of one pass for each
        def g(xf):
            f = log_psi(xf.reshape(shape))
            return jnp.stack([jnp.real(f), jnp.imag(f)])

        #linearize evaluates the gradient at x_flat, shape (2, n) (row 0: grad u, row 1: grad phi), and returns
        #hvp(v) = Hessian · v for any direction v, reusing the work of that evaluation
        grad_g, hvp = jax.linearize(jax.jacrev(g), x_flat)
        grad_u, grad_phi = grad_g[0], grad_g[1]

        #The laplacian only needs the diagonal of the Hessian: d^2/dx_k^2 is the component k of hvp(e_k).
        #We never build the n x n Hessian: lax.map computes lap_batch directions at a time, so the memory
        #does not grow as n^2 per sample
        second_derivative = lambda k: hvp(jax.nn.one_hot(k, n, dtype=x_flat.dtype))[:, k]   #(2,)
        lap_u, lap_phi = jnp.sum(jax.lax.map(second_derivative, jnp.arange(n), batch_size=self.lap_batch), axis=0)

        #The first term in the sumatory is just the laplacian of log_psi
        #Second term is given by the following expression: (nabla_i log(psi))(nabla_i log(psi)) = (nabla_i u)^2 - (nabla_i phi)^2 +2i (nabla_i u)(nabla_i phi)

        lap_f_sq=(grad_u @ grad_u -grad_phi @ grad_phi) + 2j * (grad_u @ grad_phi)

        return -0.5 * (lap_u+1j * lap_phi + lap_f_sq)

    def local_energy(self, log_psi, x):
        """
        Local energy of the system. 
        E(x) = T(x) + V(x)
        """
        return self.kinetic_local(log_psi, x) + self.trap_potential(x)

    def batch_local_energy(self, log_psi, x):
        """
        Local energy of the system for a batch of configurations. 
        E(x) = T(x) + V(x)
        """
        #We process the samples in batches of e_loc_batch instead of all at once, so the memory of the second
        #derivatives in kinetic_local stays bounded for large N and many samples
        return jax.lax.map(lambda xi: self.local_energy(log_psi, xi), x, batch_size=self.e_loc_batch)


