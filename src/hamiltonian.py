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
        #We get the shape of x and we define a flat version of it to use in the hessian.
        shape=x.shape
        x_flat = x.reshape(-1)

        #We separate the real and imaginary parts of the log_psi to compute the kinetic energy.
        u= lambda xf: jnp.real(log_psi(xf.reshape(shape)))
        phi= lambda xf: jnp.imag(log_psi(xf.reshape(shape)))

        #We calculate the gradients and laplacians of the real and imaginary parts of log_psi.
        grad_u= jax.grad(u)(x_flat)
        grad_phi= jax.grad(phi)(x_flat)

        #The laplacian is the trace of the hessian, so we compute the hessians and take their traces.
        lap_u=jnp.trace(jax.hessian(u)(x_flat))
        lap_phi=jnp.trace(jax.hessian(phi)(x_flat))

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
        #We use vmap to vectorize the local energy calculation over the batch dimension.
        return jax.vmap(lambda xi: self.local_energy(log_psi, xi))(x)


