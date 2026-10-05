""" Stochastic Reconfiguration optimizer module """

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from dataclasses import dataclass

#Function to flatten the parameters of the model
def flatten_params(params):
    theta, unravel = ravel_pytree(params)
    return theta, unravel

#Function to compute the log_derivatives
def log_derivatives(theta, log_psi, samples):
    #jax.grad needs a real output, but log_psi is complex. Since theta is real we can differentiate
    #the real and imaginary parts separately: d log_psi = d Re(log_psi) + i d Im(log_psi)
    def O_single(x):
        #We calculate the log_derivative for a single sample x
        grad_re=jax.grad(lambda t: jnp.real(log_psi(t, x)))(theta)
        grad_im=jax.grad(lambda t: jnp.imag(log_psi(t, x)))(theta)
        return grad_re + 1j * grad_im

    #We can vectorize the O_single function to compute the log_derivatives for all samples at once
    O = jax.vmap(O_single)(samples)
    return O

def compute_S_F(O, E_loc):
    """Compute the S matrix and F vector for the SR update"""
    #We can compute the S matrix and F vector using the log_derivatives O and the local energies E_loc
    #We can use the fact that the S matrix is the covariance matrix. jnp.cov conjugates the second
    #factor, so we have to conjugate again to get the correct covariance matrix
    #S = <O* O> - <O*> <O>
    S=jnp.conj(jnp.cov(O, rowvar=False, bias=True))
    #F = <O* E_loc> - <O*> <E_loc> (it's the covariance between O and E_loc)
    #E_loc[:, None] turns E_loc (Ns,) into a column (Ns, 1), so jnp.cov treats it as one more variable.
    #A 1D array would be read as a row (1, Ns) and the shapes would not match
    Cov_OE=jnp.conj(jnp.cov(O, E_loc[:, None], rowvar=False, bias=True))

    #Cov_OE is the (p+1, p+1) covariance matrix of the variables (O_1, ..., O_p, E_loc):
    #the top-left block [:-1, :-1] is S, the last column [:-1, -1] is Cov(O_k, E_loc) = F
    #and the corner [-1, -1] is Var(E_loc). We take the last column without the corner.
    F=Cov_OE[:-1, -1]

    #Maybe I could have used Cov_OE to get S

    return S, F
@dataclass(frozen=True)
class SR:
    """ SR optimizer class, works on the flat parameters vector of the model NQS (theta)
    the log_psi(theta,x) function and a value of theta for the current parameters.
    """
    learning_rate: float
    varepsilon: float=1e-4 #S diag regularization parameter

    def step(self, theta, log_psi, samples, E_loc):
        """Compute the Stochastic Reconfiguration update for the parameters theta given
        the log_psi function, samples and local energies E_loc"""

        #First we need to compute the log_der O for our current set of parameters (theta is the 
        #flat parameters vector of the model)

        O=log_derivatives(theta, log_psi, samples)

        #Then we compute the S matrix and F vector
        S, F=compute_S_F(O, E_loc)

        #We only need the real part of S and F, since the parameters are real.
        S_re=jnp.real(S)
        F_re=jnp.real(F)

        #We need to solve the system (Re S+ varepsilon I) delta_theta = -learning_rate * Re F for delta_theta

        A=S_re + self.varepsilon * jnp.eye(S_re.shape[0])
        b=-self.learning_rate * F_re

        delta_theta=jnp.linalg.solve(A, b)

        #We need to calculate all the sanity checks of the update
        E_loc_mean=jnp.real(jnp.mean(E_loc))
        E_loc_var=jnp.var(E_loc)

        theta_new=theta + delta_theta

        return theta_new, [E_loc_mean, E_loc_var]




