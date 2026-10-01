"""Here we define all the three neural networks we need to use,
RBM for the amplitud
FFNN for the phase
Deep Sets encoder"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass

#First we define the deepsets encoder (DSE=Deep Sets Encoder)

@dataclass(frozen=True)
class DSE:
    n_neurons: int #F
    dim: int=2

    def init(self, key):
        """ Initialize de encoder parameters with W following a distribution given by N(0,1/sqrt{dim})
        of shape (n_neurons, dim) and b=0 of shape (n_neurons,)"""

        W_shape=(self.n_neurons, self.dim)

        W=jax.random.normal(key,shape=W_shape)/ jnp.sqrt(self.dim)
        b=jnp.zeros(shape=self.n_neurons)

        #We have to return the parameters as a dictionary so we can save them. It wouldn't work if we saved 
        # in self.
        return {"W": W, "b": b}


    def apply(self, params, x):
        """Apply the the Deep Sets encoder to one configuration (x)"""
        #We need to transpond "W" so we can write the opreation as a scalar product between vectors
        h=jax.nn.swish(x @ params["W"].T + params["b"])

        H_global=jnp.sum(h, axis=0)

        return H_global

#Definition of the RBM used for the amplitude

@dataclass(frozen=True)
class RBM:
    n_visible: int #F
    n_hidden: int #M
    init_scale: float
  
    
    def init(self, key):
        """Initialization of RBM parameters"""


        a=jnp.zeros(shape=self.n_visible)
        b=jnp.zeros(shape=self.n_hidden)

        W_shape=(self.n_hidden, self.n_visible)

        W=self.init_scale* jax.random.normal(key, shape=W_shape)

        return {"W": W, "b": b, "a": a}

    def apply(self, params, H_global):
        """Apply the RBM to the input H"""

        #Visible term

        visible_term=jnp.dot(H_global, params["a"])

        #Hidden term, we are first calculating what I called theta_j in eq (58)

        theta=params["b"]+jnp.dot(params["W"], H_global)
        #cosh diverges for large arguments, so we have to use the identity cosh(x)=0.5*(exp(x)+exp(-x))
        # we can ignore the log 2 term becaue it only changes the normalization of the wave function,
        # which we do not care about 
        hidden_term=jnp.sum(jnp.logaddexp(theta, -theta))

        return visible_term + hidden_term


@dataclass(frozen=True)
class FFNN:
    n_visible: int #F
    n_hidden: int #K
    

    def init(self, key):
        """ Initialization of the FFNN parameters"""

        c=jnp.zeros(shape=self.n_hidden)
        #We neet 2 keys one for u and other for V
        key_V, key_u=jax.random.split(key)

        V=jax.random.normal(key_V, shape=(self.n_hidden, self.n_visible))/ jnp.sqrt(self.n_visible)
        u=jax.random.normal(key_u, shape=(self.n_hidden,)) /jnp.sqrt(self.n_hidden)

        return {"V": V, "u": u, "c": c}

    def apply(self, params, H_global):
        """Apply the FFNN to the input H"""
        
        #Visible term

        thing=jnp.dot(params["V"], H_global) + params["c"]

        #Hidden term in eq (64), we can ignore the log 2 term because it gives a global phase to the wave function, which we do not care about

        h=jnp.logaddexp(thing, -thing)

        hidden_term=jnp.dot(params["u"], h)

        return hidden_term


#We build the whole model now

@dataclass(frozen=True)
class NQS:
    n_visible: int #F
    n_hidden_rbm: int #M
    n_hidden_ffnn: int #K
    alpha: float #Gaussian envelope parameter
    init_scale: float=0.01


    def init(self, key):
        """Initialization of the NQS parameters"""

        #We need 3 keys, one for each neural network
        
        key_rbm, key_ffnn, key_dse=jax.random.split(key, 3)

        #We need to initialize the parameters of each neural network
        dse=DSE(n_neurons=self.n_visible)
        params_dse=dse.init(key_dse)

        rbm=RBM(n_visible=self.n_visible, n_hidden=self.n_hidden_rbm, init_scale=self.init_scale)
        params_rbm=rbm.init(key_rbm)

        ffnn=FFNN(n_visible=self.n_visible, n_hidden=self.n_hidden_ffnn)
        params_ffnn=ffnn.init(key_ffnn)

        #Gaussian envelope -alpha*sum_i |x_i|^2 (section 6.2.1). The RBM grows at most linearly with |x|
        #(swish and log cosh are linear for large arguments), so the envelope is what makes psi decay
        #and be normalisable. This only works if alpha > 0: with alpha < 0 psi would grow with |x| and the
        #walkers would drift to infinity. Since SR can push a parameter to any value, we do not train alpha
        #directly but alpha_tilde, with alpha = softplus(alpha_tilde) = log(1 + e^alpha_tilde) > 0 always.
        #To start from alpha = self.alpha we invert softplus: alpha_tilde = log(e^alpha - 1).

        alpha_tilde = jnp.log(jnp.expm1(self.alpha))   # expm1(x) = e^x - 1

        return {"params_dse": params_dse, "params_rbm": params_rbm, "params_ffnn": params_ffnn, "alpha_tilde": alpha_tilde}

    def apply(self, params, x):
        """Apply the whole NQS to the input x, returning f (logarithm of the wave function)"""

        #We first need to apply the Deep Sets encoder to x, which gives us H_global
        dse=DSE(n_neurons=self.n_visible)
        H_global=dse.apply(params["params_dse"], x)

        #Now we apply the RBM to H_global 

        rbm=RBM(n_visible=self.n_visible, n_hidden=self.n_hidden_rbm, init_scale=self.init_scale)
        log_psi_rbm=rbm.apply(params["params_rbm"], H_global)

        #And lastly the phase FFNN to H_global
        ffnn=FFNN(n_visible=self.n_visible, n_hidden=self.n_hidden_ffnn)
        phase=ffnn.apply(params["params_ffnn"], H_global)

        #We also need to add the Gaussian envelope to log|psi|, which is -alpha*sum_i |x_i|^2
        alpha=jax.nn.softplus(params["alpha_tilde"])
        log_psi=log_psi_rbm - alpha*jnp.sum(x**2)

        #We build the final log psi

        f=log_psi + 1j*phase

        return f
    
 


