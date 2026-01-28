import torch 
import numpy as np 
import os

from typing import Callable, Optional
import torch.optim as optim
from tqdm import tqdm

def evaluate_nll(
    log_weights: torch.Tensor,
    log_Pij: torch.Tensor,
) -> torch.Tensor:
    """
    Evaluate the negative log-likelihood of the data given the weights.

    Parameters
    ----------
    log_weights: torch.Tensor
        Log of the weights of the clusters.
    log_Pij: torch.Tensor
        Log-likelihood of generating image i from cluster j.
    cluster_size: torch.Tensor
        Number of images in each cluster
    anchor_strength: float
        Strength of the anchor term anchoring the average log weight to zero.
        (Prevents them going to infinity)

    Returns
    -------
    neg_total_ll: torch.Tensor

    """
    # Normalize the log weights
    weighted_alphas = normalize_weights(log_weights)
    log_weighted_alphas = torch.log(weighted_alphas)

    # Evaluate the log-likelihood
    likelihood_per_image = torch.logsumexp(log_Pij + log_weighted_alphas, axis=1)
    neg_total_ll = -1 * torch.mean(likelihood_per_image)
    return neg_total_ll

def normalize_weights(
    log_weights: torch.Tensor,
) -> torch.Tensor:
    """ """
    weighted_alphas = torch.exp(log_weights)
    weighted_alphas = weighted_alphas / torch.sum(weighted_alphas)
    return weighted_alphas

def fw_gap(weights, grad):
      """The Frank-Wolfe gap an upper bound on the optimality gap.

      the loss f (negative of the objective) is convex,
          f(y) >= f(x) + <f'(x), y - x>
      We can thus bound the optimality gap by
          f(x) - f(x*) <= - min_y <f'(x), y - x> : y in simplex
      and the RHS is minimized
      """
      # For a convex f, would be
      # -(np.min(grad) - np.inner(grad, param))
      # In our case, grad is the negative of the gradient, so
      # -(np.min(-grad) - np.inner(-grad, param))
      # simplifies to
      # np.max(grad) - np.inner(grad, param)
      # BUT, for ourproblem, np.inner(grad, param) is always 1
      return torch.max(grad) - 1

def grad_log_prob(
    weights: torch.Tensor,
    log_likelihood: torch.Tensor,
) -> torch.Tensor:
    """
    Evaluate the gradient of the log-likelihood of the data given the weights.

    Parameters
    ----------
    weights: torch.Tensor
        weights of the clusters.
    log_likelihood: torch.Tensor
        Log-likelihood of generating image i from cluster j.

    Returns
    -------
    grad: torch.Tensor

    """
    num_images, num_structures = log_likelihood.shape

    log_weights = torch.log(weights)
    print(log_weights.dtype)
    log_density_at_weights = torch.logsumexp(log_likelihood + log_weights, axis=1)

    aux = log_likelihood - log_density_at_weights.reshape(num_images, 1)
    grad =  (1/num_images)*(torch.exp(torch.logsumexp(aux, axis=0)))
    return grad


def multiplicative_gradient(
    log_likelihood,
    tol: Optional[float]=10**-4,
    max_iterations: Optional[int]=20000,
    stats_frequency: Optional[int]=1
)->float:
    
    """
     This function updates the weights according to the expectation maximization
     algorithm for mixture models.
     This is also known as the "multiplicative gradient" method, which has much less notation overload with "EM"!
     
     For $N$ images and $M$ structures, this updates a given weight m according to
     .. math::
         \alpha_m^{(\text{new})} = \frac{1}{N}\sum_{i=1}^N \frac{\alpha_m p(y_i|x_m)}{\sum_{m'}\alpha_{m'} p(y_i|x_{m'})}

    This actually simplifies to, literally multiplying the old guess by the gradient of the log likelihood
     .. math::
         \alpha_m^{(\text{new})} = \alpha_m \nabla L(\alpha)_m,
    where the L(\alpha) is the log likelihood at the old weights

    This is implemented with logarithms of the above equation, for stability.

    By default, the initial weights are set to equal probabilities for all structures, the `most entropic' weights.
 
    Parameters
    ----------
    log_likelihood: torch.Tensor
        Log-likelihood of generating image i from cluster j.
    tol: float
        Tolerance for the stopping criteria
    max_iterations: int
        Max iterations if stopping criteria isn't met
    stats_frequency: int:
        Stats are computed at every (stats frequency) iterations
    
    Returns
    -------
    weights: torch.tensor 
    stats_tracking: dictionary
    """
    num_images, num_structures = log_likelihood.shape

    # Initialize Weights
    weights = (1/num_structures)*torch.ones(num_structures,dtype=torch.float64)
    
    stats_tracking = {}
    stats_tracking["losses"] = []
    stats_tracking["entropies"] = []
    stats_tracking["idx"] = []

    # Iterate
    for k in range(max_iterations):

        # Update weights
        grad = grad_log_prob(weights, log_likelihood)   
        weights = weights*grad

        # Check stopping criterion
        gap = fw_gap(weights,grad)
        if k % stats_frequency == 0: 
            log_weights = torch.log(weights)
            loss = -torch.mean(torch.logsumexp(log_likelihood + log_weights, axis=1))
            entropy = -torch.sum(weights*log_weights)
            stats_tracking["losses"].append(loss)
            stats_tracking["entropies"].append(entropy)
            stats_tracking["idx"].append(k)
            print(f"#iterations: {k}")
            print(f"loss: {loss}")
            print(f"frank-wolfe gap: {gap}")
            print(f"entropy: {entropy}")
            print("\n")
        
        if gap < tol:
            print("exiting!")
            print(f"#iterations at exit: {k}")
            break

    log_weights = torch.log(weights)
    log_weights = torch.log(normalize_weights(log_weights))
    return log_weights, stats_tracking


def reweighting_wrapper (
    likelihoods_matrix,
    cross_validation_sets=5,
    random_seed=42,
    shuffle=False
    ):

    np.random.seed(random_seed)
    print(likelihoods_matrix.dtype)

    if likelihoods_matrix.dtype == torch.float32:
        likelihoods_matrix = likelihoods_matrix.to(torch.float64)
        print('here')

    print(likelihoods_matrix.dtype)
    

    if shuffle == True:
        np.random.shuffle(likelihoods_matrix)
    particle_sets = np.array_split(likelihoods_matrix,cross_validation_sets)
    weights_array = torch.zeros([cross_validation_sets,likelihoods_matrix.shape[1]])

    for i  in range(cross_validation_sets):
        weights_array[i], __ =  multiplicative_gradient (particle_sets[i])

         
    return weights_array




def reweighting (
    likelihoods_directory,
    opt=False,
    phys=False,
    integrated=False,
    cross_validation_sets=5,
    random_seed=42,
    ignore_particles=None,
    ignore_templates=None
    ):

    matrices_directory = os.path.join(likelihoods_directory, 'concatenated_matrices')

    if opt == True:
        opt_likelihoods = torch.load(os.path.join(matrices_directory, 'optimal_fourier_log_likelihood_matrix.pt'), weights_only=False).T
        weights_array = reweighting_wrapper (opt_likelihoods,cross_validation_sets=cross_validation_sets,random_seed=random_seed) 

        print("Using the optimal calculated posse and displacement, the relative weights between templates are.")
        print(torch.exp(weights_array))

        torch.save(weights_array,os.path.join(likelihoods_directory,'opt_weights.pt'))

    if integrated == True:
        integrated_likelihoods_filename = os.path.join(matrices_directory, 'integrated_fourier_log_likelihood_matrix.pt')
        print(integrated_likelihoods_filename)
        integrated_likelihoods = torch.load(integrated_likelihoods_filename,weights_only=False).T

        print(integrated_likelihoods.shape)
        weights_array = reweighting_wrapper (integrated_likelihoods,cross_validation_sets=cross_validation_sets,random_seed=random_seed) 

        print("Using the marginalised likelihoods, the relative weights between templates are.")
        print(torch.exp(weights_array))

        torch.save(weights_array,os.path.join(likelihoods_directory,'integrated_weights.pt'))

    if phys==True:
        print('physical likelihoods not supported don\'t use them')
