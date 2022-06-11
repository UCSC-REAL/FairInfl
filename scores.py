from .gradients import flatten_jacobian
from .metrics import constraints, cross_entropy_loss, hinge_loss
import flax.linen as nn
from jax import jacrev, jit, vmap
import jax.numpy as jnp
import numpy as np


def get_hinge_loss_grad_norm_fn(fn, params, state):

  @jit
  def score_fn(X, Y):
    per_sample_loss_fn = lambda p, x, y: vmap(hinge_loss)(fn(p, state, x), y)
    loss_grads = flatten_jacobian(jacrev(per_sample_loss_fn)(params, X, Y))
    scores = jnp.linalg.norm(loss_grads, axis=-1)
    return scores

  return lambda X, Y: np.array(score_fn(X, Y))


def get_covariance_fn(fn, params, state):

  @jit
  def score_fn(X, Y, Z):
    per_sample_fn = lambda p, x, y, z: vmap(lambda logits, attributes: logits * (attributes - Z.mean()))(fn(p, state, x), z)
    grads = flatten_jacobian(jacrev(per_sample_fn)(params, X, Y, Z))
    kernel = grads @ grads.T
    score = jnp.mean(kernel, axis=-1)
    return score

  return lambda X, Y, Z: np.array(score_fn(X, Y, Z))


def get_fairness_score_fn(fn, params, state, score_type):

  if score_type == 'covariance':
    print(f'compute {score_type}...')
    score_fn = get_covariance_fn(fn, params, state)
  elif score_type == 'grad_norm':
    print(f'compute {score_type}...')
    score_fn = lambda x, y, z: get_hinge_loss_grad_norm_fn(fn, params, state)(x, y)
  else:
    raise NotImplementedError
  return score_fn


def compute_fair_scores(fn, params, state, X, Y, Z, batch_size, score_type):
  n_batches = X.shape[0] // batch_size
  Xs, Ys, Zs = np.array_split(X, n_batches), np.array_split(Y, n_batches), np.array_split(Z, n_batches)
  score_fn = get_fairness_score_fn(fn, params, state, score_type)
  scores = []
  for i, (x, y, z) in enumerate(zip(Xs, Ys, Zs)):
    print(f'score batch {i+1} of {n_batches}, group balance: {z.mean()*100:.2f}%')
    scores.append(score_fn(x, y, z))
  scores = np.concatenate(scores)
  return scores