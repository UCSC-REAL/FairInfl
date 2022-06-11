from flax.training import checkpoints, lr_schedule
import jax
from jax import jit, value_and_grad, random
from jax import numpy as jnp
import numpy as np
import time
from .data import load_fairness_dataset
from .metrics import binary_correct, constraints, fairness, hinge_loss
from .models import get_apply_fn_test, get_apply_fn_train, get_model
from .recorder import init_recorder, record_ckpt, record_test_stats, record_train_stats, save_recorder
from .test import get_test_step, test
from .train_state import TrainState, get_train_state
from .utils import make_dir, print_args, save_args, set_global_seed


########################################################################################################################
#  Getters
########################################################################################################################


def create_vitaly_learning_rate_schedule():
  def learning_rate(step):
    base_lr, top, total = 0.2, 4680, 31200
    if step <= top:
      lr = base_lr * step / top
    else:
      lr = base_lr - base_lr * (step - top) / (total - top)
    return lr
  return learning_rate


def get_lr_schedule(args):
  if args.lr_vitaly:
    lr = create_vitaly_learning_rate_schedule()
  elif args.decay_steps:
    lr_sched_steps = [[e, args.decay_factor**(i + 1)] for i, e in enumerate(args.decay_steps)]
    lr_ = lr_schedule.create_stepped_learning_rate_schedule(args.lr, 1, lr_sched_steps)
    lr = lambda step: lr_(step).item()
  else:
    lr = lr_schedule.create_constant_learning_rate_schedule(args.lr, args.steps_per_epoch)
  return lr


def get_loss_fn(f_train):
  def loss_fn(params, model_state, x, y, z):
    logits, model_state = f_train(params, model_state, x)
    loss = hinge_loss(logits, y)
    con = constraints(logits, z)
    acc = jnp.mean(binary_correct(logits, y))
    return loss + 1. * con, (acc, logits, model_state)
  return loss_fn


def train_batches(I, X, Y, Z, args):
  """train_batches: I, X, Y, args -> (curr_step, I_batch, X_batch, Y_batch), ...
  In:
    I   : nparr(M)       : train (sub)set idxs
    X   : nparr(N, img)  : all train images
    Y   : nparr(N, C)    : all train labels
    args: SimpleNamespace: data generation args
  Gen:
    curr_step: int          : current train step
    I_batch  : nparr(B)     : batch train idxs
    X_batch  : nparr(B, img): batch train images
    Y_batch  : nparr(B, C)  : batch train labels
  """
  num_examples = I.shape[0]
  shuffle_key, augment_key = random.split(random.PRNGKey(args.train_seed))
  # initial shuffle
  shuffle_key, key = random.split(shuffle_key)
  I = np.array(random.permutation(key, I))
  # generate batches
  curr_step, start_idx = args.ckpt + 1, 0
  while curr_step <= args.num_steps:
    end_idx = start_idx + args.train_batch_size
    # shuffle at end of epoch
    if end_idx > num_examples:
      shuffle_key, key = random.split(shuffle_key)
      I = np.array(random.permutation(key, I))
      start_idx = 0
    # augment and yield train batch
    else:
      augment_key, key = random.split(augment_key)
      I_batch = I[start_idx:end_idx]
      X_batch, Y_batch, Z_batch =X[I_batch], Y[I_batch], Z[I_batch]
      # yield batch
      yield curr_step, I_batch, X_batch, Y_batch, Z[I_batch]
      # end step
      curr_step += 1
      start_idx = end_idx


def test_batches(X, Y, Z, batch_size):
  """test_batches: X, Y, batch_size -> (B, X_batch, Y_batch), ...
  In:
    X         : nparr(N, img): all test images
    Y         : nparr(N, C)  : all test labels
    batch_size: int          : maximum batch size
  Gen:
    B      : int          : current batch size
    X_batch: nparr(B, img): batch test images
    Y_batch: nparr(B, C)  : batch test labels
  """
  num_examples = X.shape[0]
  start_idx = 0
  while start_idx < num_examples:
    end_idx = min(start_idx + batch_size, num_examples)
    B = end_idx - start_idx
    X_batch, Y_batch, Z_batch = X[start_idx:end_idx], Y[start_idx:end_idx], Z[start_idx:end_idx]
    yield B, X_batch, Y_batch, Z_batch
    start_idx = end_idx


def get_train_step(loss_and_grad_fn):
  def train_step(state, x, y, z, lr):
    (loss, (acc, logits, model_state)), gradient = loss_and_grad_fn(state.optim.target, state.model, x, y, z)
    new_optim = state.optim.apply_gradient(gradient, learning_rate=lr)
    state = TrainState(optim=new_optim, model=model_state)
    return state, logits, loss, acc, gradient
  return train_step


def get_test_step(f_test):
  def test_step(state, x, y, z):
    logits = f_test(state.optim.target, state.model, x)
    loss = hinge_loss(logits, y)
    acc = jnp.mean(binary_correct(logits, y))
    pos, neg = fairness(logits, y, z)
    return loss, acc, pos, neg, logits
  return test_step


def test(test_step, state, X, Y, Z, batch_size):
  loss, acc, pos, neg, N = 0, 0, 0, 0, 0
  for n, x, y, z in test_batches(X, Y, Z, batch_size):
    step_loss, step_acc, step_pos, step_neg, logits = test_step(state, x, y, z)
    loss += step_loss * n
    acc += step_acc * n
    pos += step_pos
    neg += step_neg
    N += n
  # print(acc, N, pos, neg)
  loss, acc = loss / N, acc / N
  pos, neg = pos / jnp.sum(Z > 0), neg / jnp.sum(Z == 0)
  return loss, acc, pos - neg


########################################################################################################################
#  Bookkeeping
########################################################################################################################

def _log_and_save_args(args):
  print('train args:')
  print_args(args)
  save_args(args, args.save_dir, verbose=True)


def _make_dirs(args):
  make_dir(args.save_dir)
  make_dir(args.save_dir + '/ckpts')


def _print_stats(t, T, t_incr, t_tot, lr, train_acc, train_loss, test_acc, test_disp, test_loss, init=False):
  prog = t / T * 100
  lr = '  init' if init else f'{lr:.4f}'
  train_acc = ' init' if init else f'{train_acc:.3f}'
  train_loss = ' init' if init else f'{train_loss:.4f}'
  print(f'{prog:6.2f}% | time: {t_incr:5.1f}s ({t_tot/60:5.1f}m) | step: {t:6d} |',
          f'lr: {lr} | train acc: {train_acc} | train loss: {train_loss} | test acc: {test_acc:.3f} | test violation: {test_disp:.4f} | test loss: {test_loss:.4f}')


def _record_test(rec, t, T, t_prev, t_start, lr, train_acc, train_loss, test_acc, test_disp, test_loss, init=False):
  rec = record_test_stats(rec, t, test_loss, test_acc)
  t_now = time.time()
  t_incr, t_tot = t_now - t_prev, t_now - t_start
  _print_stats(t, T, t_incr, t_tot, lr, train_acc, train_loss, test_acc, test_disp, test_loss, init)
  return rec, t_now


def _save_checkpoint(save_dir, step, state, rec):
  checkpoints.save_checkpoint(save_dir + '/ckpts', state, step, keep=10000)
  rec = record_ckpt(rec, step)
  return rec


########################################################################################################################
#  Train
########################################################################################################################


def train(args):
  # setup
  set_global_seed()
  _make_dirs(args)
  I_train, X_train, Y_train, A_train, X_test, Y_test, A_test, args = load_fairness_dataset(args)
  model = get_model(args)
  state, args = get_train_state(args, model)
  f_train, f_test = get_apply_fn_train(model), get_apply_fn_test(model)
  test_step = jit(get_test_step(f_test))
  train_step = jit(get_train_step(value_and_grad(get_loss_fn(f_train), has_aux=True)))
  lr = get_lr_schedule(args)
  rec = init_recorder()

  # info
  _log_and_save_args(args)
  time_start = time.time()
  time_now = time_start
  print('train net...')

  # log and save init
  test_loss, test_acc, test_disp = test(test_step, state, X_test, Y_test, A_test, args.test_batch_size)
  rec, time_now = _record_test(
      rec, args.ckpt, args.num_steps, time_now, time_start, None, None, None, test_acc, test_disp, test_loss, True)
  rec = _save_checkpoint(args.save_dir, args.ckpt, state, rec)

  # train loop
  for t, idxs, x, y, z in train_batches(I_train, X_train, Y_train, A_train, args):
    # train step
    state, logits, loss, acc, grad = train_step(state, x, y, z, lr(t))
    rec = record_train_stats(rec, t-1, loss.item(), acc.item(), lr(t))

  #  BOOKKEEPING  #

    # test and log every log_steps
    if t % args.log_steps == 0:
      test_loss, test_acc, test_disp = test(test_step, state, X_test, Y_test, A_test, args.test_batch_size)
      rec, time_now = _record_test(rec, t, args.num_steps, time_now, time_start, lr(t), acc, loss, test_acc, test_disp, test_loss)

    # every early_save_steps before early_step and save_steps after early_step, and at end of training
    if ((t <= args.early_step and t % args.early_save_steps == 0) or
       (t > args.early_step and t % args.save_steps == 0) or
       (t == args.num_steps)):

      # test and log if not done already
      if t % args.log_steps != 0:
        test_loss, test_acc, test_disp = test(test_step, state, X_test, Y_test, A_test, args.test_batch_size)
        rec, time_now = _record_test(rec, t, args.num_steps, time_now, time_start, lr(t), acc, loss, test_acc, test_disp, test_loss)

      # save checkpoint
      rec = _save_checkpoint(args.save_dir, t, state, rec)

  # wrap it up
  save_recorder(args.save_dir, rec)
  return test_acc, test_disp
