from jax import random
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import os, time
from sklearn.model_selection import train_test_split

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


########################################################################################################################
#  Load Data
########################################################################################################################

def one_hot(labels, num_classes, dtype=np.float32):
  return (labels[:, None] == np.arange(num_classes)).astype(dtype)


def sort_by_class(X, Y):
  if len(Y.shape) == 1:
    sort_idxs = Y.argsort()
  else:
    sort_idxs = Y.argmax(1).argsort()
  X, Y = X[sort_idxs], Y[sort_idxs]
  return X, Y


def update_data_args(args, X_train, Y_train, X_test, Y_test):
  args.image_shape = X_train.shape[1:]
  if len(Y_train.shape) == 1:
    args.num_classes = 1
  else:
    args.num_classes = Y_train.shape[1]
  args.num_train_examples = X_train.shape[0]
  args.num_test_examples = X_test.shape[0]
  args.steps_per_epoch = args.num_train_examples // args.train_batch_size
  args.steps_per_test = int(np.ceil(args.num_test_examples / args.test_batch_size))
  return args


def load_adult(args):
    path = args.data_dir + '/adult.npy'
    with open(path, 'rb') as f:
        X, Y = np.load(f), np.load(f)
        A = X[:, 0]
    X_train, X_test, Y_train, Y_test, A_train, A_test = train_test_split(X, Y, A, test_size=0.2, stratify=Y)
    args = update_data_args(args, X_train, Y_train, X_test, Y_test)
    return X_train, Y_train, A_train, X_test, Y_test, A_test, args


def load_jigsaw(args):
    path = args.data_dir
    with open(path + '/Jigsaw_train.npy', 'rb') as f:
        X_train, Y_train, A_train = np.load(f), np.load(f), np.load(f)
    with open(path + '/Jigsaw_test.npy', 'rb') as f:
        X_test, Y_test, A_test = np.load(f), np.load(f), np.load(f)
    from sklearn import utils
    X_train, Y_train, A_train = utils.shuffle(X_train, Y_train, A_train)
    args = update_data_args(args, X_train, Y_train, X_test, Y_test)
    return X_train, Y_train, A_train, X_test, Y_test, A_test, args


def load_celeba(args):
  ATTR_KEY = "attributes"
  IMAGE_KEY = "image"
  LABEL_KEY = "Smiling"
  GROUP_KEY = "Male"
  IMAGE_SIZE = 28
  NUM_CLASSES = 1

  def preprocessing_function(feat_dict):
    image = tf.image.resize_with_crop_or_pad(feat_dict[IMAGE_KEY], IMAGE_SIZE, IMAGE_SIZE).numpy().astype(np.float32)/255.

    label = feat_dict[ATTR_KEY][LABEL_KEY].numpy().astype(np.float32)
    group = feat_dict[ATTR_KEY][GROUP_KEY].numpy().astype(np.float32)
    
    return image, label, group

  ds_train, ds_test = tfds.load(name='celeb_a', split=['train', 'test'], data_dir=args.data_dir,
      batch_size=-1, download=True)
  X_train, Y_train, A_train = preprocessing_function(ds_train)
  X_test, Y_test, A_test = preprocessing_function(ds_test)

  args = update_data_args(args, X_train, Y_train, X_test, Y_test)

  return X_train, Y_train, A_train, X_test, Y_test, A_test, args


def load_fairness_dataset(args):
  if args.dataset.lower() == 'celeba':
    X_train, Y_train, A_train, X_test, Y_test, A_test, args = load_celeba(args)
  elif args.dataset.lower() == 'adult':
    X_train, Y_train, A_train, X_test, Y_test, A_test, args = load_adult(args)
  elif args.dataset.lower() == 'jigsaw':
    X_train, Y_train, A_train, X_test, Y_test, A_test, args = load_jigsaw(args)
  else:
    raise NotImplementedError

  I_train = np.arange(X_train.shape[0], dtype=np.int32)
  if args.subset:
    I_train, args = subset_train_idxs(I_train, args)
  return I_train, X_train, Y_train, A_train, X_test, Y_test, A_test, args


def update_train_data_args(args, I):
  args.num_train_examples = I.shape[0]
  args.steps_per_epoch = args.num_train_examples // args.train_batch_size
  return args


def subset_train_idxs_randomly(I, args):
  rng = np.random.RandomState(args.random_subset_seed)
  I = np.sort(rng.choice(I.shape[0], args.subset_size, replace=False)).astype(np.int32)
  args = update_train_data_args(args, I)
  return I, args


def subset_train_idxs_by_scores(I, args):
  scores = np.load(args.scores_path)
  if args.subset == 'keep_min_scores':
    idxs = scores.argsort()[:args.subset_size]
  elif args.subset == 'keep_max_scores':
    idxs = scores.argsort()[-args.subset_size:]
  elif args.subset == 'keep_min_abs_scores':
    idxs = np.abs(scores).argsort()[:args.subset_size]
  elif args.subset == 'keep_max_abs_scores':
    idxs = np.abs(scores).argsort()[-args.subset_size:]
  else:
    raise NotImplementedError
  I = np.sort(idxs).astype(np.int32)
  args = update_train_data_args(args, I)
  return I, args


def subset_train_idxs_by_offset(I, args):
  scores = np.load(args.scores_path)
  idxs = scores.argsort()[args.subset_offset : args.subset_offset + args.subset_size]
  I = np.sort(idxs).astype(np.int32)
  args = update_train_data_args(args, I)
  return I, args


def subset_train_idxs(I, args):
  # full data
  if args.subset_size == -1:
    return I, args
  if args.subset == 'random':
    I, args = subset_train_idxs_randomly(I, args)
  elif args.subset == 'offset':
    I, args = subset_train_idxs_by_offset(I, args)
  else:
    I, args = subset_train_idxs_by_scores(I, args)
  return I, args


def randomize_labels(X, Y, fraction, seed):
  rng = np.random.RandomState(seed)
  num_labels = Y.shape[0]
  num_randomize = int(num_labels * fraction)
  rand_idxs = rng.choice(num_labels, num_randomize, replace=False)
  Y_rand = Y.copy()
  Y_rand[np.sort(rand_idxs)] = Y_rand[rand_idxs]
  X, Y_rand = sort_by_class(X, Y_rand)
  return X, Y_rand



########################################################################################################################
#  Data Iterators
########################################################################################################################

def augment_celeba_data(X, Y, key):
  B, H, W, C = X.shape
  crop_seed, flip_seed = key[0].item(), key[1].item()
  paddings = tf.constant([[0, 0], [4, 4], [4, 4], [0, 0]])
  X = tf.pad(X, paddings, 'REFLECT')
  X = tf.image.random_crop(X, [B, H, W, C], crop_seed)
  X = tf.image.random_flip_left_right(X, flip_seed)
  X = X.numpy()
  return X, Y


def augment_data(X, Y, key, args):
  if args.augment:
    X, Y = augment_celeba_data(X, Y, key)
  return X, Y


def train_batches(I, X, Y, args):
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
      X_batch, Y_batch = augment_data(X[I_batch], Y[I_batch], key, args)
      # yield batch
      yield curr_step, I_batch, X_batch, Y_batch
      # end step
      curr_step += 1
      start_idx = end_idx


def test_batches(X, Y, batch_size):
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
    X_batch, Y_batch = X[start_idx:end_idx], Y[start_idx:end_idx]
    yield B, X_batch, Y_batch
    start_idx = end_idx


# ######################################################################################################################
# Miscellaneous
# ######################################################################################################################

def get_class_balanced_random_subset(X, Y, cls_smpl_sz, seed):
  """get_class_balanced_random_subset: X, Y, cls_smpl_sz, seed -> X, Y
  In:
    X          : nparr(N, img): all images, ASSUME sorted by class
    Y          : nparr(N, C)  : corresponding labels, ASSUME equal number of examples per class
    cls_smpl_sz: int          : number of examples per class in subset
    seed       : int          : random seed
  Out:
    X: nparr(C * cls_smpl_sz, img): subsampled images, cls_smpl_sz examples per class, sorted by class
    Y: nparr(C * cls_smpl_sz, C)  : corresponding labels
  """
  # reshape to class x batch x image/label
  n_cls = Y.shape[1]
  X_c, Y_c = np.stack(np.split(X, n_cls)), np.stack(np.split(Y, n_cls))
  # sample from batch dimension
  rng = np.random.RandomState(seed)
  idxs = [rng.choice(X_c.shape[1], cls_smpl_sz, replace=False) for _ in range(n_cls)]
  X = np.concatenate([X_c[c, idxs[c]] for c in range(n_cls)])
  Y = np.concatenate([Y_c[c, idxs[c]] for c in range(n_cls)])
  return X, Y
