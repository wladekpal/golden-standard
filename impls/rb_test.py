import os
import functools

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import jax
import flax.struct
import jax.numpy as jnp
from jax import flatten_util
import xminigrid
from xminigrid.wrappers import GymAutoResetWrapper
from xminigrid.types import TimeStep
from xminigrid.experimental.img_obs import RGBImgObservationWrapper
from benchmark_test import build_benchmark

@flax.struct.dataclass
class ReplayBufferState:
    """Contains data related to a replay buffer."""

    data: jnp.ndarray
    insert_position: jnp.ndarray
    sample_position: jnp.ndarray
    key: jnp.ndarray

class TrajectoryUniformSamplingQueue:
    """
    Base class for limited-size FIFO reply buffers.

    Implements an `insert()` method which behaves like a limited-size queue.
    I.e. it adds samples to the end of the queue and, if necessary, removes the
    oldest samples form the queue in order to keep the maximum size within the
    specified limit.

    Derived classes must implement the `sample()` method.
    """

    def __init__(
        self,
        max_replay_size: int,
        dummy_data_sample,
        sample_batch_size: int,
        num_envs: int,
        episode_length: int,
    ):
        self._flatten_fn = jax.vmap(jax.vmap(lambda x: flatten_util.ravel_pytree(x)[0]))
        dummy_flatten, self._unflatten_fn = flatten_util.ravel_pytree(dummy_data_sample)
        self._unflatten_fn = jax.vmap(jax.vmap(self._unflatten_fn))
        data_size = len(dummy_flatten)

        self._data_shape = (max_replay_size, num_envs, data_size)
        self._data_dtype = dummy_flatten.dtype
        self._sample_batch_size = sample_batch_size
        self._size = 0
        self.num_envs = num_envs
        self.episode_length = episode_length

    def init(self, key):
        return ReplayBufferState(
            data=jnp.zeros(self._data_shape, self._data_dtype),
            sample_position=jnp.zeros((), jnp.int32),
            insert_position=jnp.zeros((), jnp.int32),
            key=key,
        )

    def insert(self, buffer_state, samples):
        """Insert data into the replay buffer."""
        self.check_can_insert(buffer_state, samples, 1)
        return self.insert_internal(buffer_state, samples)

    def check_can_insert(self, buffer_state, samples, shards):
        """Checks whether insert operation can be performed."""
        assert isinstance(shards, int), "This method should not be JITed."
        insert_size = jax.tree_util.tree_flatten(samples)[0][0].shape[0] // shards
        if self._data_shape[0] < insert_size:
            raise ValueError(
                "Trying to insert a batch of samples larger than the maximum replay"
                f" size. num_samples: {insert_size}, max replay size"
                f" {self._data_shape[0]}"
            )
        self._size = min(self._data_shape[0], self._size + insert_size)

    def check_can_sample(self, buffer_state, shards):
        """Checks whether sampling can be performed. Do not JIT this method."""

    def insert_internal(self, buffer_state, samples):
        """Insert data in the replay buffer.

        Args:
          buffer_state: Buffer state
          samples: Sample to insert with a leading batch size.

        Returns:
          New buffer state.
        """
        if buffer_state.data.shape != self._data_shape:
            raise ValueError(
                f"buffer_state.data.shape ({buffer_state.data.shape}) "
                f"doesn't match the expected value ({self._data_shape})"
            )

        update = self._flatten_fn(samples)  # Updates has shape (unroll_len, num_envs, self._data_shape[-1])
        data = buffer_state.data  # shape = (max_replay_size, num_envs, data_size)

        # If needed, roll the buffer to make sure there's enough space to fit
        # `update` after the current position.
        position = buffer_state.insert_position
        roll = jnp.minimum(0, len(data) - position - len(update))
        data = jax.lax.cond(roll, lambda: jnp.roll(data, roll, axis=0), lambda: data)
        position = position + roll

        # Update the buffer and the control numbers.
        data = jax.lax.dynamic_update_slice_in_dim(data, update, position, axis=0)
        position = (
            (position + len(update)) % (len(data) + 1)
        )  # so whenever roll happens, position becomes len(data), else it is increased by len(update), what is the use of doing % (len(data) + 1)??

        return buffer_state.replace(
            data=data,
            insert_position=position,
        )

    def sample(self, buffer_state):
        """Sample a batch of data."""
        self.check_can_sample(buffer_state, 1)
        return self.sample_internal(buffer_state)

    def sample_internal(self, buffer_state):
        if buffer_state.data.shape != self._data_shape:
            raise ValueError(
                f"Data shape expected by the replay buffer ({self._data_shape}) does "
                f"not match the shape of the buffer state ({buffer_state.data.shape})"
            )
        key, sample_key, shuffle_key = jax.random.split(buffer_state.key, 3)
        # Note: this is the number of envs to sample but it can be modified if there is OOM
        shape = self.num_envs

        # Sampling envs idxs
        envs_idxs = jax.random.choice(sample_key, jnp.arange(self.num_envs), shape=(shape,), replace=False)

        @functools.partial(jax.jit, static_argnames=("rows", "cols"))
        def create_matrix(rows, cols, min_val, max_val, rng_key):
            rng_key, subkey = jax.random.split(rng_key)
            start_values = jax.random.randint(subkey, shape=(rows,), minval=min_val, maxval=max_val)
            row_indices = jnp.arange(cols)
            matrix = start_values[:, jnp.newaxis] + row_indices
            return matrix

        @jax.jit
        def create_batch(arr_2d, indices):
            return jnp.take(arr_2d, indices, axis=0, mode="wrap")

        create_batch_vmaped = jax.vmap(create_batch, in_axes=(1, 0))

        matrix = create_matrix(
            shape,
            self.episode_length,
            buffer_state.sample_position,
            buffer_state.insert_position - self.episode_length,
            sample_key,
        )

        """
        The function create_batch will be called for every envs_idxs of buffer_state.data and every row of matrix.
        Because every row of matrix has consecutive indices of self.episode_length, for every
        envs_idx of envs_idxs, we will sample a random self.episode_length length sequence from 
        buffer_state.data[:, envs_idx, :]. But I don't think the code ensures that this sequence 
        won't be across episodes?

        flatten_batch takes care of this
        """
        batch = create_batch_vmaped(buffer_state.data[:, envs_idxs, :], matrix)
        transitions = self._unflatten_fn(batch)
        return buffer_state.replace(key=key), transitions

    def size(self, buffer_state: ReplayBufferState) -> int:
        return buffer_state.insert_position - buffer_state.sample_position




def jit_wrap(buffer):
    buffer.insert_internal = jax.jit(buffer.insert_internal)
    buffer.sample_internal = jax.jit(buffer.sample_internal)
    return buffer



key = jax.random.PRNGKey(0)
buffer_key, reset_key = jax.random.split(key, 2)
env, env_params = xminigrid.make('MiniGrid-EmptyRandom-5x5')
env = GymAutoResetWrapper(env)
timestep = jax.jit(env.reset)(env_params, reset_key)

print(timestep.observation.shape)
print(timestep.state.step_num)

replay_buffer = jit_wrap(
    TrajectoryUniformSamplingQueue(
        max_replay_size=100,
        dummy_data_sample=timestep,
        sample_batch_size=256,
        num_envs=256,
        episode_length=10,
    )
)
buffer_state = jax.jit(replay_buffer.init)(buffer_key)
# replay_buffer = TrajectoryUniformSamplingQueue(
#         max_replay_size=100,
#         dummy_data_sample=timestep,
#         sample_batch_size=256,
#         num_envs=256,
#         episode_length=10,
#     )
# buffer_state = replay_buffer.init(buffer_key)

print(replay_buffer._data_shape)



benchmark_fn = build_benchmark('MiniGrid-EmptyRandom-5x5', 256, 50)

# TypeError: Cannot determine dtype of key<fry> while using key = jax.random.key(0)
key = jax.random.PRNGKey(0)
env_step, timesteps_all = benchmark_fn(key)

buffer_state = replay_buffer.insert(buffer_state, timesteps_all)

buffer_state, transitions = replay_buffer.sample(buffer_state) 


# @jax.jit
def segment_ids_per_row(x: jnp.ndarray) -> jnp.ndarray:
    """
    Parameters
    ----------
    x : jnp.ndarray            # shape (rows, cols) or (cols,)
        Row-wise sequences whose wrap-arounds you want to label.

    Returns
    -------
    jnp.ndarray                # same shape as `x`
        0-based index of the sub-vector each element belongs to.
    """
    # 1. Where does the sequence go “backwards”?
    #    diff < 0  →  a wrap-around happened between col k-1 and k
    resets = jnp.concatenate(
        [jnp.zeros((*x.shape[:-1], 1), dtype=jnp.int32),     # first column never resets
         (jnp.diff(x, axis=-1) < 0).astype(jnp.int32)],
        axis=-1
    )

    # 2. Turn those reset flags into running segment numbers.
    return jnp.cumsum(resets, axis=-1)

row = jnp.array([13, 14, 15,  0, 1, 2, 3])
print(segment_ids_per_row(row))          # → [0 0 0 1 1 1 1]

mat = jnp.array([[37,38,39,40,41,42,43,44,45,46],
                 [17,18,19,20,21,22,23,24,25,26],
                 [16,17,18,19,20,21,22,23,24, 0],
                 [14,15,16,17,18,19,20,21, 0, 1],
                 [23,24,25,26,27,28,29,30,31,32]])
print(segment_ids_per_row(mat))




@functools.partial(jax.jit, static_argnames=("buffer_config"))
def flatten_batch(buffer_config, transition, sample_key):
    gamma, state_size, goal_indices = buffer_config

    # Because it's vmaped transition.obs.shape is of shape (episode_len, obs_dim)
    seq_len = transition.observation.shape[0]
    arrangement = jnp.arange(seq_len)
    is_future_mask = jnp.array(
        arrangement[:, None] < arrangement[None], dtype=jnp.float32
    )  # upper triangular matrix of shape seq_len, seq_len where all non-zero entries are 1
    discount = gamma ** jnp.array(arrangement[None] - arrangement[:, None], dtype=jnp.float32)
    probs = is_future_mask * discount

    # probs is an upper triangular matrix of shape seq_len, seq_len of the form:
    #    [[0.        , 0.99      , 0.98010004, 0.970299  , 0.960596 ],
    #    [0.        , 0.        , 0.99      , 0.98010004, 0.970299  ],
    #    [0.        , 0.        , 0.        , 0.99      , 0.98010004],
    #    [0.        , 0.        , 0.        , 0.        , 0.99      ],
    #    [0.        , 0.        , 0.        , 0.        , 0.        ]]
    # assuming seq_len = 5
    # the same result can be obtained using probs = is_future_mask * (gamma ** jnp.cumsum(is_future_mask, axis=-1))

    single_trajectories = segment_ids_per_row(transition.state.step_num)
    jax.debug.print("single_trajectories: {x} ", x=single_trajectories)
    # array of seq_len x seq_len where a row is an array of traj_ids that correspond to the episode index from which that time-step was collected
    # timesteps collected from the same episode will have the same traj_id. All rows of the single_trajectories are same.

    probs = probs * jnp.equal(single_trajectories, single_trajectories.T) + jnp.eye(seq_len) * 1e-5
    jax.debug.print("probs: {x} ", x=probs)
    # ith row of probs will be non zero only for time indices that
    # 1) are greater than i
    # 2) have the same traj_id as the ith time index

    # goal_index = jax.random.categorical(sample_key, jnp.log(probs))
    # future_state = jnp.take(
    #     transition.observation, goal_index[:-1], axis=0
    # )  # the last goal_index cannot be considered as there is no future.
    # future_action = jnp.take(transition.action, goal_index[:-1], axis=0)
    # goal = future_state[:, goal_indices]
    # future_state = future_state[:, :state_size]
    # state = transition.observation[:-1, :state_size]  # all states are considered
    # new_obs = jnp.concatenate([state, goal], axis=1)

    # extras = {
    #     "policy_extras": {},
    #     "state_extras": {
    #         "truncation": jnp.squeeze(transition.extras["state_extras"]["truncation"][:-1]),
    #         "traj_id": jnp.squeeze(transition.extras["state_extras"]["traj_id"][:-1]),
    #     },
    #     "state": state,
    #     "future_state": future_state,
    #     "future_action": future_action,
    # }

    # return transition._replace(
    #     observation=jnp.squeeze(new_obs),  # this has shape (num_envs, episode_length-1, obs_size)
    #     action=jnp.squeeze(transition.action[:-1]),
    #     reward=jnp.squeeze(transition.reward[:-1]),
    #     discount=jnp.squeeze(transition.discount[:-1]),
    #     extras=extras,
    # )
    return transition



buffer_state, transitions = replay_buffer.sample(buffer_state)

# process transitions for training
batch_keys = jax.random.split(buffer_state.key, transitions.observation.shape[0])
print(f"batch_keys: {batch_keys.shape}")
flattened_transitions = jax.vmap(flatten_batch, in_axes=(None, 0, 0))((0.99, 171, 0), transitions, batch_keys)
print(f"flattened_transitions.observation.shape: {flattened_transitions.observation.shape}")



