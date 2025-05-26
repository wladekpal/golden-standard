import os

from matplotlib import pyplot as plt


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import jax
import jax.numpy as jnp
from ml_collections import config_flags
from absl import app, flags

import xminigrid
from xminigrid.wrappers import GymAutoResetWrapper
from xminigrid.experimental.img_obs import RGBImgObservationWrapper
from config import ROOT_DIR
from agents import agents



FLAGS = flags.FLAGS
flags.DEFINE_integer('seed', 0, 'Random seed.')

config_flags.DEFINE_config_file('agent', ROOT_DIR + '/agents/crl.py', lock_config=False)

def main(_):
    print(ROOT_DIR)
    config = FLAGS.agent
    config['discrete'] = True
    agent_class = agents[config['agent_name']]
    print(config)


    key = jax.random.key(0)
    reset_key = jax.random.split(key, 1)[0]


    # to list available environments: xminigrid.registered_environments()
    env, env_params = xminigrid.make("MiniGrid-EmptyRandom-5x5")

    # auto-reset wrapper
    env = GymAutoResetWrapper(env)

    # Create example batch with observations and actions
    timestep = jax.jit(env.reset)(env_params, reset_key)
    example_batch = {
        'observations':timestep.observation.reshape(1, -1),  # Add batch dimension
        'actions': jnp.zeros((1,), dtype=jnp.int32),  # Single action for batch size 1
        'value_goals': timestep.state.goal_encoding.reshape(1, -1),
        'actor_goals': timestep.state.goal_encoding.reshape(1, -1),
        # 'masks': jnp.ones((1,), dtype=jnp.float32),
        # 'rewards': jnp.zeros((1,), dtype=jnp.float32),
    }
    plt.imshow(env.render(env_params, timestep))
    plt.savefig("render.png")

    plt.imshow(timestep.observation[:,:,0])
    plt.savefig("observation_0.png")
    plt.imshow(timestep.observation[:,:,1])
    plt.savefig("observation_1.png")


    # Print environment details
    print(f"Number of actions: {env.num_actions(env_params)}")
    print(f"Observation shape: {env.observation_shape(env_params)}")
    print(f"Example batch observation shape: {example_batch['observations'].shape}")
    
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )
    print(f"Observation: {timestep.observation}")
    print(f"Goal: {timestep.state.goal_encoding}")

    agent.update(example_batch)

if __name__ == '__main__':
    app.run(main)