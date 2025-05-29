import os
import numpy as np
from matplotlib import pyplot as plt

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import jax
import jax.numpy as jnp
from ml_collections import config_flags
from absl import app, flags

import xminigrid
from xminigrid.wrappers import GymAutoResetWrapper
from xminigrid.experimental.img_obs import RGBImgObservationWrapper
from config import ROOT_DIR
from agents import agents
from utils.datasets import GCDataset, ReplayBuffer


FLAGS = flags.FLAGS
flags.DEFINE_string('env', 'MiniGrid-EmptyRandom-5x5', 'Environment to use.')
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
    env, env_params = xminigrid.make(FLAGS.env)
    env_params = env_params.replace(view_size=3)
    

    # auto-reset wrapper
    env = GymAutoResetWrapper(env)

    # Create example batch with observations and actions
    timestep = jax.jit(env.reset)(env_params, reset_key)
    example_batch = {
        'observations':timestep.observation.reshape(1, -1),  # Add batch dimension
        # 'actions': jnp.zeros((1,), dtype=jnp.int32).reshape(1, -1), 
        'actions': jnp.zeros((1,), dtype=jnp.int32),  # Single action for batch size 1
        'value_goals': timestep.state.goal_encoding.reshape(1, -1),
        'actor_goals': timestep.state.goal_encoding.reshape(1, -1),
        'terminals': timestep.step_type.reshape(1,-1) == xminigrid.types.StepType.LAST,
        # 'masks': jnp.ones((1,), dtype=jnp.float32),
        # 'rewards': jnp.zeros((1,), dtype=jnp.float32),
    }
    print(f"terminals shape: {example_batch['terminals'].shape}")
    print(f"observations shape: {example_batch['observations'].shape}")
    print(f"actions shape: {example_batch['actions'].shape}")
    print(f"value_goals shape: {example_batch['value_goals'].shape}")
    print(f"actor_goals shape: {example_batch['actor_goals'].shape}")



    plt.imshow(env.render(env_params, timestep))
    plt.savefig("render.png")

    plt.imshow(timestep.observation[:,:,0])
    plt.savefig("observation_0.png")
    plt.imshow(timestep.observation[:,:,1])
    plt.savefig("observation_1.png")

    modified_observation = np.array(timestep.observation) 
    modified_observation[:,1,0] = 10 # This is the axis on which there's agent sight
    modified_observation[:,1,1] = 10
    plt.imshow(modified_observation[:,:,0])
    plt.savefig("modified_observation_0.png")
    plt.imshow(modified_observation[:,:,1])
    plt.savefig("modified_observation_1.png")


    # Print environment details
    print(f"Number of actions: {env.num_actions(env_params)}")
    print(f"Observation shape: {env.observation_shape(env_params)}")
    print(f"Example batch observation shape: {example_batch['observations'].shape}")
    
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
        example_batch['value_goals'],
    )

    print(f"Observation: {timestep.observation}")
    print(f"Modified observation: {modified_observation}")
    print(f"Goal: {timestep.state.goal_encoding}")

    agent.update(example_batch)


    print(timestep.step_type)
    transition = {
        'observations': timestep.observation,
        'actions': jnp.zeros((1,), dtype=jnp.int32),  
        'value_goals': timestep.state.goal_encoding,
        'actor_goals': timestep.state.goal_encoding,
        'terminals': timestep.step_type == xminigrid.types.StepType.LAST,
    }
    rb = ReplayBuffer.create(transition, 1000)

    for i in range(10):
        print(f"step {i}")
        timestep = jax.jit(env.step)(env_params, timestep, action=0)
        transition = {
            'observations': timestep.observation,
            'actions': jnp.zeros((1,), dtype=jnp.int32),  
            'value_goals': timestep.state.goal_encoding,
            'actor_goals': timestep.state.goal_encoding,
            'terminals': timestep.step_type == xminigrid.types.StepType.LAST,
        }
        rb.add_transition(transition)
    print(f"rb size: {rb.size}")
    
    # gc_dataset = GCDataset(rb, config)
    # print(f"gc_dataset size: {gc_dataset.size}")
    # print(f"gc_dataset sample: {gc_dataset.sample(1)}")

if __name__ == '__main__':
    app.run(main)