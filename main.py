
import jax
import xminigrid
import matplotlib.pyplot as plt
from xminigrid.wrappers import GymAutoResetWrapper
from xminigrid.experimental.img_obs import RGBImgObservationWrapper

key = jax.random.key(0)
reset_key, ruleset_key = jax.random.split(key)

# to list available benchmarks: xminigrid.registered_benchmarks()
benchmark = xminigrid.load_benchmark(name="trivial-1m")
# choosing ruleset, see section on rules and goals
ruleset = benchmark.sample_ruleset(ruleset_key)

# to list available environments: xminigrid.registered_environments()
env, env_params = xminigrid.make("XLand-MiniGrid-R9-25x25")
env_params = env_params.replace(ruleset=ruleset)

# auto-reset wrapper
env = GymAutoResetWrapper(env)

# render obs as rgb images if needed (warn: this will affect speed greatly)
env = RGBImgObservationWrapper(env)

# fully jit-compatible step and reset methods
timestep = jax.jit(env.reset)(env_params, reset_key)
timestep = jax.jit(env.step)(env_params, timestep, action=0)
# optionally render the state
plt.imshow(env.render(env_params, timestep))
plt.savefig("test.png")

timestep = jax.jit(env.step)(env_params, timestep, action=0)
plt.imshow(env.render(env_params, timestep))
plt.savefig("test2.png")


