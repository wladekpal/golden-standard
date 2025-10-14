import itertools
import sys

import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

# Printing.
np.set_printoptions(precision=3, suppress=True, linewidth=100)


# TODO: Had to remove lightsource due to some rendering artifacts
static_model = """
<mujoco>
  <statistic center="1 0 0.55" extent="1.1"/>

  <visual>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global offwidth="1000" offheight="1000" azimuth="150" elevation="-20"/>

  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>


    <texture name="wood1" file="assets/light-wood.png" type="2d"/>
    <material name="wood1" texture="wood1" shininess="0.5"/>

    <texture name="wood2" file="assets/oak.png" type="2d"/>
    <material name="wood2" texture="wood2" shininess="0.5"/>

    <texture name="wood3" file="assets/wood3.png" type="2d"/>
    <material name="wood3" texture="wood3" shininess="0.5"/>

    <texture name="steel" file="assets/steel.png" type="2d"/>
    <material name="steel" texture="steel" shininess="0.5"/>

  </asset>

  <worldbody>
  </worldbody>
</mujoco>
"""

if len(sys.argv) < 2:
    print("Error: Please provide a path to the data file with trajectory as an argument.")
    sys.exit(1)

data_path = sys.argv[1]
data = np.load(data_path)

FIELD_WIDTH = 0.5
BLOCK_WIDTH = 0.3
SUBDIV_STEPS = 10
ENV_IDX = 0
SPHERE_SIZE = 0.2
RESOLUTION = (1000, 1000)
EP_LEN = 30


def find_agent(state):
    for i, j in itertools.product(range(state.shape[0]), range(state.shape[1])):
        if state[i, j] in [3, 4, 5, 6, 7, 8, 9, 11]:  # agent
            return np.array([i, j])
    return None


def add_field(spec, x, y):
    checker_color = (x + y) % 2 == 0
    spec.worldbody.add_geom(
        name="floor-%d-%d" % (x, y),
        type=mj.mjtGeom.mjGEOM_BOX,
        rgba=([0.2, 0.3, 0.4, 1] if checker_color else [0.1, 0.2, 0.3, 1]),
        pos=[x, y, 0],
        size=[FIELD_WIDTH, FIELD_WIDTH, 0.01],
    )


def add_box(spec, x, y):
    spec.worldbody.add_geom(
        name="box-%d-%d" % (x, y),
        type=mj.mjtGeom.mjGEOM_BOX,
        pos=[x, y, BLOCK_WIDTH / 2],
        size=[BLOCK_WIDTH / 2, BLOCK_WIDTH / 2, BLOCK_WIDTH / 2],
        material="wood1",
    )


def add_carried_box(spec, x, y):
    spec.worldbody.add_geom(
        name="carried-box-%d-%d" % (x, y),
        type=mj.mjtGeom.mjGEOM_BOX,
        pos=[x, y, SPHERE_SIZE * 2.0],
        size=[BLOCK_WIDTH / 6, BLOCK_WIDTH / 6, BLOCK_WIDTH / 6],
        material="wood1",
    )


def add_actor(spec, x, y):
    spec.worldbody.add_geom(
        name="sphere",
        type=mj.mjtGeom.mjGEOM_SPHERE,
        rgba=[0, 0.7, 1, 0.7],
        pos=[x, y, SPHERE_SIZE * 1.1],
        size=[SPHERE_SIZE, SPHERE_SIZE, SPHERE_SIZE],
        material="steel",
    )


def add_target(spec, x, y):
    spec.worldbody.add_geom(
        name="target-%d-%d" % (x, y),
        type=mj.mjtGeom.mjGEOM_BOX,
        rgba=[0, 1, 0.1, 0.4],
        pos=[x, y, BLOCK_WIDTH / 1.8],
        size=[BLOCK_WIDTH / 1.8, BLOCK_WIDTH / 1.8, BLOCK_WIDTH / 1.8],
        material="wood1",
        contype=0,
        conaffinity=0,
    )


def render_trajectory(data, static_model):
    # Create an empty list to store the rendered images
    frames = []
    # Iterate through the timesteps data
    next_data = data.copy()
    next_data[:-1] = data[1:]
    next_data[-1] = data[-1]
    for state, next_state, step_num in zip(data, next_data, range(len(data))):
        print(f"Rendering frame {step_num}/{len(data)}")
        next_actor_pos = find_agent(next_state)
        curr_actor_pos = find_agent(state)

        num_subdivisions = 1 if (next_actor_pos == curr_actor_pos).all() else SUBDIV_STEPS

        for subdiv_step in range(num_subdivisions):
            interpolated_fraction = subdiv_step / (SUBDIV_STEPS - 1)
            interp_pos = (next_actor_pos - curr_actor_pos) * interpolated_fraction + curr_actor_pos

            spec = mj.MjSpec.from_string(static_model)
            grid_size = state.shape[0]

            # Add geoms based on the current state
            for i, j in itertools.product(range(grid_size), range(grid_size)):
                add_field(spec, i, j)

                if state[i, j] == 1:
                    add_box(spec, i, j)
                if state[i, j] == 2:  # target
                    add_target(spec, i, j)
                if state[i, j] == 3:  # agent
                    add_actor(spec, interp_pos[0], interp_pos[1])
                if state[i, j] == 4:  # agent_carrying_box:
                    add_actor(spec, interp_pos[0], interp_pos[1])
                    add_carried_box(spec, interp_pos[0], interp_pos[1])
                if state[i, j] == 5:  # agent_on_box
                    add_box(spec, i, j)
                    add_actor(spec, interp_pos[0], interp_pos[1])
                if state[i, j] == 6:  # agent_on_target
                    add_actor(spec, interp_pos[0], interp_pos[1])
                    add_target(spec, i, j)
                if state[i, j] == 7:  # agent_on_target_carrying_box
                    add_actor(spec, interp_pos[0], interp_pos[1])
                    add_carried_box(spec, interp_pos[0], interp_pos[1])
                    add_target(spec, i, j)
                if state[i, j] == 8:  # agent_on_target_with_box
                    add_box(spec, i, j)
                    add_actor(spec, interp_pos[0], interp_pos[1])
                    add_target(spec, i, j)
                if state[i, j] == 9:  # agent_on_target_with_box_carrying_box
                    add_box(spec, i, j)
                    add_actor(spec, interp_pos[0], interp_pos[1])
                    add_carried_box(spec, interp_pos[0], interp_pos[1])
                    add_target(spec, i, j)
                if state[i, j] == 10:  # box_on_target
                    add_box(spec, i, j)
                    add_target(spec, i, j)
                if state[i, j] == 11:  # agent_on_box_carrying_box
                    add_box(spec, i, j)
                    add_actor(spec, interp_pos[0], interp_pos[1])
                    add_carried_box(spec, interp_pos[0], interp_pos[1])

            # Recompile the model with the updated worldbody
            model = spec.compile()
            data_mj = mj.MjData(model)  # Create new data for the updated model

            # Render the current state and add it to the frames list
            with mj.Renderer(model, *RESOLUTION) as renderer:
                mj.mj_forward(model, data_mj)
                # Use the same camera settings as before
                camera = mj.MjvCamera()
                camera.type = mj.mjtCamera.mjCAMERA_FREE
                camera.distance = 8
                camera.azimuth = 65
                camera.elevation = -25
                camera.lookat[:] = [2, 2, 0.55]
                renderer.update_scene(data_mj, camera)
                frames.append(renderer.render())

    return frames


frames = render_trajectory(data[:EP_LEN, ENV_IDX], static_model)


fig, ax = plt.subplots(figsize=(frames[0].shape[1] / 50, frames[0].shape[0] / 50), dpi=100)
ax.axis("off")
im = ax.imshow(frames[0], animated=True)

output_path = "output.gif"


def update(i):
    im.set_array(frames[i])
    return (im,)


print("Creating .gif file")
fps = 12  # default frames per second; change if you want
anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=True)

# Save using PillowWriter
writer = PillowWriter(fps=fps)
anim.save(output_path, writer=writer)
plt.close(fig)  # close figure to avoid duplicate display

print(f"Saved GIF to {output_path} (frames={len(frames)}, fps={fps}).\nDisplaying below:")
