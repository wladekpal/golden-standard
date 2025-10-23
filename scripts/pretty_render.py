import itertools
import os
import sys

os.environ["MUJOCO_GL"] = "egl"

import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


# Printing.
np.set_printoptions(precision=3, suppress=True, linewidth=100)


# TODO: Had to remove lightsource due to some rendering artifacts
static_model = """
<mujoco>
    <statistic center="1.5 1.5 0.4" extent="2.2"/>

    <visual>
        <quality shadowsize="4096" offsamples="8"/>
        <map znear="0.01" zfar="80" fogstart="8" fogend="14"/>
    <headlight diffuse="0.42 0.44 0.48" ambient="0.16 0.16 0.18" specular="0.12 0.12 0.12"/>
        <rgba haze="0.12 0.18 0.25 1"/>
    <global offwidth="2200" offheight="2200" fovy="35"/>
    </visual>

    <asset>
        <texture type="skybox" builtin="gradient"
                        rgb1="0.18 0.22 0.28" rgb2="0.02 0.03 0.05" width="512" height="3072"/>
        <texture name="wood1" file="assets/light-wood.png" type="2d"/>
        <texture name="wood2" file="assets/oak.png" type="2d"/>
        <texture name="wood3" file="assets/wood3.png" type="2d"/>
        <texture name="steel" file="assets/steel.png" type="2d"/>

    <material name="tile_light" rgba="0.7 0.76 0.84 1" specular="0.08" shininess="0.18" emission="0.0"/>
    <material name="tile_dark" rgba="0.26 0.34 0.48 1" specular="0.06" shininess="0.15" emission="0.0"/>
    <material name="wood_block" texture="wood2" shininess="0.3" specular="0.14"/>
    <material name="wood_carried" texture="wood2" shininess="0.46" specular="0.22"/>
    <material name="target_empty" rgba="0.8 0.8 0.25 0.35" emission="0.12" specular="0.03" shininess="0.85"/>
    <material name="target" rgba="0.25 0.8 0.3 0.35" emission="0.12" specular="0.03" shininess="0.85"/>
    <material name="agent" texture="steel" specular="0.7" shininess="0.7" emission="0.05"
                rgba="0.92 0.22 0.18 0.9"/>
    </asset>

    <worldbody>
    <light name="key" pos="5 -3 7" dir="-0.8 0.4 -1" diffuse="0.58 0.5 0.46"
             specular="0.2 0.18 0.17" ambient="0.18 0.18 0.21" cutoff="40" exponent="6"
                     castshadow="true"/>
    <light name="fill" pos="-4 6 4" dir="0 -1 -0.4" diffuse="0.32 0.4 0.55"
             specular="0.12 0.16 0.22" ambient="0.2 0.23 0.28" cutoff="70" exponent="2"/>
    <light name="rim" pos="2 4 6" dir="-0.2 -0.5 -1" diffuse="0.22 0.3 0.44"
             specular="0.14 0.18 0.24" ambient="0.14 0.17 0.21" cutoff="55" exponent="4"/>
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
RESOLUTION = (1600, 1200)
EP_LEN = 15

AGENT_NOT_CARRYING = [3, 5, 6, 8]
AGENT_CARRYING = [4, 7, 9, 11]


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
        pos=[x, y, 0],
        size=[FIELD_WIDTH, FIELD_WIDTH, 0.02],
        material="tile_light" if checker_color else "tile_dark",
    )


def add_box(spec, x, y):
    spec.worldbody.add_geom(
        name="box-%d-%d" % (x, y),
        type=mj.mjtGeom.mjGEOM_BOX,
        pos=[x, y, BLOCK_WIDTH / 2],
        size=[BLOCK_WIDTH / 2, BLOCK_WIDTH / 2, BLOCK_WIDTH / 2],
        material="wood_block",
        rgba=[0.28, 0.24, 0.20, 1.0],  # darker tint for the box
    )


def add_interpolated_box(spec, x, y, z, size=BLOCK_WIDTH / 2):
    spec.worldbody.add_geom(
        name="box-%d-%d-%d" % (x, y, z),
        type=mj.mjtGeom.mjGEOM_BOX,
        pos=[x, y, z],
        size=[size, size, size],
        material="wood_block",
        rgba=[0.28, 0.24, 0.20, 1.0],  # darker tint for the box
    )


def add_carried_box(spec, x, y):
    spec.worldbody.add_geom(
        name="carried-box-%d-%d" % (x, y),
        type=mj.mjtGeom.mjGEOM_BOX,
        pos=[x, y, SPHERE_SIZE * 2.0],
        size=[BLOCK_WIDTH / 5, BLOCK_WIDTH / 5, BLOCK_WIDTH / 5],
        material="wood_block",
        rgba=[0.28, 0.24, 0.20, 1.0],  # darker tint for the box
    )


def add_actor(spec, x, y):
    spec.worldbody.add_geom(
        name="sphere",
        type=mj.mjtGeom.mjGEOM_SPHERE,
        pos=[x, y, SPHERE_SIZE * 1.1],
        size=[SPHERE_SIZE, SPHERE_SIZE, SPHERE_SIZE],
        material="agent",
    )


def add_target(spec, x, y, empty=True):
    spec.worldbody.add_geom(
        name="target-%d-%d" % (x, y),
        type=mj.mjtGeom.mjGEOM_BOX,
        pos=[x, y, BLOCK_WIDTH / 1.8],
        size=[BLOCK_WIDTH / 1.8, BLOCK_WIDTH / 1.8, BLOCK_WIDTH / 1.8],
        material="target" if not empty else "target_empty",
        contype=0,
        conaffinity=0,
    )


def compute_crop_bounds(frame, threshold=10, margin=4):
    max_val = frame.max()
    adjusted_threshold = threshold if max_val > 1.0 else threshold / 255.0
    mask = frame.mean(axis=2) > adjusted_threshold
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]

    if rows.size == 0 or cols.size == 0:
        return (0, frame.shape[0], 0, frame.shape[1])

    top = max(rows[0] - margin, 0)
    bottom = min(rows[-1] + 1 + margin, frame.shape[0])
    left = max(cols[0] - margin, 0)
    right = min(cols[-1] + 1 + margin, frame.shape[1])
    return (top, bottom, left, right)


def calculate_box_pickup_pos(pos, coef):
    pos_diff = SPHERE_SIZE * 2.0 - BLOCK_WIDTH / 2.0
    size_diff = BLOCK_WIDTH / 2 - BLOCK_WIDTH / 5
    return pos + np.array([0, 0, pos_diff]) * (1.0 - coef), BLOCK_WIDTH / 5 + size_diff * coef


def render_trajectory(data, static_model):
    # Create an empty list to store the rendered images
    frames = []
    scene_option = mj.MjvOption()

    # Iterate through the timesteps data
    next_data = data.copy()
    next_data[:-1] = data[1:]
    next_data[-1] = data[-1]
    for state, next_state, step_num in zip(data, next_data, range(len(data))):
        print(f"Rendering frame {step_num}/{len(data)}")
        next_actor_pos = find_agent(next_state)
        curr_actor_pos = find_agent(state)

        if curr_actor_pos is None:
            curr_actor_pos = np.array([0.0, 0.0])
        else:
            curr_actor_pos = curr_actor_pos.astype(float)

        if next_actor_pos is None:
            next_actor_pos = curr_actor_pos.copy()
        else:
            next_actor_pos = next_actor_pos.astype(float)

        curr_actor_state = state[int(curr_actor_pos[0]), int(curr_actor_pos[1])]
        next_actor_state = next_state[int(next_actor_pos[0]), int(next_actor_pos[1])]

        if curr_actor_state in AGENT_CARRYING and next_actor_state in AGENT_NOT_CARRYING:
            put_down = True
            pick_up = False
        elif curr_actor_state in AGENT_NOT_CARRYING and next_actor_state in AGENT_CARRYING:
            put_down = False
            pick_up = True
        else:
            put_down = False
            pick_up = False

        subdivide = (not np.array_equal(next_actor_pos, curr_actor_pos)) or put_down or pick_up

        num_subdivisions = SUBDIV_STEPS if subdivide else 1

        for subdiv_step in range(num_subdivisions):
            interpolated_fraction = (subdiv_step / (SUBDIV_STEPS - 1)) if num_subdivisions > 1 else 0.0
            interp_pos = (next_actor_pos - curr_actor_pos) * interpolated_fraction + curr_actor_pos

            spec = mj.MjSpec.from_string(static_model)
            grid_size = state.shape[0]

            # Add geoms based on the current state
            for i, j in itertools.product(range(grid_size), range(grid_size)):
                add_field(spec, i, j)

                if pick_up or put_down:
                    if np.array_equal(np.array([i, j]), curr_actor_pos):
                        box_fraction = (
                            interpolated_fraction if put_down else (1 - interpolated_fraction) if pick_up else 0.0
                        )
                        box_pos, box_size = calculate_box_pickup_pos(np.array([j, i, BLOCK_WIDTH / 2]), box_fraction)
                        box_z = box_pos[2]
                        add_interpolated_box(spec, i, j, box_z, size=box_size)
                        add_actor(spec, interp_pos[0], interp_pos[1])
                        if state[i, j] in [6, 7, 8, 9]:
                            add_target(spec, i, j, empty=True)
                        continue

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
                    add_target(spec, i, j, empty=False)
                if state[i, j] == 9:  # agent_on_target_with_box_carrying_box
                    add_box(spec, i, j)
                    add_actor(spec, interp_pos[0], interp_pos[1])
                    add_carried_box(spec, interp_pos[0], interp_pos[1])
                    add_target(spec, i, j, empty=False)
                if state[i, j] == 10:  # box_on_target
                    add_box(spec, i, j)
                    add_target(spec, i, j, empty=False)
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
                grid_center = (grid_size - 1) / 2
                lookat_x = grid_center
                lookat_y = grid_center
                # Use a free camera with a broader framing to keep the full board in sight
                camera = mj.MjvCamera()
                camera.type = mj.mjtCamera.mjCAMERA_FREE
                camera.distance = max(11.0, grid_size * 3.0)
                camera.azimuth = -45
                camera.elevation = -45
                camera.lookat[:] = [lookat_x, lookat_y, -0.05]
                renderer.update_scene(data_mj, camera, scene_option=scene_option)
                renderer.scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = True
                renderer.scene.flags[mj.mjtRndFlag.mjRND_SKYBOX] = True
                renderer.scene.flags[mj.mjtRndFlag.mjRND_FOG] = True
                renderer.scene.flags[mj.mjtRndFlag.mjRND_HAZE] = True
                renderer.scene.flags[mj.mjtRndFlag.mjRND_REFLECTION] = True
                frames.append(renderer.render())

    return frames


frames = render_trajectory(data[:EP_LEN, ENV_IDX], static_model)

crop_top, crop_bottom, crop_left, crop_right = compute_crop_bounds(frames[0])
if (crop_top, crop_bottom, crop_left, crop_right) != (0, frames[0].shape[0], 0, frames[0].shape[1]):
    frames = [frame[crop_top:crop_bottom, crop_left:crop_right] for frame in frames]


fig, ax = plt.subplots(figsize=(frames[0].shape[1] / 50, frames[0].shape[0] / 50), dpi=100)
ax.axis("off")
im = ax.imshow(frames[0], animated=True)

output_path = "output_above.gif"


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
