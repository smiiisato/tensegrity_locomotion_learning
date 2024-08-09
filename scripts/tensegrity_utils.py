import importlib.util
import sys
import numpy as np
import mujoco
import csv

# Function to load a module from a given path
def load_module(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def rescale_actions(low, high, action):
    """
    remapping the normalized actions from [-1, 1] to [low, high]
    """
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    rescaled_action = action * d + m
    return rescaled_action

def get_projected_gravity(data, model):
        """
        calculate the projected gravity on the body frame
        """
        gravity = np.array([0., 0., -1.0]) # gravity direction in the world frame
        projected_gravity = np.zeros(18)
        for i in range(6):
            rot_mat = data.xmat[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link{}".format(i+1))].reshape(3, 3)
            projected_gravity[i*3:i*3+3] = rot_mat.T @ gravity # gravity projected on the body frame
        return projected_gravity

def calculate_angular_momentum(data, model, qpos, qvel, com_position, com_vel, body_inertial):
        total_angular_momentum = np.zeros(3)
        body_mass = 0.76
        links_position = qpos.reshape(-1, 7)[:, 0:3]
        links_velocity = qvel.reshape(-1, 6)[:, 0:3]
        links_ang_vel = qvel.reshape(-1, 6)[:, 3:]
        for i in range(6):
            rot_mat = data.xmat[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link{}".format(i+1))].reshape(3, 3)  # R_i: rotation matrix
            angular_momentum = body_mass*np.cross((links_position[i] - com_position), (links_velocity[i] - com_vel))
            angular_momentum += rot_mat @ body_inertial[i] @ rot_mat.transpose() @ links_ang_vel[i]

            total_angular_momentum += angular_momentum

        return total_angular_momentum

def save_log_data(step, log_data, log_file):
        with open(log_file, 'a') as f:
            writer = csv.writer(f)
            data = [step] + list(log_data)
            writer.writerow(data)