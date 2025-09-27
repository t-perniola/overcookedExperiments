from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, Overcooked
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.agents.agent import GreedyHumanModel
from overcooked_ai_py.planning.planners import MediumLevelActionManager
from tensorflow.keras.models import load_model
import random
import numpy as np
import matplotlib.pyplot as plt
import imageio
import tensorflow as tf

# utils.py
# This file contains utility functions and classes for Overcooked environments

# ------------------------------------ GENERAL OVERCOOKED CLASS ---------------------------------------------

# A generalized Overcooked environment that can handle multiple layouts
class GeneralizedOvercooked:   
    def __init__(self, layouts, info_level=0, horizon=400):
        self.envs = []
        for layout in layouts:
            base_mdp = OvercookedGridworld.from_layout_name(layout)
            base_env = OvercookedEnv.from_mdp(base_mdp, info_level=info_level, horizon=horizon)
            env = Overcooked(base_env=base_env, featurize_fn=base_env.featurize_state_mdp)
            self.envs.append(env)
        self.layout_dict = {idx: layout for layout, idx in zip(layouts, np.arange(len(layouts)))}
        self.cur_env = self.envs[0]
        self.observation_space, self.action_space = self.cur_env.observation_space, self.cur_env.action_space

    def reset(self, idx=None, force_random=False):
        """
        Reset the environment.
        - If idx is given → switch to that env and reset.
        - If force_random=True → sample a random env and reset.
        - If neither → just reset the current env (self.cur_env).
        """
        if force_random:
            idx = np.random.randint(0, len(self.envs))
            self.cur_env = self.envs[idx]
            return self.cur_env.reset()

        if idx is not None:
            self.cur_env = self.envs[idx]
            return self.cur_env.reset()

        # Default: reset current env, don't randomize
        return self.cur_env.reset()
    
    def step(self, *args):
        return self.cur_env.step(*args)
    
    def render(self, *args):
        return self.cur_env.render(*args)
    
    def switch_to_env(self, idx=None):
        if idx is None:
            idx = np.random.choice(len(self.envs))
        self.cur_env = self.envs[idx]

    def sample_env(self, prob = None):
        if prob is None: # Uniform sampling
            idx = np.random.choice(len(self.envs))
        else: # Weighted sampling
            idx = np.random.choice(len(self.envs), p=prob)
        self.cur_env = self.envs[idx]
        return idx
    

# --------------------------- AUX CLASS FOR PARTNER POOLING ------------------------------------

class GreedyHumanAgent:
    def __init__(self, layouts=["cramped_room", "asymmetric_advantages", "bottleneck", "counter_circuit", "coordination_ring"]):
        # Create a dictionary to hold GreedyHumanModel agents for each layout
        self.gha_dict = {}
        self.create_greedy_human_agents(layouts)

    def create_greedy_human_agents(self, layouts):
        mlams = [] # one per layout 
        mdps = [] # one per layout
        DEFAULT_MLAM_PARAMS = {
            "wait_allowed": False,
            "counter_goals": [],
            "counter_drop": [],
            "counter_pickup": [],
            "same_motion_goals": True,
            "start_orientations": False
        }
        for layout in layouts:
            mdp = OvercookedGridworld.from_layout_name(layout)
            mdps.append(mdp)
            mlam = MediumLevelActionManager.from_pickle_or_compute(mdp, mlam_params=DEFAULT_MLAM_PARAMS, custom_filename=None)
            mlams.append(mlam)
            self.gha_dict[layout] = GreedyHumanModel(mlam)

    def select_gh_agent(self, layout_name):
        return self.gha_dict[layout_name]
    

# --------------------------- WRAPPER FOR SCRIPTED AGENTS ------------------------------------

class WrappedPartner:
    def __init__(self, agent, action_fn="action", action_space=6, actions_dict=None, agent_index=1):
        """
        agent: underlying scripted/random agent
        agent_index: which agent this wrapper controls (default = 1 = partner)
        """
        self.agent = agent
        self.action_fn = getattr(agent, action_fn)
        self.is_wrapped = True
        self.agent_index = agent_index
        self.action_space = action_space

        if actions_dict is None:
            actions_dict = {
                0: "Up", 1: "Down", 2: "Right", 3: "Left", 4: "Stay", 5: "Interact"
            }
        # lowercase mapping
        self.action_to_idx = {v.lower(): k for k, v in actions_dict.items()}

    def actor_model(self, state, training=False):
        """
        For wrapped scripted agents: takes OvercookedState, not obs vec.
        """
        action, _ = self.action_fn(state)

        if isinstance(action, tuple) and len(action) > 1:
            action = action[self.agent_index]
        elif isinstance(action, tuple):
            action = action[0]

        if isinstance(action, str):
            action_idx = self.action_to_idx[action.lower()]
        else:
            action_idx = int(action)

        probs = np.zeros((1, self.action_space), dtype=np.float32)
        probs[0, action_idx] = 1.0
        return probs
    
    def set_agent_index(self, index):
        self.agent_index = index
        self.agent.set_agent_index(index)


# --------------------------- UTILITIES -----------------------------------
    
# Utility function to render the environment    
def render_env(env):
    frame = env.render() 
    #print(frame.shape)  # 928x1056 resolution
    plt.imshow(frame)
    plt.axis('off')
    plt.show()

def epsilon_greedy(probs):
    epsilon = 0.01
    if random.random() < epsilon:
        action = np.random.randint(len(probs))  # uniform random
    else:
        action = np.argmax(probs)
    return action

def rollout_with_gif(env_name, actor, gif_path="episode.gif", episodes=1, sampling_type="categorical"):
    env = GeneralizedOvercooked([env_name])
    steps = env.cur_env.base_env.horizon
    frames = []

    for _ in range(episodes):
        obs = env.reset()
        obs = obs["both_agent_obs"]
        done = False
        step = 0
        frames.append(env.render())  # Initial frame

        while not done and step < steps:
            obs_A1 = obs[0]
            obs_A2 = obs[1]

            action_probs_A1 = actor.predict(np.array([obs_A1]), verbose=0)[0]
            action_probs_A2 = actor.predict(np.array([obs_A2]), verbose=0)[0]

            if sampling_type == "categorical":
                action_A1 = tf.random.categorical(tf.math.log([action_probs_A1]), 1)[0, 0].numpy()
                action_A2 = tf.random.categorical(tf.math.log([action_probs_A2]), 1)[0, 0].numpy()
            else:
                action_A1 = epsilon_greedy(action_probs_A1)
                action_A2 = epsilon_greedy(action_probs_A2)

            next_obs, _, done, _ = env.step((action_A1, action_A2))
            obs = next_obs["both_agent_obs"]
            frames.append(env.render())
            step += 1

    imageio.mimsave(f"../gifs/{gif_path}", frames, fps=5)
    print(f"GIF saved as {gif_path}")

def rollout_with_gif_unseen(env_name, actor, gif_path="episode.gif", sampling_type="categorical", thresh=10):
    env = GeneralizedOvercooked([env_name])
    steps = env.cur_env.base_env.horizon
    frames = []
    i = 0

    while True:
        obs = env.reset()
        obs = obs["both_agent_obs"]
        done = False
        step = 0
        frames = []
        frames.append(env.render())  # Initial frame
        ep_cum_rew = 0

        while not done and step < steps:
            obs_A1 = obs[0]
            obs_A2 = obs[1]

            action_probs_A1 = actor.predict(np.array([obs_A1]), verbose=0)[0]
            action_probs_A2 = actor.predict(np.array([obs_A2]), verbose=0)[0]

            if sampling_type == "categorical":
                action_A1 = tf.random.categorical(tf.math.log([action_probs_A1]), 1)[0, 0].numpy()
                action_A2 = tf.random.categorical(tf.math.log([action_probs_A2]), 1)[0, 0].numpy()
            else:
                action_A1 = epsilon_greedy(action_probs_A1)
                action_A2 = epsilon_greedy(action_probs_A2)

            next_obs, sparse_reward, done, info = env.step((action_A1, action_A2))
            shaped_reward = info["shaped_r_by_agent"]
            ep_cum_rew += sparse_reward + np.sum(shaped_reward)
            obs = next_obs["both_agent_obs"]
            frames.append(env.render())
            step += 1

        i += 1
        print(f"Total reward episode {i}: {ep_cum_rew}")
        if ep_cum_rew > thresh:  # Only save if a certain reward was achieved
            break

    imageio.mimsave(f"../gifs/{gif_path}", frames, fps=10)
    print(f"GIF saved as {gif_path}")


def save_models(mappo, actor_name, critic_name):
    mappo.actor_model.save(f"../saved_models/{actor_name}.keras")
    mappo.critic_model.save(f"../saved_models/{critic_name}.keras")

def load_models(actor_name, critic_name):
    actor = load_model(f"../saved_models/{actor_name}.keras")
    critic = load_model(f"../saved_models/{critic_name}.keras")
    return actor, critic

def mean_list(lst, N=None):
    if N is None:
        return np.mean(lst) if lst else 0.0
    else:
        return np.mean(lst[-N:]) if len(lst) >= N else np.mean(lst)

def import_gh_agent(pool, gha, layout):
    greedy_agent = gha.select_gh_agent(layout)
    pool.append(greedy_agent)
    return pool

def build_augm_obs(state):
    """
    Extract context from the Overcooked state.
    Context includes:
    - Agent positions
    - Facing positions
    - Held objects
    """
    # Extract relevant information from the state
    # A


    return augm_obs