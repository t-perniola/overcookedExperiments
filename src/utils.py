from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, Overcooked
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from tensorflow.summary import create_file_writer
from tensorflow.keras.models import load_model
import datetime
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

def save_models(mappo, actor_name, critic_name):
    mappo.actor_model.save(f"../saved_models/{actor_name}.keras")
    mappo.critic_model.save(f"../saved_models/{critic_name}.keras")

def load_models(actor_name, critic_name):
    actor = load_model(f"../saved_models/{actor_name}.keras")
    critic = load_model(f"../saved_models/{critic_name}.keras")
    return actor, critic


# --------------------------- RL ALGO -------------------------------------
# MAPPO: Try 1
class MAPPO_v0:
    def __init__(self, env, input_shape, action_dim, num_agents=2, fine_tune=False):
        self.env = env
        self.obs_dim = input_shape[0]
        self.act_dim = action_dim
        self.num_agents = num_agents
        self.fine_tune = fine_tune

        # Shared networks
        self.actor = self.build_actor()
        self.critic = self.build_critic()

        self.actor_optimizer = Adam(5e-4)
        self.critic_optimizer = Adam(2e-4)

        # Logging
        logdir = (
            "logs/mappo_finetune/" if fine_tune else "logs/mappo_clean/"
        ) + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer = create_file_writer(logdir)
        self.writer = create_file_writer(logdir)

        # Hyperparameters
        self.ts_per_batch = 4096
        self.steps_per_episode = env.cur_env.base_env.horizon
        self.gamma = 0.99
        self.lam = 0.90
        self.epochs = 4
        self.clip_init = 0.15
        self.batch_size = 512
        self.entropy_coeff = 0.01

        # fine‑tune overrides -----------------------------------------
        if self.fine_tune:
            self.actor_optimizer.learning_rate  = 2e-4
            self.critic_optimizer.learning_rate = 1e-4
            self.batch_size  = 256
            self.epochs      = 6
            self.ent_init    = 0.03   # extra exploration early
            # final clip & entropy values we decay towards
            self.clip_final  = 0.05
            self.ent_final   = 0.0
            self.ft_idx      = 0      # counts rollouts during fine‑tune
        else:
            self.clip_final = 0.05  # still used for linear decay prints
            self.ent_final  = 0.0

        # Curriculum tracking
        self.avg_return_per_rollout = {name: [] for name in self.env.layout_dict.values()}
        self.phase = 1
        self.phase2_start_steps = None  # filled when Phase 2 triggers
        self.env_steps = 0

    def build_actor(self):
        return models.Sequential([
            Input(shape=(self.obs_dim,)),
            Dense(128, activation='relu'),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(self.act_dim, activation='softmax')
        ])

    def build_critic(self):
        inp = Input(shape=(self.obs_dim * self.num_agents,))  # no trailing comma

        x = Dense(256, activation='relu')(inp)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)

        h1 = Dense(1, activation='linear')(x)
        h2 = Dense(1, activation='linear')(x)

        output = tf.keras.layers.Concatenate(axis=1)([h1, h2])  # shape: (batch, 2)

        return tf.keras.Model(inputs=inp, outputs=output)

    def reset_env(self):
        obs = self.env.reset()
        return obs["both_agent_obs"]

    def compute_gae(self, rewards, values, dones):
        advantages = []
        last_advantage = 0
        for t in reversed(range(len(rewards))):
            next_value = values[t + 1] if t + 1 < len(values) else 0
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            last_advantage = delta + self.gamma * self.lam * (1 - dones[t]) * last_advantage
            advantages.insert(0, last_advantage)
        return tf.convert_to_tensor(advantages, dtype=tf.float32)

    def compute_policy(self, obs, actions):
        probs = self.actor(obs)
        indices = tf.stack([tf.range(tf.shape(probs)[0]), actions], axis=1)
        selected = tf.gather_nd(probs, indices)
        return tf.math.log(selected + 1e-8)
    
    def reward_scheduling(self, env_steps):
        """Return a scalar ∈ [0.3, 1.0] multiplying shaped reward.
        * Phase 1 : always 1.0
        * Phase 2+: linearly decays from 1.0 → 0.3 over 100 000 steps starting
          at the moment Phase 2 is triggered (self.phase2_start_steps).
        """
        if self.phase == 1 or self.phase2_start_steps is None:
            return 1.0
        elapsed = env_steps - self.phase2_start_steps
        frac    = np.clip(elapsed / 100000.0, 0.0, 1.0)
        base = 1.0 - 0.7 * frac

        # floors by layout --------------------------------------------
        if self.cur_layout == 'bottleneck':
            return max(base, 0.5)
        if self.cur_layout == 'cramped_room':
            return max(base, 0.4)
        return base
    
    def _fine_tune_sampler(self):
        """Return sampling probabilities [cramped, asymmetric, bottleneck]."""
        r = self.ft_idx
        if r < 40:
            return [0.60, 0.10, 0.30]
        if r < 80:
            return [0.45, 0.10, 0.45]
        return [0.40, 0.20, 0.40]

    def dynamic_layout_switching(self):
        cramped, asym, bottleneck = (self.env.layout_dict[i] for i in range(3))
        mean = lambda k: np.mean(self.avg_return_per_rollout[k]) if self.avg_return_per_rollout[k] else 0.0

        # Phase transitions -------------------------------------------
        if self.phase == 1 and mean(cramped) >= 30:
            print("\n🎓 Phase 2 – Interleaved layouts")
            self.phase = 2
            self.phase2_start_steps = self.env_steps
        if self.phase == 2 and mean(cramped) >= 50 and mean(asym) >= 20 and mean(bottleneck) >= 20:
            print("\n🌀 Phase 3 – Random layouts")
            self.phase = 3

        # choose env ---------------------------------------------------
        if self.fine_tune:
            probs = self._fine_tune_sampler()
            idx   = np.random.choice(3, p=probs)
        else:
            if self.phase == 1:
                idx = 0
            elif self.phase == 2:
                idx = np.random.choice(3, p=[0.5, 0.3, 0.2])
            else:
                idx = np.random.choice(3)

        self.env.switch_to_env(idx)
        self.cur_layout = self.env.layout_dict[idx]  # remember for reward flooring
        return self.cur_layout

    def rollout(self):
        """Collect one batch of size `ts_per_batch` *per agent*.
        Side effect: updates `self.env_steps`.
        """
        # storage ------------------------------------------------------------
        batch_obs_critic            = []
        batch_obs_actor_per_agent   = [[] for _ in range(self.num_agents)]
        batch_actions_per_agent     = [[] for _ in range(self.num_agents)]
        batch_log_probs_per_agent   = [[] for _ in range(self.num_agents)]
        batch_rewards_per_agent     = [[] for _ in range(self.num_agents)]
        batch_dones_per_agent       = [[] for _ in range(self.num_agents)]
        batch_values_per_agent      = [[] for _ in range(self.num_agents)]
        batch_lens_per_episode      = []

        flat_values_per_agent       = [[] for _ in range(self.num_agents)]

        ts = 0
        while ts < self.ts_per_batch:
            # reset env ------------------------------------------------------
            obs_n = self.reset_env()
            ep_len = 0

            step_rews, step_dones, step_vals = ( [[] for _ in range(self.num_agents)]
                                                for _ in range(3) )

            # run one episode -----------------------------------------------
            for _ in range(self.steps_per_episode):
                ep_len += 1
                actions = []

                # ── each agent picks an action ────────────────────────────
                for i in range(self.num_agents):
                    obs_i = tf.convert_to_tensor([obs_n[i]], dtype=tf.float32)
                    probs = self.actor(obs_i, training=False).numpy()[0]
                    act   = tf.random.categorical(tf.math.log([probs]), 1)[0,0].numpy()
                    logp  = np.log(probs[act] + 1e-8)

                    batch_obs_actor_per_agent[i].append(obs_n[i])
                    batch_actions_per_agent[i].append(act)
                    batch_log_probs_per_agent[i].append(logp)
                    actions.append(int(act))

                # ── centralized critic input ──────────────────────────────
                joint_obs = np.concatenate(obs_n, axis=-1)
                batch_obs_critic.append(joint_obs)

                V_all = self.critic(tf.convert_to_tensor([joint_obs], dtype=tf.float32)).numpy().flatten()
                for i in range(self.num_agents):
                    step_vals[i].append(V_all[i])

                # ── env step ───────────────────────────────────────────────
                next_obs, total_sparse_r, done, info = self.env.step(actions)
                shaped_r  = info.get("shaped_r_by_agent", [0]*self.num_agents)
                decay = self.reward_scheduling(self.env_steps)
                if total_sparse_r == 20:
                    print(f"  🍛 Dish delivered! 😍")
                self.env_steps += 1

                for i in range(self.num_agents):
                    shaped_r[i] *= decay 
                    step_rews[i].append(shaped_r[i] + total_sparse_r)
                    step_dones[i].append(int(done))

                ts += 1
                if done or ts >= self.ts_per_batch:
                    break
                obs_n = next_obs["both_agent_obs"]

            # episode done ---------------------------------------------------
            batch_lens_per_episode.append(ep_len)
            for i in range(self.num_agents):
                batch_rewards_per_agent[i].append(step_rews[i])
                batch_dones_per_agent[i].append(step_dones[i])
                batch_values_per_agent[i].append(step_vals[i])
                flat_values_per_agent[i].extend(step_vals[i])

        # convert lists → arrays
        for i in range(self.num_agents):
            batch_obs_actor_per_agent[i] = np.vstack(batch_obs_actor_per_agent[i])
            batch_actions_per_agent[i]   = np.asarray(batch_actions_per_agent[i], dtype=np.int32)
            batch_log_probs_per_agent[i] = np.asarray(batch_log_probs_per_agent[i], dtype=np.float32)

        return (batch_obs_critic, batch_obs_actor_per_agent, batch_actions_per_agent,
                batch_log_probs_per_agent, batch_lens_per_episode, batch_rewards_per_agent,
                batch_values_per_agent, batch_dones_per_agent, flat_values_per_agent)

    def learn(self, total_timesteps):
        t_so_far = 0
        rollout_counter = 0
        print("▶ Starting learning process!\nPHASE 1️⃣: cramped room")

        while t_so_far < total_timesteps:
            current_layout = self.dynamic_layout_switching()
            print(f"🎲 Layout sampled: {current_layout}")

            rollout_counter += 1
            print(f"\nRollout: {rollout_counter}\n 📦 Gathering transitions...")

            (batch_obs_critic, batch_obs_actor_per_agent, batch_actions_per_agent,
             batch_log_probs_per_agent, batch_lens_per_episode, batch_rewards_per_agent,
             batch_values_per_agent, batch_dones_per_agent, flat_values_per_agent) = self.rollout()

            steps_collected = sum(batch_lens_per_episode)
            t_so_far += steps_collected
            self.env_steps += steps_collected

            # linear decay of clip & entropy during fine‑tune -------------
            if self.fine_tune:
                decay_frac = min(self.ft_idx / 200.0, 1.0)
                self.clip_range   = self.clip_init - (self.clip_init - self.clip_final) * decay_frac
                self.entropy_coef = self.ent_init  - (self.ent_init - self.ent_final) * decay_frac
            else:
                self.clip_range   = self.clip_init
                self.entropy_coef = self.ent_init

            # Compute advantages
            flat_adv = tf.concat([
                self.compute_gae(
                    [r for ep in batch_rewards_per_agent[i] for r in ep],
                    [v for ep in batch_values_per_agent[i] for v in ep],
                    [d for ep in batch_dones_per_agent[i] for d in ep]
                ) for i in range(self.num_agents)
            ], axis=0)

            flat_obs_actor = tf.convert_to_tensor(np.vstack(batch_obs_actor_per_agent), dtype=tf.float32)
            flat_actions = tf.convert_to_tensor(np.concatenate(batch_actions_per_agent), dtype=tf.int32)
            flat_logps = tf.convert_to_tensor(np.concatenate(batch_log_probs_per_agent), dtype=tf.float32)
            flat_obs_critic = tf.convert_to_tensor(np.repeat(batch_obs_critic, self.num_agents, axis=0), dtype=tf.float32)

            A_k = (flat_adv - tf.reduce_mean(flat_adv)) / (tf.math.reduce_std(flat_adv) + 1e-8)
            indices = tf.range(len(flat_actions))

            print(" 📖 Learning...")

            for _ in range(self.epochs):
                shuffled = tf.random.shuffle(indices)
                for start in range(0, len(indices), self.batch_size):
                    mb_idx = shuffled[start:start+self.batch_size]

                    mb_obs_actor = tf.gather(flat_obs_actor, mb_idx)
                    mb_actions = tf.gather(flat_actions, mb_idx)
                    mb_logps = tf.gather(flat_logps, mb_idx)
                    mb_A_k = tf.gather(A_k, mb_idx)
                    mb_adv = tf.gather(flat_adv, mb_idx)
                    mb_obs_critic = tf.gather(flat_obs_critic, mb_idx)
                    mb_agent_ids = mb_idx % self.num_agents

                    # ACTOR
                    with tf.GradientTape() as tape:
                        probs = self.actor(mb_obs_actor, training=True)
                        probs = tf.clip_by_value(probs, 1e-8, 1.0)
                        logp = tf.math.log(tf.reduce_sum(probs * tf.one_hot(mb_actions, self.act_dim), axis=1))
                        ratio = tf.exp(logp - mb_logps)
                        surr1 = ratio * mb_A_k
                        surr2 = tf.clip_by_value(ratio, 1 - self.clip_range, 1 + self.clip_range) * mb_A_k
                        entropy = -tf.reduce_sum(probs * tf.math.log(probs), axis=1)
                        progress   = self.env_steps / float(total_timesteps)
                        ent_coeff  = self.entropy_coeff * max(0.0, 1.0 - progress*1.5)  # zero after 150 %
                        actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2)) - ent_coeff * tf.reduce_mean(entropy)
                        
                    grads = tape.gradient(actor_loss, self.actor.trainable_variables)
                    grads = [tf.clip_by_norm(g, 0.5) if g is not None else None for g in grads]
                    self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

                    # CRITIC
                    with tf.GradientTape() as tape:
                        V_pred = self.critic(mb_obs_critic, training=True)
                        rows = tf.range(tf.shape(V_pred)[0])
                        V_selected = tf.gather_nd(V_pred, tf.stack([rows, mb_agent_ids], axis=1))
                        values_matrix = tf.convert_to_tensor(
                            np.vstack(flat_values_per_agent),    # shape (num_agents, ts_per_batch)
                            dtype=tf.float32
                        )
                        mb_values = tf.gather_nd(
                            values_matrix,
                            tf.stack([mb_agent_ids, mb_idx % self.ts_per_batch], axis=1)
                        )  # shape (mini_batch,)
                        returns      = mb_adv + mb_values
                        critic_loss  = tf.reduce_mean(tf.square(returns - V_selected))

                    grads = tape.gradient(critic_loss, self.critic.trainable_variables)
                    grads = [tf.clip_by_norm(g, 0.5) if g is not None else None for g in grads]
                    self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

            # fine‑tune rollout counter & debug print ----------------------
            if self.fine_tune:
                self.ft_idx += 1
                if self.ft_idx % 5 == 0:
                    c_mean = np.mean(self.avg_return_per_rollout['cramped_room'][-5:]) if self.avg_return_per_rollout['cramped_room'] else 0.0
                    b_mean = np.mean(self.avg_return_per_rollout['bottleneck'][-5:])   if self.avg_return_per_rollout['bottleneck'] else 0.0
                    print(f"[FT] Rollout {self.ft_idx}  cramped μ {c_mean:.1f}  bottleneck μ {b_mean:.1f}")

            with self.writer.as_default():
                tf.summary.scalar("Loss/Actor", actor_loss, step=t_so_far)
                tf.summary.scalar("Loss/Critic", critic_loss, step=t_so_far)
                self.writer.flush()

            # --- compute avg return from true rewards ---------------------------
            episode_returns = [
                np.sum(ep)                            # sum over timesteps
                for agent_eps in batch_rewards_per_agent   # loop over agents
                for ep in agent_eps                        #   and their episodes
            ]
            avg_return = float(np.mean(episode_returns))
            self.avg_return_per_rollout[current_layout].append(avg_return)
            print(f" ➗ Average return: {avg_return:.3f}")
            with self.writer.as_default():
                tf.summary.scalar("Return/Avg", avg_return, step=t_so_far)
                self.writer.flush()