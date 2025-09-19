from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras import models, Model
from tensorflow.summary import create_file_writer
from tensorflow.keras.optimizers import AdamW
from utils import mean_list, import_gh_agent, WrappedPartner
import numpy as np
import tensorflow as tf
import datetime

# Defining the NEURAL NETWORK class
class NN():
    def __init__(self, obs_dim, act_dim, actor_units, critic_units, num_agents):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.actor_units = actor_units
        self.critic_units = critic_units
        self.num_agents = num_agents
    
    # ACTOR MODEL
    def build_actor(self):
        return models.Sequential(
            [Input(shape=(self.obs_dim,))] +
            [Dense(units, activation = "relu") for units in self.actor_units] +
            [Dense(self.act_dim, activation = "softmax")]
        )
    
    # CRITIC MODEL: single-headed
    def build_critic(self):
        joint_obs_shape = self.obs_dim * self.num_agents
        return models.Sequential(
            [Input(shape=(joint_obs_shape,))] +
            [Dense(units, activation = "relu") for units in self.critic_units] +
            [Dense(1, activation = "linear")]
        )

    # Multi-head critic (one for each agent)
    def build_mh_critic(self):
        inp = Input(shape = (self.obs_dim * self.num_agents,))

        # Add hidden layers
        h = inp
        for units in self.critic_units:
            h = Dense(units, activation = "relu")(h)

        # Add output head
        head_list = []
        for _ in range(self.num_agents):
            head = Dense(1, activation = "linear")(h)
            head_list.append(head)
        output = Concatenate(axis = 1)([*head_list])

        return Model(inputs=inp, outputs=output)
    
    # Augmented actor (with role input)
    def build_augmented_actor(self, role_dim):
        return models.Sequential(
            [Input(shape=(self.obs_dim + role_dim,))] +
            [Dense(units, activation = "relu") for units in self.actor_units] +
            [Dense(self.act_dim, activation = "softmax")]
        )
    

# Storing trajectories over episodes
class Buffer:
    def __init__(self):
        self.data = {k: [] for k in ["obs","j_obs","act","logp","val","rew","done"]}

    def add(self, obs, j_obs, act, logp, val, rew, done):
        self.data["obs"].append(obs)
        self.data["j_obs"].append(j_obs)
        self.data["act"].append(act)
        self.data["logp"].append(logp)
        self.data["val"].append(val)
        self.data["rew"].append(rew)
        self.data["done"].append(done)


# Defining the general Multi-Agent PPO algorithm
class MAPPO:
    def __init__(self, env, env_params):
        self.env = env
        self.num_agents = env_params["num_agents"]
        self.act_dim = env_params["act_dim"]
        self.horizon = self.env.cur_env.base_env.horizon   
        self.define_models(env_params)
        self.init_hyperparams()
        self.logging()
    
    def logging(self):
        self.avg_return_per_rollout = {name: [] for name in self.env.layout_dict.values()}
        self.phase = 1
        logdir = ("../logs/mappo") + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer = create_file_writer(logdir)

    def reset_env(self):
        obs = self.env.reset()
        return obs["both_agent_obs"]

    def init_hyperparams(self):
        self.gamma = 0.99
        self.lambd = 0.95
        self.entropy_coeff = 0.01
        self.clip_range = 0.2


    def compute_gae(self, buffer, last_value=None):
        num_agents = self.num_agents
        state_values = np.array(buffer.data["val"], dtype=np.float32).reshape(-1, num_agents)
        rewards      = np.array(buffer.data["rew"], dtype=np.float32).reshape(-1, num_agents)
        dones        = np.array(buffer.data["done"], dtype=np.float32).reshape(-1, num_agents)

        T = rewards.shape[0]
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)

        if last_value is None:
            last_value = np.zeros(num_agents)
        else:
            last_value = np.array(last_value).reshape(num_agents)

        for agent in range(num_agents):
            adv = 0.0
            for t in reversed(range(T)):
                if t == T - 1:
                    next_value = last_value[agent]
                    next_nonterminal = 1.0 - dones[t, agent]
                else:
                    next_value = state_values[t+1, agent]
                    next_nonterminal = 1.0 - dones[t+1, agent]
                delta = rewards[t, agent] + self.gamma * next_value * next_nonterminal - state_values[t, agent]
                adv = delta + self.gamma * self.lambd * next_nonterminal * adv
                advantages[t, agent] = adv
                returns[t, agent] = advantages[t, agent] + state_values[t, agent]

        # Flatten back to original buffer order
        buffer.data["adv"] = advantages.flatten().tolist()
        buffer.data["ret"] = returns.flatten().tolist()
        return advantages

    def define_models(self, env_params):
        nn = NN(**env_params)
        self.actor_model = nn.build_actor()
        self.critic_model = nn.build_critic()
        self.actor_optimizer = AdamW(learning_rate=5e-4, weight_decay=1e-4)
        self.critic_optimizer = AdamW(learning_rate=2e-4, weight_decay=1e-4)

    def sample_action(self, probs):
        """Sample action and logp from a categorical defined by probs."""
        action = np.random.choice(len(probs), p=probs)
        logp = np.log(probs[action] + 1e-8)
        return action, logp

    def reward_shaping(self, n_rollout, rew, shaping_type, lin_k = 0.002, exp_k = 0.01):
        if shaping_type == "linear":
            rew = max(0, rew - (lin_k* n_rollout))
        elif shaping_type == "exp":
            rew = max(0, rew * np.exp(-exp_k * n_rollout))
        else: rew = rew
        return rew 
    
    
    # COLLECTING TRAJECTORIES
    def rollout(self, episodes, rew_shaping, env_idx = None):
        buffer = Buffer()           
        
        # Loop over episodes
        for _ in range(episodes):
            if env_idx is not None:
                obs = self.env.reset(idx=env_idx)
            else:
                obs = self.env.reset()
            obs = obs["both_agent_obs"]  
            deliveries = 0 # count how many deliveries within one ep.

            # Loop over steps (400)
            for _ in range(self.horizon):
                # --- State-value estimation ---
                joint_obs = np.concatenate(obs, axis=-1).reshape(1, -1)
                V_s = float(self.critic_model(joint_obs).numpy().squeeze()) # store a scalar

                # --- Sample actions per agent
                # and stepping into the env. ---
                actions = []
                logps = []

                for a in range(self.num_agents):
                    obs_a = np.expand_dims(obs[a], axis=0)   # (1, 96)
                    probs = self.actor_model(obs_a, training=False).numpy()[0]
                    action, logp = self.sample_action(probs=probs)
                    actions.append(int(action))
                    logps.append(float(logp))
                
                # Step env.
                next_obs, sparse_reward, done, info = self.env.step(actions)

                # --- Reward handling ---
                total_shaped = sum(info['shaped_r_by_agent'])    
                team_reward = sparse_reward + total_shaped # scalar, obv
                if sparse_reward == 20:
                    print("   🍛 Delivered!")
                    deliveries += 1

                # Apply decaying
                team_reward = self.reward_shaping(n_rollout=self.num_rollout,
                                                  rew=team_reward,
                                                  shaping_type=rew_shaping)

                # Assign team reward to each agent
                rewards = [team_reward] * self.num_agents

                # --- Store transitions per agent ---
                for a in range(self.num_agents):
                    buffer.add(
                        obs = obs[a],
                        j_obs = joint_obs.squeeze(0),
                        val = V_s,
                        act = actions[a],
                        logp = logps[a],
                        rew = rewards[a],
                        done = done
                    )
                # --- Update obs ---
                obs = next_obs
                obs = obs["both_agent_obs"]

        print(f"  🍛 Delivered dishes: {deliveries}")

        joint_obs = np.concatenate(obs, axis=-1).reshape(1, -1)
        last_value = float(self.critic_model(joint_obs, training=False).numpy().squeeze())

        # GAE computation
        self.compute_gae(buffer, last_value)
        return buffer


    # MAIN LEARNING FUNCTION
    def train_loop(self, epochs = 6, ep_per_rollout = 8, num_rollouts = 10, mb_size = 64, rew_shaping = "None"):
        print("▶ Starting learning process!\nPHASE 1️⃣: cramped room")
        self.num_rollout = 0

        # Loop over total number of rollouts
        for r in range(num_rollouts):
            # --- Layout manager ---
            current_layout = self.env.layout_dict[0]
            print(f"\n🎲 Layout sampled: {current_layout}") if not self.phase == 1 else None

            # --- Gather transitions from one rollout ---
            print(f"\nRollout: {r+1}\n 📦 Gathering transitions...")
            t_buffer = self.rollout(episodes = ep_per_rollout, rew_shaping=rew_shaping)
            transitions = {k: np.array(v) for k, v in t_buffer.data.items()}
            self.num_rollout = r

            # --- Batching and PPO updates ---
            N = len(transitions["obs"])

            print(" 📖 Learning...")

            # Looping over #epochs
            for _ in range(epochs):
                indices = np.arange(N)
                np.random.shuffle(indices)

                # Looping over minibatches
                for start in range(0, N, mb_size):
                    end = start + mb_size
                    mb_idx = indices[start:end]

                    obs_mb   = transitions["obs"][mb_idx]
                    act_mb   = transitions["act"][mb_idx]
                    logp_old = transitions["logp"][mb_idx]
                    adv_mb   = transitions["adv"][mb_idx]
                    ret_mb   = transitions["ret"][mb_idx]

                    # -- Actor update --
                    with tf.GradientTape() as tape:
                        probs = self.actor_model(obs_mb, training = True)
                        logp = tf.math.log( # encoding the chosen action in the rollout
                            tf.reduce_sum(probs * tf.one_hot(act_mb, self.act_dim), axis=1) + 1e-8
                            ) # collapsing across the action dim, leaving a vector [batch_size] with the prob. of the taken action under the current policy
                        ratio = tf.exp(logp - logp_old)
                        clip_adv = tf.clip_by_value(ratio, 1 - self.clip_range, 1 + self.clip_range) * adv_mb
                        actor_loss = -tf.reduce_mean(tf.minimum(ratio * adv_mb, clip_adv))
                    
                    grads = tape.gradient(actor_loss, self.actor_model.trainable_variables)
                    grads = [tf.clip_by_norm(g, 0.5) if g is not None else None for g in grads]
                    self.actor_optimizer.apply_gradients(zip(grads, self.actor_model.trainable_variables))

                # -- Critic update --
                joint_arr = np.array(t_buffer.data["j_obs"], dtype=np.float32)
                ret_arr = np.array(t_buffer.data["ret"],   dtype=np.float32)

                # Reduce duplicates: keep one entry every num_agents
                step_idx = np.arange(0, len(joint_arr), self.num_agents)
                joint_arr_steps = joint_arr[step_idx]      # [T, joint_obs_dim]
                ret_steps = ret_arr[step_idx]        # [T]

                # Shuffle for minibatching
                idx = np.arange(len(joint_arr_steps))
                np.random.shuffle(idx)

                for start in range(0, len(idx), mb_size):
                    mb = idx[start:start+mb_size]
                    joint_mb = joint_arr_steps[mb]
                    ret_mb = ret_steps[mb]

                    with tf.GradientTape() as tape:
                        v = self.critic_model(joint_mb, training=True)  # [B,1]
                        v = tf.squeeze(v, axis=1)
                        critic_loss = tf.reduce_mean(tf.square(ret_mb - v))

                    grads = tape.gradient(critic_loss, self.critic_model.trainable_variables)
                    grads = [tf.clip_by_norm(g, 0.5) if g is not None else None for g in grads]
                    self.critic_optimizer.apply_gradients(zip(grads, self.critic_model.trainable_variables))

            # --- Store losses ---
            with self.writer.as_default():
                tf.summary.scalar("Loss/Actor", actor_loss, step=r)
                tf.summary.scalar("Loss/Critic", critic_loss, step=r)
                self.writer.flush()

            # --- Compute avg return from true rewards ---
            # Current buffer has rewards duplicated across agents -> dropping duplicates
            rewards = np.array(transitions["rew"], dtype=np.float32)
            episode_rewards = rewards[::self.num_agents] # keep one entry every num_agents

            # Now split episode_rewards back into episodes
            ep_returns = [
                np.sum(episode_rewards[i * self.horizon:(i + 1) * self.horizon])
                for i in range(ep_per_rollout)
            ]
            avg_return = float(np.mean(ep_returns))
            self.avg_return_per_rollout[current_layout].append(avg_return)
            print(f" ➗ Average return: {avg_return:.2f}")
            with self.writer.as_default():
                tf.summary.scalar("Return/Avg", avg_return, step=r)
                self.writer.flush()
    
# Multi-head critic version of MAPPO
class MAPPO_mh(MAPPO):
    def define_models(self, env_params):
        nn = NN(**env_params)
        self.actor_model = nn.build_actor()
        self.critic_model = nn.build_mh_critic()
        self.actor_optimizer = AdamW(learning_rate=5e-4, weight_decay=1e-4)
        self.critic_optimizer = AdamW(learning_rate=2e-4, weight_decay=1e-4)

    def logging(self):
        super().logging()    
        self.deliveries_per_layout = {name: [] for name in self.env.layout_dict.values()}
        self.rollout_deliveries_per_layout = {name: [] for name in self.env.layout_dict.values()}
    
    def hyperparams_decaying(self, rollout):
        frac = rollout / self.total_rollouts
        self.entropy_coeff = (1 - frac) * 0.05   # 0.05 -> 0
        self.clip_range = 0.2 - frac * (0.15)    # 0.2 -> 0.05
        new_lr_actor  = 5e-4 * (1 - frac) + 5e-5 * frac
        new_lr_critic = 2e-4 * (1 - frac) + 2e-5 * frac
        self.actor_optimizer.learning_rate.assign(new_lr_actor)
        self.critic_optimizer.learning_rate.assign(new_lr_critic)

    def dynamic_layout_switching(self):
        layouts = self.env.layout_dict

        # assuming always the same layout order
        cramped_room = layouts[0]
        asymmetric_advantages = layouts[1]
        coordination_ring = layouts[4]

        # compute mean of last N rollouts for a given layout
        mean = lambda k, N: np.mean(self.avg_return_per_rollout[k][-N:])\
              if len(self.avg_return_per_rollout[k]) >= N else np.mean(self.avg_return_per_rollout[k])\
                  if self.avg_return_per_rollout[k] else 0.0
        
        # -- Avg return computation --
        # if the avg return and the avg deliveries of the last N episodes...
        if self.phase == 1 and mean(cramped_room, N = 5) >= 60 and\
            mean_list(self.rollout_deliveries_per_layout["cramped_room"], N=4) >= 4:
            print("\n🎓 Phase 2 – Interleaved layouts")
            self.phase = 2
        if self.phase == 2 and mean(cramped_room, N = 5) >= 70 and\
              mean(asymmetric_advantages, N = 5) >= 50 and mean(coordination_ring, N = 5) >= 20 and\
              mean_list(self.rollout_deliveries_per_layout["cramped_room"], N=3) >= 3.5 and\
                  mean_list(self.rollout_deliveries_per_layout["asymmetric_advantages"], N=3) >= 1 and\
                      mean_list(self.rollout_deliveries_per_layout["coordination_ring"], N=3) >= 0.2:
            print("\n🌀 Phase 3 – Random layouts")
            self.phase = 3

        # -- Phase check --
        if self.phase == 1:
            idx = 0
        elif self.phase == 2:
            n_layouts = len(layouts)
            idx = np.random.choice(n_layouts, p = [0.1, 0.2, 0.2, 0.3, 0.2])
            # layouts: cramped_room, asymmetric_advantages, bottleneck, counter_circuit, coordination_ring
        else:
            idx = np.random.choice(n_layouts)

        # -- Layout switching --
        self.env.switch_to_env(idx)
        self.cur_layout = self.env.layout_dict[idx]
        return idx, self.cur_layout

    
    # COLLECTING TRAJECTORIES
    def rollout(self, episodes, rew_shaping, env_idx = None):
        buffer = Buffer()           
        deliveries = [] # count how many deliveries within one ep.

        # Loop over episodes
        for _ in range(episodes):
            if env_idx is not None:
                obs = self.env.reset(idx=env_idx)
            else:
                obs = self.env.reset()
            obs = obs["both_agent_obs"]  
            ep_deliveries = 0  # <-- re-initialized every episode

            # Loop over steps (400)
            for _ in range(self.horizon):
                # --- State-value estimation ---
                joint_obs = np.concatenate(obs, axis=-1).reshape(1, -1)
                V_s = self.critic_model(joint_obs).numpy().squeeze() # store a scalar for each agent

                # --- Sample actions per agent
                # and stepping into the env. ---
                actions = []
                logps = []

                for a in range(self.num_agents):
                    obs_a = np.expand_dims(obs[a], axis=0)   # (1, 96)
                    probs = self.actor_model(obs_a, training=False).numpy()[0]
                    action, logp = self.sample_action(probs=probs)
                    actions.append(int(action))
                    logps.append(float(logp))
                
                # Step env.
                next_obs, sparse_reward, done, info = self.env.step(actions)

                # --- Reward handling ---
                total_shaped = sum(info['shaped_r_by_agent'])    
                team_reward = sparse_reward + total_shaped # scalar, obv
                if sparse_reward == 20:
                    ep_deliveries += 1

                # Apply decaying
                team_reward = self.reward_shaping(n_rollout=self.num_rollout,
                                                  rew=team_reward,
                                                  shaping_type=rew_shaping)

                # Assign team reward to each agent
                rewards = [team_reward] * self.num_agents

                # --- Store transitions per agent ---
                for a in range(self.num_agents):
                    buffer.add(
                        obs = obs[a],
                        j_obs = joint_obs.squeeze(0),
                        val = V_s[a],
                        act = actions[a],
                        logp = logps[a],
                        rew = rewards[a],
                        done = done
                    )
                # --- Update obs ---
                obs = next_obs
                obs = obs["both_agent_obs"]

            # Track per-episode deliveries
            deliveries.append(ep_deliveries)
        
        # Print total deliveries in the rollout, per-episode
        print(f"  🍛 Delivered dishes per-ep: {deliveries}")
        self.deliveries_per_layout[self.cur_layout].extend(deliveries)

        joint_obs = np.concatenate(obs, axis=-1).reshape(1, -1)
        last_value = self.critic_model(joint_obs, training=False).numpy().squeeze()  # shape: (num_agents,)

        # GAE computation
        self.compute_gae(buffer, last_value)
        return buffer
    
    # MAIN LEARNING FUNCTION
    def train_loop(self, epochs = 6, ep_per_rollout = 8, num_rollouts = 10, mb_size = 64, rew_shaping = "None"):
        print("▶ Starting learning process!\nPHASE 1️⃣: cramped room")
        self.num_rollout = 0
        self.total_rollouts = num_rollouts

        # Loop over total number of rollouts
        for r in range(num_rollouts):
            # --- Schedule coeffs as training progresses ---
            self.hyperparams_decaying(r)

            # --- Layout manager ---
            layout_idx, current_layout = self.dynamic_layout_switching()
            print(f"\n🎲 Layout sampled: {current_layout}") if not self.phase == 1 else None

            # --- Gather transitions from one rollout ---
            print(f"\nRollout: {r+1}\n 📦 Gathering transitions...")
            t_buffer = self.rollout(episodes = ep_per_rollout, rew_shaping=rew_shaping, env_idx = layout_idx)
            transitions = {k: np.array(v) for k, v in t_buffer.data.items()}
            self.num_rollout = r

            # --- Batching and PPO updates ---
            N = len(transitions["obs"])

            print(" 📖 Learning...")

            # Looping over #epochs
            for _ in range(epochs):
                indices = np.arange(N)
                np.random.shuffle(indices)

                # Looping over minibatches
                for start in range(0, N, mb_size):
                    end = start + mb_size
                    mb_idx = indices[start:end]

                    obs_mb   = transitions["obs"][mb_idx]
                    act_mb   = transitions["act"][mb_idx]
                    logp_old = transitions["logp"][mb_idx]
                    adv_mb   = transitions["adv"][mb_idx]
                    ret_mb   = transitions["ret"][mb_idx]

                    # --- Normalize advantages ---
                    norm_adv_mb = (adv_mb - tf.reduce_mean(adv_mb)) / (tf.math.reduce_std(adv_mb) + 1e-8)
                    norm_adv_mb = tf.cast(norm_adv_mb, dtype=tf.float32)

                    # -- Actor update --
                    with tf.GradientTape() as tape:
                        probs = self.actor_model(obs_mb, training = True)
                        logp = tf.math.log( # encoding the chosen action in the rollout
                            tf.reduce_sum(probs * tf.one_hot(act_mb, self.act_dim), axis=1) + 1e-8
                            ) # collapsing across the action dim, leaving a vector [batch_size] with the prob. of the taken action under the current policy
                        ratio = tf.exp(logp - logp_old)
                        clip_adv = tf.clip_by_value(ratio, 1 - self.clip_range, 1 + self.clip_range) * norm_adv_mb
                        entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=1) # entropy bonus
                        actor_loss = -tf.reduce_mean(tf.minimum(ratio * norm_adv_mb, clip_adv)) - self.entropy_coeff * tf.reduce_mean(entropy)
                    
                    grads = tape.gradient(actor_loss, self.actor_model.trainable_variables)
                    grads = [tf.clip_by_norm(g, 0.5) if g is not None else None for g in grads]
                    self.actor_optimizer.apply_gradients(zip(grads, self.actor_model.trainable_variables))

                # -- Critic update --
                joint_arr = np.array(t_buffer.data["j_obs"], dtype=np.float32)
                ret_arr = np.array(t_buffer.data["ret"], dtype=np.float32)

                # Reduce duplicates: keep one entry every num_agents
                step_idx = np.arange(0, len(joint_arr), self.num_agents)
                joint_arr_steps = joint_arr[step_idx]      # [T, joint_obs_dim]
                ret_steps = ret_arr[step_idx]              # [T]

                # Shuffle for minibatching
                idx = np.arange(len(joint_arr_steps))
                np.random.shuffle(idx)

                for start in range(0, len(idx), mb_size):
                    mb = idx[start:start+mb_size]
                    joint_mb = joint_arr_steps[mb]
                    ret_mb = ret_steps[mb]

                    # Critic expects shape [B, n_agents] for ret_mb -> expand it
                    ret_mb = np.tile(ret_mb[:, None], (1, self.num_agents))

                    with tf.GradientTape() as tape:
                        v = self.critic_model(joint_mb, training=True)  # [B, n_agents]
                        critic_loss = tf.reduce_mean(tf.square(ret_mb - v))

                    grads = tape.gradient(critic_loss, self.critic_model.trainable_variables)
                    grads = [tf.clip_by_norm(g, 0.5) if g is not None else None for g in grads]
                    self.critic_optimizer.apply_gradients(zip(grads, self.critic_model.trainable_variables))

            # --- Store losses ---
            with self.writer.as_default():
                tf.summary.scalar("Loss/Actor", actor_loss, step=r)
                tf.summary.scalar("Loss/Critic", critic_loss, step=r)
                self.writer.flush()

            # --- Compute avg return from true rewards ---
            # Current buffer has rewards duplicated across agents -> dropping duplicates
            rewards = np.array(transitions["rew"], dtype=np.float32)
            episode_rewards = rewards[::self.num_agents] # keep one entry every num_agents

            # Now split episode_rewards back into episodes
            ep_returns = [
                np.sum(episode_rewards[i * self.horizon:(i + 1) * self.horizon])
                for i in range(ep_per_rollout)
            ]
            avg_return = float(np.mean(ep_returns))
            self.avg_return_per_rollout[current_layout].append(avg_return)
            print(f" ➗ Average return: {avg_return:.2f}")
            with self.writer.as_default():
                tf.summary.scalar("Return/Avg", avg_return, step=r)
                self.writer.flush()

            # --- Track deliveries ---
            avg_deliveries = np.mean(self.deliveries_per_layout[current_layout][-ep_per_rollout:])
            with self.writer.as_default():
                tf.summary.scalar("Deliveries/Avg", avg_deliveries, step=r)
                self.writer.flush()
            
            # --- Store deliveries at rollout level ---
            self.rollout_deliveries_per_layout[current_layout].append(avg_deliveries)


# Partner-pairing version of MAPPO_mh
class MAPPO_pp(MAPPO_mh):
    def __init__(self, env, env_params,):
        self.partner_prob_phase = {1: 0.6, 2: 0.45, 3: 0.4}
        self.partner_prob_layout = self.define_layout_partner_probs()
        self.name = "self_agent"
        super().__init__(env, env_params)
    
    def define_layout_partner_probs(self):
        prob_dict = {
            "phase_1": {
                "actor_model_cramped_2": 0.2,   # ""
                "actor_model_mh_cramped": 0.2,  # ""
                "random_agent": 0.3,           # //
                "greedy_human_agent": 0.3,     # [sampled layout] only
                "actor_model_ml": 0.07,         # [cramped, asymm, bottleneck]
                "actor_model_mh": 0.07 ,        # ""
                "actor_model_mh_full": 0.05,    # [cramped, asymm, bottleneck, counter_circuit, coord_ring]
                "actor_model_mh_full_2": 0.025, # ""
                "actor_model_mh_ft": 0.025      # ""
            },
            "phase_2": {
                "actor_model_cramped_2": 0.025,
                "actor_model_mh_cramped": 0.05,
                "random_agent": 0.025,
                "greedy_human_agent": 0.05, 
                "actor_model_ml": 0.1,
                "actor_model_mh": 0.1,
                "actor_model_mh_full": 0.2,
                "actor_model_mh_full_2": 0.2,
                "actor_model_mh_ft": 0.2
            },
            "phase_3": {
                "actor_model_cramped_2": 0.025,
                "actor_model_mh_cramped": 0.05,
                "random_agent": 0,
                "greedy_human_agent": 0.05, 
                "actor_model_ml": 0.1,
                "actor_model_mh": 0.1,
                "actor_model_mh_full": 0.2,
                "actor_model_mh_full_2": 0.225,
                "actor_model_mh_ft": 0.2
            }
        }
        return prob_dict

    def sample_partner(self, pool, gha, downweight=0.25):
        """
        Sample a partner agent depending on phase and current layout.
        
        Args:
            pool: dict of {name -> agent} for learned/random agents
            gha: dict of {layout_name -> GreedyHumanModel} for each layout
            downweight: factor to apply to specialists when sampled outside their layout
        """
        layout = self.cur_layout
        phase = self.phase
        prob_dict = self.partner_prob_layout[f"phase_{phase}"]

        partner_names = []
        partner_probs = []

        for name, p in prob_dict.items():
            # cramped specialists -> downweight if not in cramped_room
            if "cramped" in name and layout != "cramped_room":
                partner_names.append(name)
                partner_probs.append(p * downweight)
                continue

            # greedy human agent
            if name == "greedy_human_agent":
                partner_names.append(name)
                partner_probs.append(p)
                continue

            # generic case
            partner_names.append(name)
            partner_probs.append(p)

        # renormalize probs
        partner_probs = np.array(partner_probs, dtype=np.float32)
        partner_probs /= partner_probs.sum()

        # sample
        sampled_name = np.random.choice(partner_names, p=partner_probs)

        # special case: gh agent
        if sampled_name == "greedy_human_agent":
            partner = gha.select_gh_agent(layout)   # pick correct GHA for this layout
            partner = WrappedPartner(partner, action_space=6)
            partner.name = "greedy_human_agent"

        else: # learned or random agents
            partner = pool[sampled_name]

        return partner


    # COLLECTING TRAJECTORIES
    def rollout(self, episodes, rew_shaping, partner_pool, gha, env_idx = None):
        buffer = Buffer()           
        deliveries = [] # count how many deliveries within one ep.
        self_bool = False

        # Sample a partner from the pool
        if np.random.rand() < self.partner_prob_phase[self.phase]: # prob. depends on the phase
            partner = self.sample_partner(pool=partner_pool, gha=gha)
            print(f" 🤝 Partner sampled: {partner.name}")
        else:
            partner = self  # chance of pairing with itself: self-play
            print(" 🤝 Partner sampled: self")
            self_bool = True

        # Loop over episodes
        for _ in range(episodes):
            if env_idx is not None:
                state = self.env.reset(idx=env_idx)
            else:
                state = self.env.reset()
            obs = state["both_agent_obs"]  
            ep_deliveries = 0  # <-- re-initialized every episode

            # Loop over steps (400)
            for _ in range(self.horizon):
                # --- State-value estimation ---
                joint_obs = np.concatenate(obs, axis=-1).reshape(1, -1)
                V_s = self.critic_model(joint_obs).numpy().squeeze() # store a scalar for each agent

                # --- Sample actions per agent
                # and stepping into the env. ---
                actions = []
                logps = []

                # -- Agent 0: the learning agent --
                obs_a = np.expand_dims(obs[0], axis=0)   # (1, 96)
                probs = self.actor_model(obs_a, training=False).numpy()[0]
                action, logp = self.sample_action(probs=probs)
                actions.append(int(action))
                logps.append(float(logp))
                
                # -- Agent 1: a partner sampled from the pool --
                obs_p = np.expand_dims(obs[1], axis=0)   # (1, 96)

                # - Scripted agents - 
                if partner.name == "greedy_human_agent":
                    env_state = self.env.cur_env.base_env.state  # scripted partners want the full state
                    partner.name = "greedy_human_agent"
                    partner.set_agent_index(state["other_agent_env_idx"])
                    probs_p = partner.actor_model(env_state)[0]

                elif partner.name == "random_agent":
                    env_state = self.env.cur_env.base_env.state 
                    probs_p = partner.actor_model(env_state)[0]

                # - Learned agents -
                else: 
                    if self_bool == True: # self-play -> use own actor ...
                        probs_p = partner.actor_model(obs_p, training=False).numpy()[0]
                    else: # ...or use Sequential stored models
                        probs_p = partner(obs_p, training=False).numpy()[0]

                action_p, logp = self.sample_action(probs=probs_p)
                actions.append(int(action_p))
                logps.append(0.0)  # we don't optimize partner's logp (dummy value)

                # Step env.
                next_obs, sparse_reward, done, info = self.env.step(actions)

                # --- Reward handling ---
                total_shaped = sum(info['shaped_r_by_agent'])    
                team_reward = sparse_reward + total_shaped # scalar, obv
                if sparse_reward == 20:
                    ep_deliveries += 1

                # Apply decaying
                team_reward = self.reward_shaping(n_rollout=self.num_rollout,
                                                  rew=team_reward,
                                                  shaping_type=rew_shaping)

                # Assign team reward to each agent
                rewards = [team_reward] * self.num_agents

                # --- Store transitions per agent ---
                for a in range(self.num_agents):
                    buffer.add(
                        obs = obs[a],
                        j_obs = joint_obs.squeeze(0),
                        val = V_s[a],
                        act = actions[a],
                        logp = logps[a],
                        rew = rewards[a],
                        done = done
                    )
                # --- Update obs ---
                obs = next_obs
                obs = obs["both_agent_obs"]

            # Track per-episode deliveries
            deliveries.append(ep_deliveries)
        
        # Print total deliveries in the rollout, per-episode
        print(f"  🍛 Delivered dishes per-ep: {deliveries}")
        self.deliveries_per_layout[self.cur_layout].extend(deliveries)

        joint_obs = np.concatenate(obs, axis=-1).reshape(1, -1)
        last_value = self.critic_model(joint_obs, training=False).numpy().squeeze()  # shape: (num_agents,)

        # GAE computation
        self.compute_gae(buffer, last_value)
        return buffer

    # MAIN LEARNING FUNCTION
    def train_loop(self, gha, partner_pool, epochs = 6, ep_per_rollout = 8, num_rollouts = 10, mb_size = 64, rew_shaping = "None"):
        print("▶ Starting learning process!\nPHASE 1️⃣: cramped room")
        self.num_rollout = 0
        self.total_rollouts = num_rollouts

        # Loop over total number of rollouts
        for r in range(num_rollouts):
            # --- Schedule coeffs as training progresses ---
            self.hyperparams_decaying(r)

            # --- Layout manager ---
            layout_idx, current_layout = self.dynamic_layout_switching()
            print(f"\n🎲 Layout sampled: {current_layout}") if not self.phase == 1 else None

            # --- Gather transitions from one rollout ---
            print(f"\nRollout: {r+1}\n 📦 Gathering transitions...")
            t_buffer = self.rollout(gha = gha, episodes = ep_per_rollout, rew_shaping=rew_shaping, env_idx = layout_idx, partner_pool= partner_pool)
            transitions = {k: np.array(v) for k, v in t_buffer.data.items()}
            self.num_rollout = r

            # --- Batching and PPO updates ---
            N = len(transitions["obs"])

            print(" 📖 Learning...")

            # Looping over #epochs
            for _ in range(epochs):
                indices = np.arange(N)
                np.random.shuffle(indices)

                # Looping over minibatches
                for start in range(0, N, mb_size):
                    end = start + mb_size
                    mb_idx = indices[start:end]

                    obs_mb   = transitions["obs"][mb_idx]
                    act_mb   = transitions["act"][mb_idx]
                    logp_old = transitions["logp"][mb_idx]
                    adv_mb   = transitions["adv"][mb_idx]
                    ret_mb   = transitions["ret"][mb_idx]

                    # --- Normalize advantages ---
                    norm_adv_mb = (adv_mb - tf.reduce_mean(adv_mb)) / (tf.math.reduce_std(adv_mb) + 1e-8)
                    norm_adv_mb = tf.cast(norm_adv_mb, dtype=tf.float32)

                    # -- Actor update --
                    with tf.GradientTape() as tape:
                        probs = self.actor_model(obs_mb, training = True)
                        logp = tf.math.log( # encoding the chosen action in the rollout
                            tf.reduce_sum(probs * tf.one_hot(act_mb, self.act_dim), axis=1) + 1e-8
                            ) # collapsing across the action dim, leaving a vector [batch_size] with the prob. of the taken action under the current policy
                        ratio = tf.exp(logp - logp_old)
                        clip_adv = tf.clip_by_value(ratio, 1 - self.clip_range, 1 + self.clip_range) * norm_adv_mb
                        entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=1) # entropy bonus
                        actor_loss = -tf.reduce_mean(tf.minimum(ratio * norm_adv_mb, clip_adv)) - self.entropy_coeff * tf.reduce_mean(entropy)
                    
                    grads = tape.gradient(actor_loss, self.actor_model.trainable_variables)
                    grads = [tf.clip_by_norm(g, 0.5) if g is not None else None for g in grads]
                    self.actor_optimizer.apply_gradients(zip(grads, self.actor_model.trainable_variables))

                # -- Critic update --
                joint_arr = np.array(t_buffer.data["j_obs"], dtype=np.float32)
                ret_arr = np.array(t_buffer.data["ret"], dtype=np.float32)

                # Reduce duplicates: keep one entry every num_agents
                step_idx = np.arange(0, len(joint_arr), self.num_agents)
                joint_arr_steps = joint_arr[step_idx]      # [T, joint_obs_dim]
                ret_steps = ret_arr[step_idx]              # [T]

                # Shuffle for minibatching
                idx = np.arange(len(joint_arr_steps))
                np.random.shuffle(idx)

                for start in range(0, len(idx), mb_size):
                    mb = idx[start:start+mb_size]
                    joint_mb = joint_arr_steps[mb]
                    ret_mb = ret_steps[mb]

                    # Critic expects shape [B, n_agents] for ret_mb -> expand it
                    ret_mb = np.tile(ret_mb[:, None], (1, self.num_agents))

                    with tf.GradientTape() as tape:
                        v = self.critic_model(joint_mb, training=True)  # [B, n_agents]
                        critic_loss = tf.reduce_mean(tf.square(ret_mb - v))

                    grads = tape.gradient(critic_loss, self.critic_model.trainable_variables)
                    grads = [tf.clip_by_norm(g, 0.5) if g is not None else None for g in grads]
                    self.critic_optimizer.apply_gradients(zip(grads, self.critic_model.trainable_variables))

            # --- Store losses ---
            with self.writer.as_default():
                tf.summary.scalar("Loss/Actor", actor_loss, step=r)
                tf.summary.scalar("Loss/Critic", critic_loss, step=r)
                self.writer.flush()

            # --- Compute avg return from true rewards ---
            # Current buffer has rewards duplicated across agents -> dropping duplicates
            rewards = np.array(transitions["rew"], dtype=np.float32)
            episode_rewards = rewards[::self.num_agents] # keep one entry every num_agents

            # Now split episode_rewards back into episodes
            ep_returns = [
                np.sum(episode_rewards[i * self.horizon:(i + 1) * self.horizon])
                for i in range(ep_per_rollout)
            ]
            avg_return = float(np.mean(ep_returns))
            self.avg_return_per_rollout[current_layout].append(avg_return)
            print(f" ➗ Average return: {avg_return:.2f}")
            with self.writer.as_default():
                tf.summary.scalar("Return/Avg", avg_return, step=r)
                self.writer.flush()

            # --- Track deliveries ---
            avg_deliveries = np.mean(self.deliveries_per_layout[current_layout][-ep_per_rollout:])
            with self.writer.as_default():
                tf.summary.scalar("Deliveries/Avg", avg_deliveries, step=r)
                self.writer.flush()
            
            # --- Store deliveries at rollout level ---
            self.rollout_deliveries_per_layout[current_layout].append(avg_deliveries)


# Role-embedding version of MAPPO_pp
class MAPPO_re(MAPPO_mh):
    def __init__(self, env, env_params, partner_pool, role_dim):
        self.role_dim = role_dim
        super().__init__(env, env_params, partner_pool)

    def define_models(self, env_params):
        nn = NN(**env_params)
        self.actor_model = nn.build_augmented_actor(role_dim=self.role_dim)
        self.critic_model = nn.build_mh_critic()
        self.actor_optimizer = AdamW(learning_rate=5e-4, weight_decay=1e-4)
        self.critic_optimizer = AdamW(learning_rate=2e-4, weight_decay=1e-4)

    def extract_role(self, context, role_dim):
        # TODO
        context_vector = np.zeros((1, role_dim), dtype=np.float32)
        return context_vector

    # COLLECTING TRAJECTORIES
    def rollout(self, episodes, rew_shaping, partner_pool, env_idx = None):
        buffer = Buffer()           
        deliveries = [] # count how many deliveries within one ep.

        # Sample a partner from the pool
        if np.random.rand() < self.partner_prob[self.phase]: # prob. depends on the phase
            partner = self.sample_partner(partner_pool)
            print(f" 🤝 Partner type sampled from the pool: {partner.name}")
        else:
            partner = self  # chance of pairing with itself: self-play
            print(" 🤝 Partner type sampled: self")

        # Loop over episodes
        for _ in range(episodes):
            if env_idx is not None:
                state = self.env.reset(idx=env_idx)
            else:
                state = self.env.reset()

            # Get obs and contextual info
            obs = state["both_agent_obs"]  
            context = state["overcooked_state"]

            aug_obs = []
            ep_deliveries = 0  # <-- re-initialized every episode

            # Loop over steps (400)
            for _ in range(self.horizon):
                # --- State-value estimation ---
                joint_obs = np.concatenate(obs, axis=-1).reshape(1, -1)
                V_s = self.critic_model(joint_obs).numpy().squeeze() # store a scalar for each agent

                # --- Sample actions per agent
                # and stepping into the env. ---
                actions = []
                logps = []

                for a in range(self.num_agents):
                    obs_a = np.expand_dims(obs[a], axis=0)   # (1, 96)

                    # Extract contextual role
                    role_a = self.extract_role(context=context, n_agent=a)
                    aug_obs_a = np.concatenate([obs_a, role_a], axis=-1)  # (1, 96 + role_dim)
                    aug_obs.append(aug_obs_a)

                    # Sample action
                    probs = self.actor_model(aug_obs_a, training=False).numpy()[0]
                    action, logp = self.sample_action(probs=probs)
                    actions.append(int(action))
                    logps.append(float(logp))
                
                # Step env.
                next_obs, sparse_reward, done, info = self.env.step(actions)

                # --- Reward handling ---
                total_shaped = sum(info['shaped_r_by_agent'])    
                team_reward = sparse_reward + total_shaped # scalar, obv
                if sparse_reward == 20:
                    ep_deliveries += 1

                # Apply decaying
                team_reward = self.reward_shaping(n_rollout=self.num_rollout,
                                                  rew=team_reward,
                                                  shaping_type=rew_shaping)

                # Assign team reward to each agent
                rewards = [team_reward] * self.num_agents

                # --- Store transitions per agent ---
                for a in range(self.num_agents):
                    buffer.add(
                        obs = aug_obs[a],
                        j_obs = joint_obs.squeeze(0),
                        val = V_s[a],
                        act = actions[a],
                        logp = logps[a],
                        rew = rewards[a],
                        done = done
                    )
                # --- Update obs ---
                obs = next_obs
                obs = obs["both_agent_obs"]

            # Track per-episode deliveries
            deliveries.append(ep_deliveries)
        
        # Print total deliveries in the rollout, per-episode
        print(f"  🍛 Delivered dishes per-ep: {deliveries}")
        self.deliveries_per_layout[self.cur_layout].extend(deliveries)

        joint_obs = np.concatenate(obs, axis=-1).reshape(1, -1)
        last_value = self.critic_model(joint_obs, training=False).numpy().squeeze()  # shape: (num_agents,)

        # GAE computation
        self.compute_gae(buffer, last_value)
        return buffer
    

    # MAIN LEARNING FUNCTION
    def train_loop(self, gha, partner_pool, epochs = 6, ep_per_rollout = 8, num_rollouts = 10, mb_size = 64, rew_shaping = "None"):
        print("▶ Starting learning process!\nPHASE 1️⃣: cramped room")
        self.num_rollout = 0
        self.total_rollouts = num_rollouts

        # Loop over total number of rollouts
        for r in range(num_rollouts):
            # --- Schedule coeffs as training progresses ---
            self.hyperparams_decaying(r)

            # --- Layout manager ---
            layout_idx, current_layout = self.dynamic_layout_switching()
            print(f"\n🎲 Layout sampled: {current_layout}") if not self.phase == 1 else None

            # --- Import GreedyHumanAgent ---
            partner_pool = import_gh_agent(pool = partner_pool, gha=gha, layout = current_layout)

            # --- Gather transitions from one rollout ---
            print(f"\nRollout: {r+1}\n 📦 Gathering transitions...")
            t_buffer = self.rollout(episodes = ep_per_rollout, rew_shaping=rew_shaping, env_idx = layout_idx, partner_pool= partner_pool)
            transitions = {k: np.array(v) for k, v in t_buffer.data.items()}
            self.num_rollout = r

            # --- Batching and PPO updates ---
            N = len(transitions["obs"])

            print(" 📖 Learning...")

            # Looping over #epochs
            for _ in range(epochs):
                indices = np.arange(N)
                np.random.shuffle(indices)

                # Looping over minibatches
                for start in range(0, N, mb_size):
                    end = start + mb_size
                    mb_idx = indices[start:end]

                    obs_mb   = transitions["obs"][mb_idx]
                    act_mb   = transitions["act"][mb_idx]
                    logp_old = transitions["logp"][mb_idx]
                    adv_mb   = transitions["adv"][mb_idx]
                    ret_mb   = transitions["ret"][mb_idx]

                    # --- Normalize advantages ---
                    norm_adv_mb = (adv_mb - tf.reduce_mean(adv_mb)) / (tf.math.reduce_std(adv_mb) + 1e-8)
                    norm_adv_mb = tf.cast(norm_adv_mb, dtype=tf.float32)

                    # -- Actor update --
                    with tf.GradientTape() as tape:
                        probs = self.actor_model(obs_mb, training = True)
                        logp = tf.math.log( # encoding the chosen action in the rollout
                            tf.reduce_sum(probs * tf.one_hot(act_mb, self.act_dim), axis=1) + 1e-8
                            ) # collapsing across the action dim, leaving a vector [batch_size] with the prob. of the taken action under the current policy
                        ratio = tf.exp(logp - logp_old)
                        clip_adv = tf.clip_by_value(ratio, 1 - self.clip_range, 1 + self.clip_range) * norm_adv_mb
                        entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=1) # entropy bonus
                        actor_loss = -tf.reduce_mean(tf.minimum(ratio * norm_adv_mb, clip_adv)) - self.entropy_coeff * tf.reduce_mean(entropy)
                    
                    grads = tape.gradient(actor_loss, self.actor_model.trainable_variables)
                    grads = [tf.clip_by_norm(g, 0.5) if g is not None else None for g in grads]
                    self.actor_optimizer.apply_gradients(zip(grads, self.actor_model.trainable_variables))

                # -- Critic update --
                joint_arr = np.array(t_buffer.data["j_obs"], dtype=np.float32)
                ret_arr = np.array(t_buffer.data["ret"], dtype=np.float32)

                # Reduce duplicates: keep one entry every num_agents
                step_idx = np.arange(0, len(joint_arr), self.num_agents)
                joint_arr_steps = joint_arr[step_idx]      # [T, joint_obs_dim]
                ret_steps = ret_arr[step_idx]              # [T]

                # Shuffle for minibatching
                idx = np.arange(len(joint_arr_steps))
                np.random.shuffle(idx)

                for start in range(0, len(idx), mb_size):
                    mb = idx[start:start+mb_size]
                    joint_mb = joint_arr_steps[mb]
                    ret_mb = ret_steps[mb]

                    # Critic expects shape [B, n_agents] for ret_mb -> expand it
                    ret_mb = np.tile(ret_mb[:, None], (1, self.num_agents))

                    with tf.GradientTape() as tape:
                        v = self.critic_model(joint_mb, training=True)  # [B, n_agents]
                        critic_loss = tf.reduce_mean(tf.square(ret_mb - v))

                    grads = tape.gradient(critic_loss, self.critic_model.trainable_variables)
                    grads = [tf.clip_by_norm(g, 0.5) if g is not None else None for g in grads]
                    self.critic_optimizer.apply_gradients(zip(grads, self.critic_model.trainable_variables))

            # --- Store losses ---
            with self.writer.as_default():
                tf.summary.scalar("Loss/Actor", actor_loss, step=r)
                tf.summary.scalar("Loss/Critic", critic_loss, step=r)
                self.writer.flush()

            # --- Compute avg return from true rewards ---
            # Current buffer has rewards duplicated across agents -> dropping duplicates
            rewards = np.array(transitions["rew"], dtype=np.float32)
            episode_rewards = rewards[::self.num_agents] # keep one entry every num_agents

            # Now split episode_rewards back into episodes
            ep_returns = [
                np.sum(episode_rewards[i * self.horizon:(i + 1) * self.horizon])
                for i in range(ep_per_rollout)
            ]
            avg_return = float(np.mean(ep_returns))
            self.avg_return_per_rollout[current_layout].append(avg_return)
            print(f" ➗ Average return: {avg_return:.2f}")
            with self.writer.as_default():
                tf.summary.scalar("Return/Avg", avg_return, step=r)
                self.writer.flush()

            # --- Track deliveries ---
            avg_deliveries = np.mean(self.deliveries_per_layout[current_layout][-ep_per_rollout:])
            with self.writer.as_default():
                tf.summary.scalar("Deliveries/Avg", avg_deliveries, step=r)
                self.writer.flush()
            
            # --- Store deliveries at rollout level ---
            self.rollout_deliveries_per_rollout[current_layout].append(avg_deliveries)
   