import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value, ConditionedScoreNet

from functools import partial
from utils.networks import MLP, TanhNormal, LogParam


def clip_ste(x, min_val, max_val):
    """
    Forward pass: jnp.clip(x)
    Backward pass: acts as an identity function (gradient = 1 everywhere)
    """
    clipped = jnp.clip(x, min_val, max_val)
    return x + jax.lax.stop_gradient(clipped - x)

class MEAMAgent(flax.struct.PyTreeNode):
    """Q-learning with adjoint matching, mixture of priors, and entropy maximization."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    
    def critic_loss(self, batch, grad_params, rng):
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]
        
        next_actions = self.sample_actions(batch['next_observations'][..., -1, :], rng=rng)
        next_actions = jnp.clip(next_actions, -1, 1)
        
        next_qs = self.network.select('target_critic')(batch['next_observations'][..., -1, :], next_actions)
        
        # --- DIAGNOSTICS ---
        next_q_mean = next_qs.mean(axis=0)
        next_q_std = next_qs.std(axis=0)
        # -------------------
        
        next_q = next_q_mean - self.config["rho"] * next_q_std
        
        target_q = batch['rewards'][..., -1] + \
            (self.config['discount'] ** self.config["horizon_length"]) * batch['masks'][..., -1] * next_q

        q = self.network.select('critic')(batch['observations'], batch_actions, params=grad_params)
        critic_loss = (jnp.square(q - target_q) * batch['valid'][..., -1]).mean()

        total_loss = critic_loss
        return total_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
            # --- NEW DIAGNOSTICS ---
            'diag_next_q_mean': next_q_mean.mean(),
            'diag_next_q_std': next_q_std.mean(),
        }
    
    @partial(jax.jit, static_argnames=("flow_steps"))
    def adj_matching(self, obs, rng, flow_steps=None):
        flow_steps = self.config["flow_steps"] if flow_steps is None else flow_steps

        action_dim = self.config['action_dim'] * \
                        (self.config['horizon_length'] if self.config["action_chunking"] else 1)
        x = jax.random.normal(rng, shape=obs.shape[:-1] + (action_dim,))

        actor_slow = self.network.select("target_actor_slow" if self.config["target_actor"] else "actor_slow")

        h = 1 / flow_steps
        xs = [x]
        ts = []
        for i, key in zip(range(flow_steps), jax.random.split(rng, flow_steps)):
            t = i / flow_steps * jnp.ones_like(x[..., 0:1])
            sigma = jnp.sqrt(2 * (1 - t + h) / (t + h))
            noise = jax.random.normal(key, x.shape)
            if i != flow_steps - 1:
                if self.config["residual"]:
                    v = self.network.select("actor_fast")(obs, x, t) + actor_slow(obs, x, t)
                else:
                    v = self.network.select("actor_fast")(obs, x, t)
                x = x + h * (2 * v - x / (t + h)) + jnp.sqrt(h) * sigma * noise
            else:  # use ODE integration for the last step
                
                if self.config["me_am_alpha"] > 0.0 and 1==0:
                    tau = self.config["inv_temp"]
                    inv_alpha = self.config["me_am_alpha"]
                    v_base = actor_slow(obs, x, t)
                    v_target = self.network.select("target_actor_fast" if self.config["target_actor"] else "actor_fast")(obs, x, t)
                    v_final = (tau * v_base + inv_alpha * v_target) / (tau + inv_alpha)
                else:
                    v_final = actor_slow(obs, x, t)
                    
                x = x + h * v_final

            xs.append(x)
            ts.append(t)

        # Compute the critic's action gradient as the adjoint state initialization
        critic_network = "target_critic" if self.config["use_target_grad"] else "critic"
        
        if self.config['clip_adj']:
            grad_fn = jax.grad(lambda x, y: self.network.select(critic_network)(x, jnp.clip(y, -1., 1.)).mean(axis=0).sum(), 1)
        else:
            grad_fn = jax.grad(lambda x, y: self.network.select(critic_network)(x, y).mean(axis=0).sum(), 1)

        critic_grad = grad_fn(obs, xs[-1])
        
        # --- NEW DIAGNOSTICS ---
        # Evaluate how much of the batch is physically out of bounds
        out_of_bounds_mask = (jnp.abs(xs[-1]) > 1.0)
        fraction_out_of_bounds = jnp.mean(out_of_bounds_mask)
        # Evaluate how many scalar gradients in the batch have completely died
        fraction_dead_grad = jnp.mean(critic_grad == 0.0)
        # Evaluate what the critic actually thinks of these generated states
        critic_vals_at_x = self.network.select(critic_network)(obs, xs[-1])
        diag_adj_critic_mean = critic_vals_at_x.mean(axis=0).mean()
        diag_adj_critic_std = critic_vals_at_x.std(axis=0).mean()
        # -----------------------

        critic_grad_norm = jnp.linalg.norm(critic_grad, axis=-1, keepdims=True)
        critic_grad_norm_val = critic_grad_norm.mean()

        if self.config.get("clip_q_grad", True):
            critic_clip_max = self.config.get("q_grad_clip_max", 1.0)
            critic_clip_scale = jnp.minimum(1.0, critic_clip_max / (critic_grad_norm + 1.e-6))
            critic_grad_safe = critic_grad * critic_clip_scale
        else:
            critic_grad_safe = critic_grad

        score_norm_val = 0.0
        fraction_unsafe_variance = 0.0  # Initialize the metric

        if self.config["tau_score"] > 0.:
            # 1. Explicitly query at the microscopic noise floor
            sigmas_min = jnp.full((obs.shape[0], 1), self.config["score_sigma_min"])
            
            # 2. Hard clip as usual
            clipped_actions = jnp.clip(xs[-1], -1., 1.)
            
            # 3. Detect dangerously low variance (The LayerNorm killers)
            action_vars = jnp.var(clipped_actions, axis=-1, keepdims=True)
            is_unsafe = action_vars < 1e-4 
            
            # ---> Calculate the fraction for WandB logging
            fraction_unsafe_variance = jnp.mean(is_unsafe)
            
            # 4. The Decoy: Swap unsafe rows with dummy noise BEFORE entering the network
            dummy_noise = jax.random.normal(rng, clipped_actions.shape)
            safe_actions = jnp.where(is_unsafe, dummy_noise, clipped_actions)
            
            # 5. Predict EPSILON safely (LayerNorm survives)
            pred_epsilon = self.network.select("score_net")(
                obs, safe_actions, sigmas_min
            )
            
            # 6. The Mask: Recover the true score, but FORCE it to 0.0 for the unsafe rows
            raw_score_est = -pred_epsilon / sigmas_min
            score_est = jnp.where(is_unsafe, 0.0, raw_score_est)
            
            score_norm = jnp.linalg.norm(score_est, axis=-1, keepdims=True)
            score_norm_val = score_norm.mean()
            
            if self.config.get("clip_score", True):
                score_clip_max = self.config.get("score_clip_max", 1.0)
                score_clip_scale = jnp.minimum(1.0, score_clip_max / (score_norm + 1.e-6))
                score_est_safe = score_est * score_clip_scale
            else:
                score_est_safe = score_est
            
            # --- COMBINED TEMPERATURE SCALING ---
            tau = self.config["inv_temp"]
            inv_alpha = self.config["me_am_alpha"]
            
            tau_critic = self.config["tau_critic"]
            tau_score = self.config["tau_score"]
            
            total_grad = (tau_critic * critic_grad_safe - tau_score * score_est_safe) / (tau + inv_alpha)
        else:
            total_grad = critic_grad_safe * self.config["tau_critic"]

        adj = -total_grad
        
        pre_adj_info = {
            "adj_max": jnp.abs(adj).max(),
            "adj_std": jnp.abs(adj).std(),
            "adj_mean": jnp.abs(adj).mean(),
            "critic_grad_norm": critic_grad_norm_val,
            "score_norm": score_norm_val,
            # --- NEW DIAGNOSTICS TO WANDB ---
            "diag_frac_out_of_bounds": fraction_out_of_bounds,
            "diag_frac_dead_grad": fraction_dead_grad,
            "diag_frac_unsafe_variance": fraction_unsafe_variance,
            "diag_action_max": xs[-1].max(),
            "diag_action_min": xs[-1].min(),
            "diag_adj_critic_mean": diag_adj_critic_mean,
            "diag_adj_critic_std": diag_adj_critic_std,
        }
        
        adjs = []
        actor_fast = self.network.select("target_actor_fast" if self.config["target_actor"] else "actor_fast")

        for i in reversed(range(flow_steps)):
            t = (i / flow_steps) * jnp.ones_like(x[..., 0:1])

            def fn(xi):
                v_base = actor_slow(obs, xi, t + h)
                
                # --- NEW: INTERPOLATED VECTOR FIELD ---
                if self.config["me_am_alpha"] > 0.0:
                    tau = self.config["inv_temp"]
                    inv_alpha = self.config["me_am_alpha"]
                    v_target = actor_fast(obs, xi, t + h)
                    v_ref = (tau * v_base + inv_alpha * v_target) / (tau + inv_alpha)
                else:
                    v_ref = v_base
                    
                return 2 * v_ref - xi / (t + h)
            
            vjp = jax.vjp(fn, xs[i])[1](adj)[0]
            adj = adj + h * vjp
            
            adjs.append(adj)
            
        return jnp.stack(xs[:-1], axis=0), jnp.stack(list(reversed(adjs)), axis=0), jnp.stack(ts, axis=0), pre_adj_info

    def actor_loss(self, batch, grad_params, rng):
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]
        
        batch_size, full_action_dim = batch_actions.shape
        
        # 9-Way split to accommodate all augmentations
        rng, x_rng, t_rng, adj_rng, edit_rng, mix_mask_rng, aug_rng, target_noise_rng, score_rng = jax.random.split(rng, 9)

        ## BC flow-matching loss.
        x_0 = jax.random.normal(x_rng, (batch_size, full_action_dim))
        
        info = {}
        target_augmentation_loss = 0.0

        # --- Base Augmentations / Target Modifications ---
        if self.config["mixture_prob"] > 0.0:
            base_action_dim = self.config['action_dim']
            horizon = self.config['horizon_length'] if self.config["action_chunking"] else 1
            
            # 1. Determine exact number of elements to augment (Static Integer)
            num_mix = int(batch_size * self.config["mixture_prob"])
            
            if num_mix > 0:
                # 2. Slice out ONLY the elements we are actually going to augment
                obs_mix = jax.tree_util.tree_map(lambda x: x[:num_mix], batch['observations'])
                actions_mix = batch_actions[:num_mix]
                
                # Option A: Learnable Maximized Target Noise

                if self.config.get("target_noise_scale", 0.0) > 0.0:
                    is_edit = self.config.get("augment_as_edit", True)
                    
                    if is_edit:
                        nn_input = jnp.concatenate((obs_mix, actions_mix), axis=-1)
                    else:
                        nn_input = obs_mix

                    target_noise_dist = self.network.select('target_noise_actor')(
                        nn_input, 
                        params=grad_params
                    )
                    
                    # ---------------------------------------------------------
                    # PATH 1: STOCHASTIC (Used ONLY to train the noise generator)
                    # ---------------------------------------------------------
                    target_noise_sample = target_noise_dist.sample(seed=target_noise_rng)
                    scaled_stochastic_noise = target_noise_sample * self.config["target_noise_scale"]
                    
                    # --- RESTORED BEHAVIOR: Add residual if is_edit ---
                    if is_edit:
                        raw_stochastic_actions = actions_mix + scaled_stochastic_noise
                    else:
                        raw_stochastic_actions = scaled_stochastic_noise

                    stochastic_actions = jnp.clip(raw_stochastic_actions, -1.0, 1.0)
                    
                    # Evaluate critic of the stochastic samples to train the generator
                    critic_stochastic = self.network.select('critic')(obs_mix, actions=stochastic_actions)
                    target_noise_critic_loss = -jnp.mean(critic_stochastic, axis=0).mean()
                    
                    target_augmentation_loss = target_noise_critic_loss
                    info["target_noise_critic_loss"] = target_noise_critic_loss
                    info["diag_target_noise_critic_mean"] = jnp.mean(critic_stochastic, axis=0).mean()
                    
                    # --- FLAG FOR ENTROPY REGULARIZATION ---
                    if self.config.get("use_target_noise_entropy", True):
                        target_noise_log_probs = target_noise_dist.log_prob(target_noise_sample)
                        target_alpha = self.network.select('target_noise_alpha')(params=grad_params)
                        target_entropy = -jax.lax.stop_gradient(target_noise_log_probs).mean()
                        
                        target_alpha_loss = (target_alpha * (target_entropy - self.config['target_noise_target_entropy'])).mean()
                        target_entropy_loss = (target_noise_log_probs * self.network.select('target_noise_alpha')()).mean()
                        
                        target_augmentation_loss += target_entropy_loss + target_alpha_loss
                        
                        info["target_noise_entropy"] = target_entropy
                    
                    # ---------------------------------------------------------
                    # PATH 2: DETERMINISTIC (Used ONLY to augment the dataset)
                    # ---------------------------------------------------------

                    if self.config.get("use_gaussian_mode", True):
                        target_noise_mean = target_noise_dist.mode() 
                        selected_noise = target_noise_mean * self.config["target_noise_scale"]
                    else:
                        selected_noise = scaled_stochastic_noise

                    # --- RESTORED BEHAVIOR: Add residual if is_edit ---
                    if is_edit:
                        raw_augmented_actions = actions_mix + selected_noise
                    else:
                        raw_augmented_actions = selected_noise

                    augmented_actions_mix = jnp.clip(raw_augmented_actions, -1.0, 1.0)
                    
                    # Stop gradient so the downstream flow model doesn't backprop into the noise generator
                    augmented_actions_mix = jax.lax.stop_gradient(augmented_actions_mix)
                    
                    info["diag_target_noise_magnitude"] = jnp.mean(jnp.abs(selected_noise))
                    info["diag_target_noise_frac_oob"] = jnp.mean(jnp.abs(raw_augmented_actions) > 1.0)
                # Option B: Heuristic or Structured Data Augmentations
                else:
                    mix_type = self.config.get("mixture_prob_type", "ou")
                    is_edit = self.config.get("augment_as_edit", True)
                    
                    if mix_type == "best_of_n_uniform":
                        N = int(self.config.get("target_best_of_n", 10))
                        
                        if is_edit:
                            margin = self.config.get("mixture_margin", 0.5)
                            
                            # Generate candidates ONLY for the num_mix subset
                            candidate_noises = jax.random.uniform(
                                aug_rng, 
                                (num_mix, N, horizon, base_action_dim), 
                                minval=-margin, maxval=margin
                            )
                            candidate_noises = candidate_noises.reshape((num_mix, N, full_action_dim))
                            
                            expanded_actions_mix = jnp.expand_dims(actions_mix, axis=1)
                            candidates = jnp.clip(expanded_actions_mix + candidate_noises, -1.0, 1.0)
                        else:
                            # Generate full actions globally across the bounds
                            candidate_actions = jax.random.uniform(
                                aug_rng, 
                                (num_mix, N, horizon, base_action_dim), 
                                minval=-1.0, maxval=1.0
                            )
                            candidates = candidate_actions.reshape((num_mix, N, full_action_dim))
                        
                        expanded_obs_mix = jnp.repeat(jnp.expand_dims(obs_mix, axis=1), N, axis=1)
                        flat_obs_mix = jax.tree_util.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), expanded_obs_mix)
                        flat_candidates = candidates.reshape((-1, full_action_dim))
                        
                        # Critic evaluation now runs on a tiny fraction of the data
                        critic_values_raw = self.network.select('critic')(flat_obs_mix, actions=jax.lax.stop_gradient(flat_candidates))
                        critic_values = jnp.mean(critic_values_raw, axis=0).reshape((num_mix, N))
                        
                        best_indices = jnp.argmax(critic_values, axis=-1)
                        mix_indices = jnp.arange(num_mix)
                        augmented_actions_mix = candidates[mix_indices, best_indices, :]
                        
                        # --- NEW DIAGNOSTICS TO WANDB ---
                        info["diag_mix_critic_std"] = jnp.std(critic_values_raw, axis=0).mean()
                        info["diag_mix_critic_mean"] = critic_values.mean()

                    else:
                        # Standard additive noises (All strictly sized to num_mix)
                        if mix_type == "ou":
                            theta = self.config.get("ou_theta", 0.15)
                            sigma = self.config.get("ou_sigma", 0.2)
                            
                            def ou_step(x_prev, key):
                                noise = jax.random.normal(key, (num_mix, base_action_dim))
                                x_new = x_prev * (1.0 - theta) + sigma * noise
                                return x_new, x_new
                                
                            ou_keys = jax.random.split(aug_rng, horizon)
                            init_state = jnp.zeros((num_mix, base_action_dim))
                            _, generated_noise = jax.lax.scan(ou_step, init_state, ou_keys)
                            generated_noise = jnp.transpose(generated_noise, (1, 0, 2)).reshape((num_mix, -1))
                            
                        elif mix_type == "uniform":
                            margin = self.config.get("mixture_margin", 0.5)
                            base_noise = jax.random.uniform(aug_rng, (num_mix, 1, base_action_dim), 
                                                            minval=-margin, maxval=margin)
                            generated_noise = jnp.repeat(base_noise, horizon, axis=1).reshape((num_mix, -1))

                        elif mix_type == "independent_uniform":
                            margin = self.config.get("mixture_margin", 0.5)
                            generated_noise = jax.random.uniform(aug_rng, (num_mix, horizon, base_action_dim), 
                                                                 minval=-margin, maxval=margin)
                            generated_noise = generated_noise.reshape((num_mix, -1))
                            
                        elif mix_type == "gaussian":
                            sigma = self.config.get("mixture_sigma", 0.2)
                            base_noise = jax.random.normal(aug_rng, (num_mix, 1, base_action_dim)) * sigma
                            generated_noise = jnp.repeat(base_noise, horizon, axis=1).reshape((num_mix, -1))
                            
                        else:
                            generated_noise = jnp.zeros_like(actions_mix)
                        
                        augmented_actions_mix = jnp.clip(generated_noise, -1.0, 1.0)
                
                # 3. Concatenate the augmented slice back with the untouched remainder of the batch
                x_1 = jnp.concatenate([augmented_actions_mix, batch_actions[num_mix:]], axis=0)
            else:
                x_1 = batch_actions
        else:
            x_1 = batch_actions

        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_slow')(batch['observations'], x_t, t, params=grad_params)
        flow_loss = jnp.mean(jnp.square(pred - vel).mean(axis=-1) * batch["valid"][..., -1])
        
        # Include the learnable target noise objective
        actor_loss = flow_loss + target_augmentation_loss
        
        total_fast_loss = 0
        actor_slow = self.network.select("target_actor_slow" if self.config["target_actor"] else "actor_slow")
        
        ## Adjoint-matching
        xs, adjs, ts, pre_adj_info = self.adj_matching(batch["observations"], adj_rng)
        

        # === SCORE NET TRAINING (EPSILON PREDICTION) ===
        if self.config["tau_score"] > 0.0:
            if self.config["score_mode"] == "slow":
                model_for_score = "slow"
            else: 
                model_for_score = "slow,fast" if self.config["residual"] else "fast"

            flow_rng, sigma_rng, eps_rng = jax.random.split(score_rng, 3)

            flow_actions = self.compute_flow_actions(
                batch["observations"], 
                jax.random.normal(flow_rng, (batch_size, full_action_dim)),
                model=model_for_score,
                use_target=True 
            )
            
            # 1. Log-Uniform Sampling of Sigma
            log_sig_min = jnp.log(self.config["score_sigma_min"])
            log_sig_max = jnp.log(self.config["score_sigma_max"])
            log_sigmas = jax.random.uniform(sigma_rng, (batch_size, 1), minval=log_sig_min, maxval=log_sig_max)
            sigmas = jnp.exp(log_sigmas)
            
            # 2. Perturb Actions physically
            epsilon = jax.random.normal(eps_rng, flow_actions.shape)
            perturbed_actions = flow_actions + sigmas * epsilon
            
            # 3. Predict the NOISE (epsilon), not the score
            predicted_epsilon = self.network.select('score_net')(
                batch["observations"], perturbed_actions, sigmas, params=grad_params
            )
            
            # 4. Standard MSE on the noise
            score_error = jnp.square(predicted_epsilon - epsilon).mean(axis=-1)
            score_loss = jnp.mean(score_error * batch["valid"][..., -1])
            
            actor_loss += score_loss
            info["score_loss"] = score_loss
            info["diag_sigma_mean"] = sigmas.mean()

        h = 1 / self.config["flow_steps"]
        sigmas = jnp.sqrt(2 * (1 - ts + h) / (ts + h))

        observations = jnp.repeat(batch["observations"][None], self.config["flow_steps"], axis=0)
        vf_fine = self.network.select("actor_fast")(observations, xs, ts, params=grad_params)

        vf_base = actor_slow(observations, xs, ts)
        
        # --- NEW: Log the norm of vf_base ---
        info["diag_vf_base_norm"] = jnp.linalg.norm(vf_base, axis=-1).mean()
        # ------------------------------------
        
        # --- NEW: INTERPOLATED REFERENCE FLOW FOR REGRESSION ---
        if self.config["me_am_alpha"] > 0.0:
            tau = self.config["inv_temp"]
            inv_alpha = self.config["me_am_alpha"]
            actor_fast_target = self.network.select("target_actor_fast" if self.config["target_actor"] else "actor_fast")
            vf_target = actor_fast_target(observations, xs, ts)
            
            # --- NEW: Log the norm of vf_target ---
            info["diag_vf_target_norm"] = jnp.linalg.norm(vf_target, axis=-1).mean()
            # --------------------------------------
            
            vf_ref = (tau * vf_base + inv_alpha * vf_target) / (tau + inv_alpha)
        else:
            vf_ref = vf_base
            # --- NEW: Default logging if me_am_alpha is 0 ---
            info["diag_vf_target_norm"] = 0.0
            # ------------------------------------------------
    
        
        # Compute the adjoint matching loss against the interpolated target
        if self.config["residual"]:
            adj_loss = jnp.sum(jnp.square(vf_fine * 2 / sigmas + sigmas * adjs), axis=-1)
        else:
            adj_loss = jnp.sum(jnp.square((vf_fine - vf_ref) * 2 / sigmas + sigmas * adjs), axis=-1)

        adj_loss = jnp.mean(jnp.sum(adj_loss, axis=0))

        info["adj_loss"] = adj_loss
        info.update(pre_adj_info)
        total_fast_loss += adj_loss

        if self.config["fql_alpha"] > 0.:
            edit_base_rng, edit_rng = jax.random.split(edit_rng, 2)
            fql_noises = jax.random.normal(edit_base_rng, (batch_size, full_action_dim))
            flow_actions = self.compute_flow_actions(batch["observations"], 
                fql_noises, 
                model="slow,fast" if self.config["residual"] else "fast")
            
            os_actions = self.network.select('one_step_actor')(
                batch["observations"], fql_noises, 
                params=grad_params)
            fql_distill_loss = jnp.mean((flow_actions - os_actions) ** 2)
            
            os_actions = jnp.clip(os_actions, -1, 1)
            fql_critic_vals = self.network.select(f'critic')(batch['observations'], actions=os_actions)
            fql_critic_val = jnp.mean(fql_critic_vals, axis=0)
            fql_critic_loss = -fql_critic_val.mean()

            info["fql_distill_loss"] = fql_distill_loss
            info["fql_critic_loss"] = fql_critic_loss

            actor_loss += fql_critic_loss + fql_distill_loss * self.config["fql_alpha"]

        if self.config["edit_scale"] > 0.:
            edit_base_rng, edit_rng = jax.random.split(edit_rng, 2)
            flow_actions = self.compute_flow_actions(batch["observations"], 
                jax.random.normal(edit_base_rng, (batch_size, full_action_dim)), 
                model="slow,fast" if self.config["residual"] else "fast")
            
            edit_dist = self.network.select('edit_actor')(
                jnp.concatenate((batch["observations"], flow_actions), axis=-1), 
                params=grad_params)
            edit = edit_dist.sample(seed=edit_rng)
            edit_log_probs = edit_dist.log_prob(edit)
            
            edited_actions = flow_actions + edit * self.config["edit_scale"]
            
            edited_actions = jnp.clip(edited_actions, -1, 1)
            critic_vals = self.network.select(f'critic')(batch['observations'], actions=edited_actions)
            critic_val = jnp.mean(critic_vals, axis=0)
            edit_critic_loss = -critic_val.mean()

            edit_entropy_loss = (edit_log_probs * self.network.select('edit_alpha')()).mean()

            alpha = self.network.select('edit_alpha')(params=grad_params)
            entropy = -jax.lax.stop_gradient(edit_log_probs).mean()
            edit_alpha_loss = (alpha * (entropy - self.config['edit_target_entropy'])).mean()

            actor_loss += edit_critic_loss + edit_entropy_loss + edit_alpha_loss

            info["edit_critic_loss"] = edit_critic_loss
            info["edit_entropy_loss"] = edit_entropy_loss
            info["edit_alpha_loss"] = edit_alpha_loss
            info["edit_entropy"] = entropy
            info["edit_alpha"] = alpha

        return actor_loss + total_fast_loss, {'flow_loss': flow_loss, "fast_loss": total_fast_loss, **info}
    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @staticmethod
    def _update(agent, batch):
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, 'critic')
        agent.target_update(new_network, 'actor_slow')
        agent.target_update(new_network, 'actor_fast')

        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)
    
    @jax.jit
    def batch_update(self, batch):
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    @jax.jit
    def sample_actions(
        self,
        observations,
        rng,
    ):
        rng, edit_rng = jax.random.split(rng)
        
        action_dim = self.config['action_dim'] * \
                        (self.config['horizon_length'] if self.config["action_chunking"] else 1)
        noises = jax.random.normal(
            rng,
            (
                *observations.shape[: -len(self.config['ob_dims'])],  # batch_size
                self.config["best_of_n"], action_dim
            ),
        )
        observations = jnp.repeat(observations[..., None, :], self.config["best_of_n"], axis=-2)

        if self.config["fql_alpha"] > 0.:  
            actions = self.network.select('one_step_actor')(
                observations, noises)
            actions = jnp.clip(actions, -1, 1)
        else:   
            if self.config["inv_temp"] == 0.:
                actions = self.compute_flow_actions(observations, noises, model="slow")
            else:
                actions = self.compute_flow_actions(observations, noises, model="slow,fast" if self.config["residual"] else "fast")
            if self.config["edit_scale"] > 0.:  
                edit_dist = self.network.select("edit_actor")(jnp.concatenate((observations, actions), axis=-1))
                actions = actions + edit_dist.sample(seed=edit_rng) * self.config["edit_scale"]
            actions = jnp.clip(actions, -1, 1)
        
        q = self.network.select("critic")(observations, actions).mean(axis=0)
        indices = jnp.argmax(q, axis=-1)

        bshape = indices.shape
        indices = indices.reshape(-1)
        bsize = len(indices)
        actions = jnp.reshape(actions, (-1, self.config["best_of_n"], action_dim))[jnp.arange(bsize), indices, :].reshape(
            bshape + (action_dim,))

        return actions

    @partial(jax.jit, static_argnames=["model", "use_target"])
    def compute_flow_actions(
        self,
        observations,
        noises,
        model="slow",
        use_target=False,
    ):
        actions = noises
        prefix = "target_actor" if use_target else "actor"
        networks = [self.network.select(f'{prefix}_{m}') for m in model.split(",")]

        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = sum([network(observations, actions, t) for network in networks])
            
            actions = actions + vels / self.config['flow_steps']

        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        if config["action_chunking"]:
            full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]

        if config['edit_target_entropy'] is None:
            config['edit_target_entropy'] = -config['edit_target_entropy_multiplier'] * full_action_dim

        # Compute entropy target for the learnable noise as well
        if config['target_noise_target_entropy'] is None:
            config['target_noise_target_entropy'] = -config['target_noise_target_entropy_multiplier'] * full_action_dim

        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=config['num_qs'],
        )
        actor_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            layer_norm=config['actor_layer_norm'],
            action_dim=full_action_dim,
        )
        
        # === SCORE NET DEFINITION ===
        score_net_def = ConditionedScoreNet(
            hidden_dims=config['score_net_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=True
        )
        
        # We need a dummy sigma to initialize the flax module shapes correctly
        dummy_sigmas = jnp.ones_like(full_actions[..., :1])
        
        network_info = dict(
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
            actor_fast=(copy.deepcopy(actor_def), (ex_observations, full_actions, ex_times)),
            target_actor_fast=(copy.deepcopy(actor_def), (ex_observations, full_actions, ex_times)),
            actor_slow=(copy.deepcopy(actor_def), (ex_observations, full_actions, ex_times)),
            target_actor_slow=(copy.deepcopy(actor_def), (ex_observations, full_actions, ex_times)),
            score_net=(score_net_def, (ex_observations, full_actions, dummy_sigmas)), # <-- Passes the 3 dummy inputs
        )
        assert (config["fql_alpha"] * config["edit_scale"] == 0.), "Only one of fql_alpha and edit_scale can be non-zero."
        
        if config["fql_alpha"] > 0.:
            network_info.update(dict(
                one_step_actor=(copy.deepcopy(actor_def), (ex_observations, full_actions, None)),
            ))

        if config["edit_scale"] > 0.:
            edit_actor_base_cls = partial(MLP, hidden_dims=config["actor_hidden_dims"], activate_final=True)
            edit_actor_def = TanhNormal(edit_actor_base_cls, full_action_dim)
            alpha_def = LogParam()

            network_info.update(dict(
                edit_actor=(edit_actor_def, jnp.concatenate((ex_observations, full_actions), axis=-1)),
                edit_alpha=(alpha_def, ()),
            ))
            
        # Add target_noise actor and alpha networks 
        if config["target_noise_scale"] > 0.:
            target_noise_actor_base_cls = partial(MLP, hidden_dims=config["actor_hidden_dims"], activate_final=True)
            target_noise_actor_def = TanhNormal(target_noise_actor_base_cls, full_action_dim)
            target_noise_alpha_def = LogParam()
            
            # Determine input shape based on the flag
            if config.get("augment_as_edit", True):
                dummy_input = jnp.concatenate((ex_observations, full_actions), axis=-1)
            else:
                dummy_input = ex_observations

            network_info.update(dict(
                target_noise_actor=(target_noise_actor_def, dummy_input),
                target_noise_alpha=(target_noise_alpha_def, ()),
            ))
        
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)

        if config["clip_grad"]:
            network_tx = optax.chain(optax.clip_by_global_norm(max_norm=1.0), 
                optax.adam(learning_rate=config["lr"]))
        else:
            network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']
        params['modules_target_actor_slow'] = params['modules_actor_slow']
        params['modules_target_actor_fast'] = params['modules_actor_fast']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='meam',  
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            
            ## Common hyperparameters
            lr=3e-4,  
            batch_size=256,  
            actor_hidden_dims=(512, 512, 512, 512),  
            actor_layer_norm=False,
            value_hidden_dims=(512, 512, 512, 512),  
            value_layer_norm=True,
            
            ## Q-chunking hyperparameters
            horizon_length=ml_collections.config_dict.placeholder(int), 
            action_chunking=False,                                      
            
            ## RL hyperparameters
            num_qs=10,       
            rho=0.5,        

            discount=0.99,  
            tau=0.005,      
            flow_steps=10,  

            best_of_n=1,    
            
            ## Main hyperparameter(s)
            inv_temp=0.3,   
            tau_critic=2.0,
            tau_score=0.1,
            
            fql_alpha=0.,   
            edit_scale=0.,  
            
            ## MEAM Components 
            me_am_alpha=0.0,            
            mixture_prob=0.1,           
            
            # --- The Data Augmentation / Target Modification Hub ---
            mixture_prob_type='independent_uniform', # 'ou', 'uniform', 'independent_uniform', 'gaussian'
            mixture_margin=1.0,                      # Used for 'uniform' and 'independent_uniform'
            mixture_sigma=0.2,                       # Used for 'gaussian'
            ou_theta=0.15,                           # Used for 'ou'
            ou_sigma=0.2,                            # Used for 'ou'

            target_best_of_n=10,
            
            # Target Noise (Learnable Q-Maximized Targets)
            target_noise_scale=0.1,                  # Set > 0.0 to enable the learnable target generator
            use_gaussian_mode=True,
            target_noise_target_entropy=ml_collections.config_dict.placeholder(float),
            target_noise_target_entropy_multiplier=0.5,
            augment_as_edit=True,
            use_target_noise_entropy=True,
            
            score_net_hidden_dims=(256, 256),
            score_sigma_min=3e-1,     # Microscopic noise for Adjoint initialization
            score_sigma_max=0.7,      # Spread for exploring the manifold       
            score_mode='slow',
            score_sigma=0.1,
            clip_q_grad=False,
            q_grad_clip_max=1.0,
            clip_score=False,
            score_clip_max=1.0,
            
            ## Other variants/hyperparameter(s)
            target_actor=True,
            residual=False,
            clip_adj=True,
            clip_grad=True,
            use_target_grad=True,
            edit_target_entropy=ml_collections.config_dict.placeholder(float),  
            edit_target_entropy_multiplier=0.5,  
        )
    )
    return config
