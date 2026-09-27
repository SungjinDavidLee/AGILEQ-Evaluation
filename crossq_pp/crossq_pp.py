from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from torch.nn import functional as F
from stable_baselines3.common.utils import polyak_update
import torch.nn as nn


from sb3_contrib.crossq.policies import Actor, CrossQCritic, CrossQPolicy, MlpPolicy, MultiInputPolicy

SelfCrossQ = TypeVar("SelfCrossQ", bound="CrossQpp")


class CrossQpp(OffPolicyAlgorithm):


    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "MultiInputPolicy" : MultiInputPolicy,
    }
    policy: CrossQPolicy
    actor: Actor
    critic: CrossQCritic
    critic_target: Optional[CrossQCritic]

    def __init__(
        self,
        policy: Union[str, Type[CrossQPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_entropy: Union[str, float] = "auto",
        policy_delay: int = 3,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,

        reset_interval: Optional[int] = 200_000,
        reset_critic: bool = False,
        reset_actor: bool = False,
        reset_encoder: bool = False,
        use_target_network: bool = False,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
        )

        self.target_entropy = target_entropy
        self.log_ent_coef = None
        self.ent_coef = ent_coef
        self.ent_coef_optimizer: Optional[th.optim.Adam] = None
        self.policy_delay = policy_delay


        self.reset_interval = reset_interval
        self.reset_critic = reset_critic
        self.reset_actor = reset_actor
        self.reset_encoder = reset_encoder
        self.use_target_network = use_target_network
        self.critic_target = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()


        if self.use_target_network:
            if self.policy.share_features_extractor:
                self.critic_target = self.policy.make_critic(features_extractor=self.actor.features_extractor)
            else:
                self.critic_target = self.policy.make_critic(features_extractor=None)

            self.critic_target.load_state_dict(self.critic.state_dict())
            self.critic_target.set_training_mode(False)
            self.critic_target.requires_grad_(False)
        else:
            self.critic_target = None


        if self.target_entropy == "auto":
            self.target_entropy = float(-np.prod(self.env.action_space.shape).astype(np.float32))
        else:
            self.target_entropy = float(self.target_entropy)


        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)

        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for _ in range(gradient_steps):
            self._n_updates += 1


            if self.reset_interval is not None and self._n_updates % self.reset_interval == 0:

                def reset_weights(m):
                    if isinstance(m, (th.nn.Linear, th.nn.Conv2d)):
                        m.reset_parameters()

                did_reset = False


                if self.reset_encoder:
                    self.actor.features_extractor.apply(reset_weights)
                    if not self.policy.share_features_extractor:
                        self.critic.features_extractor.apply(reset_weights)
                    did_reset = True
                    if self.verbose > 0: print(" -> Encoder (CNN) Reset!")


                if self.reset_critic:
                    self.critic.q_networks.apply(reset_weights)
                    self.critic.optimizer.state.clear()
                    did_reset = True
                    if self.verbose > 0: print(" -> Critic Head (MLP) Reset!")


                if self.reset_actor:
                    self.actor.latent_pi.apply(reset_weights)
                    self.actor.mu.apply(reset_weights)
                    self.actor.log_std.apply(reset_weights)
                    self.actor.optimizer.state.clear()

                    if self.ent_coef_optimizer is not None:
                        self.ent_coef_optimizer.state.clear()

                    did_reset = True
                    if self.verbose > 0: print(" -> Actor Head (MLP) Reset!")

                if did_reset and self.critic_target is not None:
                    self.critic_target.load_state_dict(self.critic.state_dict())
                    if self.verbose > 0: print(f"INFO: Custom Reset & Target Sync performed at step {self._n_updates}")

            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            if self.use_sde:
                self.actor.reset_noise()

            if self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            with th.no_grad():
                self.actor.set_bn_training_mode(False)
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)


            if isinstance(replay_data.observations, dict):
                all_obs = {
                    key: th.cat([replay_data.observations[key], replay_data.next_observations[key]], dim=0)
                    for key in replay_data.observations.keys()
                }
            else:
                all_obs = th.cat([replay_data.observations, replay_data.next_observations], dim=0)

            all_actions = th.cat([replay_data.actions, next_actions], dim=0)


            self.critic.set_bn_training_mode(True)
            all_q_values = th.cat(self.critic(all_obs, all_actions), dim=1)
            self.critic.set_bn_training_mode(False)

            current_q_values_student, next_q_values_student = th.split(all_q_values, batch_size, dim=0)
            current_q_values = current_q_values_student.T[..., None]


            with th.no_grad():
                if self.use_target_network and self.critic_target is not None:

                    target_q_out = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                    next_q_values_target = target_q_out
                else:

                    next_q_values_target = next_q_values_student

                next_q_values, _ = th.min(next_q_values_target, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values.detach()) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()

            self.critic.optimizer.step()


            if self.use_target_network and self.critic_target is not None:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

            if self._n_updates % self.policy_delay == 0:
                self.actor.set_bn_training_mode(True)
                actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
                log_prob = log_prob.reshape(-1, 1)
                self.actor.set_bn_training_mode(False)

                if self.ent_coef_optimizer is not None:
                    ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                    ent_coef_losses.append(ent_coef_loss.item())

                    self.ent_coef_optimizer.zero_grad()
                    ent_coef_loss.backward()
                    self.ent_coef_optimizer.step()

                self.critic.set_bn_training_mode(False)
                q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)

                min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
                actor_loss = (ent_coef * log_prob.reshape(-1, 1) - min_qf_pi).mean()
                actor_losses.append(actor_loss.item())

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def learn(
        self: SelfCrossQ,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "CrossQ",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfCrossQ:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> List[str]:
        return [*super()._excluded_save_params(), "actor", "critic", "critic_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]

        if self.critic_target is not None:
            state_dicts.append("critic_target")

        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables
