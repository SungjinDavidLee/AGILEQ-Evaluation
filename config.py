import torch as th
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
from utils import lr_schedule


import torch as th
import torch.nn as nn
import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class XtMaCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim=256):
        super().__init__(observation_space, features_dim=features_dim)

        # token embedding dim
        self.d_token = 128

        n_input_channels = observation_space['bev_image'].shape[0]

        self.cnn = nn.Sequential(   # input : (N, C, 196, 196)
            # Layer 1: (196 - 5) / 2 + 1 = 96
            nn.Conv2d(n_input_channels, 8, kernel_size=5, stride=2), # (8, 96, 96)
            nn.ReLU(),
            
            # Layer 2: (96 - 5) / 2 + 1 = 46
            nn.Conv2d(8, 16, kernel_size=5, stride=2), # (16, 46, 46)
            nn.ReLU(),
            
            # Layer 3: (46 - 5) / 2 + 1 = 21
            nn.Conv2d(16, 16, kernel_size=5, stride=2), # (16, 21, 21)
            nn.ReLU(),
            
            # Layer 4: (21 - 3) / 2 + 1 = 10
            nn.Conv2d(16, 32, kernel_size=3, stride=2), # (32, 10, 10)
            nn.ReLU(),
            
            # Layer 5: (10 - 3) / 1 + 1 = 8
            nn.Conv2d(32, 32, kernel_size=3, stride=1), # (32, 8, 8)
            nn.ReLU(),
            
            # Layer 6: (8 - 3) / 2 + 1 = 3
            nn.Conv2d(32, 64, kernel_size=3, stride=2), # (64, 3, 3)
            nn.ReLU(),
            
            # Layer 7: (3 - 3) / 1 + 1 = 1
            nn.Conv2d(64, 128, kernel_size=3, stride=1), # (128, 1, 1)
            
            nn.Flatten(), # 128 * 1 * 1 = 128
            nn.LayerNorm(self.d_token)
        )

        with th.no_grad():
            sample = th.as_tensor(observation_space['bev_image'].sample()[None]).float()
            cnn_out = self.cnn(sample)
            self.cnn_output_dim = cnn_out.shape[1]
            if self.cnn_output_dim != self.d_token:
                raise ValueError(f"CNN output dim({self.cnn_output_dim}) != d_model({self.d_token}). Check CNN architecture.")
            
        self.n_ego = observation_space['vehicle_measures'].shape[0] - 29

        # Token A : Ego state - Steer, Throttle, Brake, Speed, Jerk, speed_dev
        self.embed_ego = nn.Sequential(
            nn.Linear(self.n_ego, self.d_token),
            nn.LayerNorm(self.d_token)
        )

        # Token D : Traffic Light - One hot vector (4) + stop line dist (1)
        self.embed_traffic = nn.Sequential(
            nn.Linear(5, self.d_token),
            nn.LayerNorm(self.d_token)
        )

        # Token C : waypoints - 7 points * 2 coords = 14
        self.embed_wp = nn.Sequential(
            nn.Linear(14, self.d_token),
            nn.LayerNorm(self.d_token)
        )

        # Token E : Previous state - prev control (3) + prev speed (1) + prev jerk (1) + prev TL (4) + prev stop line dist (1)
        self.embed_prev = nn.Sequential(
            nn.Linear(10, self.d_token),
            nn.LayerNorm(self.d_token)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model = self.d_token,
            nhead = 4,
            dim_feedforward = 512,
            dropout = 0.0,
            activation = 'relu',
            batch_first = True,
            norm_first = True,
        )
        self.Transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, observations):
        measures = observations['vehicle_measures']

        bev_data = self.cnn(observations['bev_image'])

        # Token A: Ego
        # steer, throttle, brake, speed, jerk
        ego_data = measures[:, 0:self.n_ego]

        # Token D: Traffic / Rule
        # traffic_light_state(4), stop_line(1)
        traffic_data = measures[:, self.n_ego:self.n_ego + 5]

        # Token C: Waypoints
        # 7 * 2 = 14
        waypoint_data = measures[:, self.n_ego + 5:self.n_ego + 19]

        # Token E: Previous
        # prev ego_data (5) + prev tl(4) + prev stop line(1)
        prev_data = measures[:, self.n_ego + 19:self.n_ego + 29]
                    
        # Embedding & Unsqueeze -> (Batch, 1, 128)
        
        token_a = self.embed_ego(ego_data).unsqueeze(1)
        token_d = self.embed_traffic(traffic_data).unsqueeze(1)
        token_c = self.embed_wp(waypoint_data).unsqueeze(1)
        token_e = self.embed_prev(prev_data).unsqueeze(1)
        token_b = bev_data.unsqueeze(1)
        


        x = th.cat([token_a, token_b, token_c, token_d, token_e], dim=1)
        x = self.Transformer(x)
        x = x.flatten(start_dim = 1)

        return x




policy_kwargs = dict(
    features_extractor_class=XtMaCombinedExtractor,
    features_extractor_kwargs=dict(features_dim=640),
    net_arch=[512,512],
    use_sde=True,
    share_features_extractor=True,
)

policy_kwargs_AdamW = dict(
    features_extractor_class=XtMaCombinedExtractor,
    features_extractor_kwargs=dict(features_dim=640),
    net_arch=[1024,1024],
    use_sde=True,
    share_features_extractor=True,
    optimizer_class=th.optim.Adam,
)




algorithm_params = {
    "PPO": dict(
        learning_rate=lr_schedule(5e-4, 1e-6, 2),
        gamma=0.99,
        batch_size=256,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        n_epochs=10,
        n_steps=1024,
        policy_kwargs=policy_kwargs,  
    ),
    "SAC": dict(
        learning_rate=lr_schedule(5e-4, 1e-6, 2),
        buffer_size=30000,
        batch_size=256,
        ent_coef='auto',
        gamma=0.99,
        tau=0.02,
        train_freq=64,
        gradient_steps=64,
        learning_starts=10000,
        use_sde=True,
        policy_kwargs=policy_kwargs,  
    ),
    "DDPG": dict(
        gamma=0.99,
        buffer_size=300000,
        learning_starts=10000,
        action_noise=NormalActionNoise(mean=np.zeros(2), sigma=0.5 * np.ones(2)),
        gradient_steps=-1,
        learning_rate=lr_schedule(5e-4, 1e-6, 2),
        policy_kwargs=policy_kwargs,  
    ),
    "TD3": dict(
        gamma=0.99,
        batch_size=256,
        learning_starts=10000,
        buffer_size=30000,
        action_noise=NormalActionNoise(mean=np.zeros(2), sigma=0.5 * np.ones(2)),
        train_freq=64,
        gradient_steps=-1,
        learning_rate=lr_schedule(5e-4, 1e-6, 2),
        policy_kwargs=policy_kwargs,  
    ),
    
    "TQC": dict(
        learning_rate=lr_schedule(5e-4, 1e-6, 2),
        buffer_size=30000,
        batch_size=256,
        ent_coef="auto",
        gamma=0.99,
        tau=0.02,
        gradient_steps=64,
        learning_starts=10000,
        train_freq=64,
        use_sde=True,
        policy_kwargs=policy_kwargs,  
    ),

    "crossq": dict(
        learning_rate=3e-4,
        buffer_size=35000,
        batch_size=256,
        ent_coef="auto",  # automatic entropy tuning
        gamma=0.98,
        train_freq=(64,"step"),
        gradient_steps=64,
        learning_starts=10000,
        use_sde=True,
        policy_kwargs=policy_kwargs, 
    ),

    "crossq_pp": dict(
        learning_rate=3e-4,
        buffer_size=35000,
        batch_size=512,
        tau = 1.0, # tau = 1.0 is equivalent to having no target network
        ent_coef="auto",  # automatic entropy tuning
        gamma=0.98,
        train_freq=(64,"step"),
        gradient_steps=64,
        learning_starts=10000,
        use_sde=True,
        policy_kwargs=policy_kwargs_AdamW, 
        reset_interval=200_000,
        reset_critic = False,
        reset_actor=False,
        reset_encoder=False,
        use_target_network= False,
    )
}

states = {
    "test": ["steer", "throttle", "brake", "speed", "bev_image", "traffic_light_state", "jerk", "waypoints", "previous_state", "stop_line", "right_turn_indicator", "bypass_indicator"],
    "2": ["steer", "throttle", "speed", "maneuver"],
}

reward_params = {
    "reward_params": dict(
        early_stop=True,
        min_speed=20.0,  # km/h
        max_speed=40.0,  # km/h
        target_speed=30.0,  # kmh
        max_distance=2.0,  # Max distance from center before terminating
        max_std_center_lane=0.35,
        max_angle_center_lane=90,
        max_jerk=30,
        penalty_reward=-1.0,
    ),
}

CONFIG_PPO = {
    "algorithm": "PPO",
    "algorithm_params": algorithm_params["PPO"],
    "state": states["test"],
    "vae_model": "vae_64",
    "action_smoothing": 0.5,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_params"],
    "obs_res": (160, 80),
    "seed": 100,
    "wrappers": []
}

CONFIG_SAC = {
    "algorithm": "SAC",
    "algorithm_params": algorithm_params["SAC"],
    "state": states["test"],
    "vae_model": "vae_64",
    "action_smoothing": 0.5,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_params"],
    "obs_res": (160, 80),
    "seed": 100,
    "wrappers": []
}

CONFIG_DDPG = {
    "algorithm": "DDPG",
    "algorithm_params": algorithm_params["DDPG"],
    "state": states["test"],
    "vae_model": "vae_64",
    "action_smoothing": 0.5,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_params"],
    "obs_res": (160, 80),
    "seed": 100,
    "wrappers": []
}

CONFIG_TD3 = {
    "algorithm": "TD3",
    "algorithm_params": algorithm_params["TD3"],
    "state": states["test"],
    "vae_model": "vae_64",
    "action_smoothing": 0.5,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_params"],
    "obs_res": (160, 80),
    "seed": 100,
    "wrappers": []
}

CONFIG_TQC = {
    "algorithm": "TQC",
    "algorithm_params": algorithm_params["TQC"],
    "state": states["test"],
    "vae_model": "vae_64",
    "action_smoothing": 0.5,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_params"],
    "obs_res": (240, 160),
    "seed": 100,
    "wrappers": []
}

CONFIG_CROSSQ = {
    "algorithm": "crossq",
    "algorithm_params": algorithm_params["crossq"],
    "state": states["test"],
    "vae_model": "vae_64",
    "action_smoothing": 0.5,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_params"],
    "obs_res": (160, 80),
    "seed": 100,
    "wrappers": []
}

CONFIG_CROSSQ_PP = {
    "algorithm": "crossq_pp",
    "algorithm_params": algorithm_params["crossq_pp"],
    "state": states["test"],
    "vae_model": "vae_64",
    "action_smoothing": 0.5,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_params"],
    "obs_res": (160, 80),
    "seed": 100,
    "wrappers": []
}

CONFIGS = {
    "PPO": CONFIG_PPO,
    "SAC": CONFIG_SAC,
    "DDPG": CONFIG_DDPG,
    "TD3": CONFIG_TD3,
    "TQC": CONFIG_TQC,
    "crossq": CONFIG_CROSSQ,
    "crossq_pp" : CONFIG_CROSSQ_PP,
}
