import gymnasium as gym
import numpy as np
from gymnasium import spaces
import torch
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from graph_data_loader_slide_SF_RLFT import get_dataloader
from model import TCGCNTransformer
from cold_start import load_model
from pre_training_ztp import _last_step_edge_mask
import wandb
import json
from collections import deque
import os


# ---------------------- reward utilities ----------------------
def compute_sample_weights(y, cfg):
    # down-weight small flows, up-weight tail counts
    y = y.float()
    w = torch.ones_like(y, dtype=torch.float32)

    if cfg.small_flow_thresh > 0:
        mask_1 = (y < cfg.small_flow_thresh)
        if mask_1.any():
            w[mask_1] *= cfg.small_flow_weight

    if hasattr(cfg, "w_y2"):
        mask_2 = (y == 2)
        if mask_2.any():
            w[mask_2] *= cfg.w_y2

    if cfg.weight_mode != "none":
        mask_tail = (y >= cfg.tail_min_count) & (y < cfg.tail_max_count)
        if mask_tail.any():
            if cfg.weight_mode == "log":
                w_tail = 1.0 + cfg.tail_alpha * torch.log(y[mask_tail])
            elif cfg.weight_mode == "power":
                base = (y[mask_tail] / float(cfg.tail_min_count)).clamp_min(1.0)
                w_tail = base ** cfg.tail_alpha
            else:
                w_tail = torch.ones_like(y[mask_tail])
            w[mask_tail] *= torch.clamp(w_tail, max=cfg.w_max)

    return w


def compute_weighted_reward(pred, true, cfg):
    pred = pred.flatten()
    true = true.flatten()
    w = compute_sample_weights(true, cfg)

    if cfg.use_topk:
        E = true.numel()
        K = max(1, int(E * cfg.topk_ratio))
        scores = true if cfg.topk_by == "true" else w * torch.abs(pred - true)
        topk_idx = torch.topk(scores, K, largest=True).indices
        pred, true, w = pred[topk_idx], true[topk_idx], w[topk_idx]

    se = (pred - true) ** 2
    ae = torch.abs(pred - true)
    w_sum = w.sum() + 1e-8
    wmse = (w * se).sum() / w_sum
    wmae = (w * ae).sum() / w_sum

    reward = -(cfg.reward_alpha * wmse + cfg.reward_beta * wmae) / 10
    return reward.item(), wmse.item(), wmae.item()


# ---------------------- PPO callback ----------------------

class WandbCallback(BaseCallback):
    def __init__(self, save_path: str, model, verbose=1, max_len=96, min_step=480):
        super().__init__(verbose)
        self.save_path = save_path
        self.best_reward = -np.inf
        self.last_rewards = deque(maxlen=max_len)
        self.step_count = 0
        self.min_step = min_step
        self.tcgcn_model = model
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        self.step_count += 1

        rewards = self.locals.get("rewards", [0])
        step_reward = rewards[0] if isinstance(rewards, (list, np.ndarray)) else rewards
        self.last_rewards.append(step_reward)

        info = self.locals.get("infos", [{}])[0]
        wandb.log({
            "step": self.step_count,
            "reward": step_reward,
            "wmse": info.get("wmse", 0),
            "wmae": info.get("wmae", 0),
            "mean_reward": np.mean(self.last_rewards),
        })

        current_mean = np.mean(self.last_rewards)
        if current_mean > self.best_reward and self.step_count > self.min_step:
            self.best_reward = current_mean
            self.model.save(f"{self.save_path}/ppo_agent_sf_9d")
            torch.save(self.tcgcn_model.state_dict(), f"{self.save_path}/rl_sf_9d.pth")
            print(f"[Saved] step={self.step_count}  mean_reward={current_mean:.6f}")

        return True


# ---------------------- RL environment ----------------------

class MobilityPredictionEnv(gym.Env):
    def __init__(self, dataloader, model, device,
                 state_dim=1560, weight_action_value=0.001,
                 bias_action_value=5e-5, reward_cfg=None):
        super().__init__()
        self.dataloader = dataloader
        self.model = model
        self.device = device
        self.state_dim = state_dim
        self.weight_action_value = weight_action_value
        self.bias_action_value = bias_action_value
        self.global_step = 0
        self.episode_length = 1
        self.reward_cfg = reward_cfg

        # PCA frequency groups used to partition fc_edge_out weights
        print("Loading PCA frequency groups from ./model/PCA_results/sf_rl_9d.json")
        with open("./model/PCA_results/sf_rl_9d.json") as f:
            freq = json.load(f)
        self.low_group = freq["low"]
        self.mid_group = freq["mid"]
        self.high_group = freq["high"]
        print(f"low={len(self.low_group)}, mid={len(self.mid_group)}, high={len(self.high_group)}")

        # action: multiplicative delta for low/mid/high weight groups + additive bias delta
        self.action_space = spaces.Box(
            low=np.array([-self.weight_action_value, -self.weight_action_value,
                          -self.weight_action_value, -self.bias_action_value], dtype=np.float32),
            high=np.array([self.weight_action_value, self.weight_action_value,
                           self.weight_action_value, self.bias_action_value], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

        self.data_iterator = None
        self.current_batch = None
        self.reset()

    def _apply_action(self, action):
        # multiplicative update on fc_edge_out weight groups + additive bias shift
        a_low, a_mid, a_high, a_bias = [float(a) for a in action]
        weight = self.model.transformer_decoder.fc_edge_out.weight.data
        bias = self.model.transformer_decoder.fc_edge_out.bias.data

        if self.low_group:
            weight[0, self.low_group] *= (1 + a_low)
        if self.mid_group:
            weight[0, self.mid_group] *= (1 + a_mid)
        if self.high_group:
            weight[0, self.high_group] *= (1 + a_high)

        bias.add_(a_bias)
        weight.clamp_(min=-2, max=2)
        bias.clamp_(min=-2, max=2)

    def _get_state(self, batch):
        # state = normalised temporal-diff statistics of edge embeddings [mean, std, max, min]
        x_batch, edge_index_batch, _ = batch
        x_batch = x_batch.to(self.device)
        edge_index_batch = edge_index_batch.to(self.device)

        with torch.no_grad():
            batch_size, num_nodes, _, time_steps = x_batch.shape

            poi_idx = x_batch[:, :, 1, :].long()
            other = torch.cat([x_batch[:, :, 0:1, :], x_batch[:, :, 2:, :]], dim=2)
            poi_embed = self.model.emb_drop(
                self.model.poi_embedding(poi_idx).permute(0, 1, 3, 2)
            )
            x_enc = torch.cat([other, poi_embed], dim=2)
            x_enc = self.model.input_projection(
                x_enc.permute(0, 1, 3, 2)
            ).permute(0, 1, 3, 2)
            x_enc = self.model.temporal_encoding(x_enc, time_steps)

            edge_emb, _ = self.model.tc_gcn(x_enc, edge_index_batch)  # [T, B*E, H]
            temporal_diff = edge_emb[1:] - edge_emb[:-1]              # [T-1, B*E, H]

            feat_mean = temporal_diff.mean(dim=(0, 1))
            feat_std = temporal_diff.std(dim=(0, 1))
            feat_max = torch.amax(temporal_diff, dim=(0, 1))
            feat_min = torch.amin(temporal_diff, dim=(0, 1))

            state = torch.cat([feat_mean, feat_std, feat_max, feat_min], dim=0)
            state = (state - state.mean()) / (state.std() + 1e-8)

        return state.cpu().numpy()

    def reset(self, *, seed=None, options=None):
        self.global_step = 0
        self.step_count = 0
        self.data_iterator = iter(self.dataloader)
        try:
            self.current_batch = next(self.data_iterator)
        except StopIteration:
            self.data_iterator = iter(self.dataloader)
            self.current_batch = next(self.data_iterator)
        return self._get_state(self.current_batch), {}

    def step(self, action):
        self.step_count += 1
        self.global_step += 1

        self._apply_action(action)

        x, edge_index, edge_attr = self.current_batch
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device)

        B, N, _, T = x.shape
        last_mask = _last_step_edge_mask(edge_index, B, N, T, self.device)
        true = edge_attr[last_mask]

        with torch.no_grad():
            pred = self.model(x, edge_index).squeeze().flatten()

        reward, wmse, wmae = compute_weighted_reward(pred, true, self.reward_cfg)

        if self.step_count >= self.episode_length:
            try:
                self.current_batch = next(self.data_iterator)
            except StopIteration:
                self.data_iterator = iter(self.dataloader)
                self.current_batch = next(self.data_iterator)
            terminated = True
            self.step_count = 0
        else:
            reward = 0.0
            terminated = False

        next_state = self._get_state(self.current_batch)
        return next_state, reward, terminated, False, {"wmse": wmse, "wmae": wmae}


# ---------------------- main ----------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="./model/cold_start_sf_9d.pth")
    parser.add_argument('--state_dim', type=int, default=1024)
    parser.add_argument('--weight_action_value', type=float, default=4e-4)
    parser.add_argument('--bias_action_value', type=float, default=5e-8)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--n_steps', type=int, default=128)
    parser.add_argument('--n_epochs', type=int, default=3)
    parser.add_argument('--batch_size_rl', type=int, default=128)
    parser.add_argument('--reward_alpha', type=float, default=0.25)
    parser.add_argument('--reward_beta', type=float, default=1.0)
    parser.add_argument('--small_flow_thresh', type=int, default=2)
    parser.add_argument('--small_flow_weight', type=float, default=0.05)
    parser.add_argument('--w_y2', type=float, default=0.10)
    parser.add_argument('--weight_mode', type=str, default='log', choices=['none', 'log', 'power'])
    parser.add_argument('--tail_alpha', type=float, default=2.2)
    parser.add_argument('--w_max', type=float, default=4.5)
    parser.add_argument('--tail_min_count', type=int, default=3)
    parser.add_argument('--tail_max_count', type=int, default=8)
    parser.add_argument('--use_topk', type=bool, default=False)
    parser.add_argument('--topk_ratio', type=float, default=0.05)
    parser.add_argument('--topk_by', type=str, default='true', choices=['true', 'weighted_error'])
    args = parser.parse_args()

    wandb.init(project="Paper1_RLFT_PPO")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    sf_loader = get_dataloader(
        gpickle_dir=["./graph_data/SF/RL 9d"],
        batch_size=1, input_dim=7, window_sizes=[12], num_workers=1, stride=6,
    )

    model = TCGCNTransformer(
        input_dim=11, temporal_hidden_dim1=128, temporal_hidden_dim2=256,
        temporal_dropout_rate=0, kernel_size=3, gcn_hidden_dim1=512,
        gcn_hidden_dim2=256, gcn_dropout_rate=0, decoder_hidden_dim=256,
        edge_output_dim=1, num_heads=4, num_layers=2, decoder_dropout_rate=0,
        num_poi_types=456, embed_dim=5,
    ).to(device)
    model = load_model(model, args.model_path)
    print(f"Loaded: {args.model_path}")

    env = MobilityPredictionEnv(
        dataloader=sf_loader, model=model, device=device,
        state_dim=args.state_dim,
        weight_action_value=args.weight_action_value,
        bias_action_value=args.bias_action_value,
        reward_cfg=args,
    )
    vec_env = make_vec_env(lambda: env, n_envs=1)

    ppo = PPO(
        "MlpPolicy", vec_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size_rl,
        n_epochs=args.n_epochs,
        gamma=0.95, gae_lambda=0.9, clip_range=0.2, ent_coef=0.005,
        verbose=1, device=device,
        policy_kwargs=dict(net_arch=dict(pi=[512, 512], vf=[512, 512])),
        normalize_advantage=True,
    )

    callback = WandbCallback(save_path="./model", model=model, max_len=128, min_step=512)
    ppo.learn(total_timesteps=20000, callback=callback)

    wandb.finish()
    print("RL fine-tuning complete.")