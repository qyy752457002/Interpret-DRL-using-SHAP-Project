from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
import gymnasium as gym
import ale_py
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo

gym.register_envs(ale_py)

def make_MsPacman_env():
  env = gym.make("MsPacmanNoFrameskip-v4", render_mode="rgb_array")
  env = AtariWrapper(env)
  env = RecordVideo(env, video_folder="./videos", name_prefix="ppo_MsPacman")  
  return env

# 加载模型
PPO_model = PPO.load("./ppo/best_model.zip")

# 设置 coef 为 0 来关闭探索，完全依赖于模型学到的策略
PPO_model.ent_coef = 0.0
PPO_model.vf_coef = 0.0

n_envs = 1
MsPacman_env = DummyVecEnv([make_MsPacman_env for _ in range(n_envs)])  # 用DummyVecEnv包装环境，使其并行
MsPacman_env = VecTransposeImage(MsPacman_env) # 转换成 (C, H, W) 而不是 (H, W, C)
MsPacman_env = VecFrameStack(MsPacman_env, n_stack=4)  # 堆叠帧，4帧堆叠

num_episodes = 1

for ep in range(num_episodes):
    obs = MsPacman_env.reset() 
    done = False

    while not done:
        action, _states = PPO_model.predict(obs, deterministic=True) 
        next_obs, reward, done, info = MsPacman_env.step(action)

MsPacman_env.close()
