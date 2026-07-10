import carla
import numpy as np
import argparse
import os


# 导入你的 World 环境 (假设存放在 env 模块中)
# from env import World
# from utils.HUD import HUD

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1", type=str)
    parser.add_argument("--port", default=2000, type=int)
    parser.add_argument("--FPS", default=20, type=int)
    parser.add_argument("--spawn_x", default=0.0, type=float)
    parser.add_argument("--spawn_y", default=0.0, type=float)
    parser.add_argument("--desired_speed", default=8.0, type=float)
    parser.add_argument("--waypoint_resolution", default=2.0, type=float)
    parser.add_argument("--waypoint_lookahead_distance", default=10.0, type=float)
    parser.add_argument("--control_mode", default="RL", type=str)
    parser.add_argument("--max_timesteps", default=500000, type=int)
    parser.add_argument("--start_timesteps", default=10000, type=int)  # 纯随机探索的预热步数
    parser.add_argument("--expl_noise", default=0.1, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    args = parser.parse_args()

    # 初始化 CARLA 客户端
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()

    # hud = HUD(...) # 根据你的代码初始化
    hud = None

    # 初始化环境
    env = World(client, world, hud, args, visuals=False)

    # 动作与状态维度
    state_shape = env.observation_space.shape  # (128, 128, 1)
    action_dim = env.action_space.shape[0]  # 2
    max_action = float(env.action_space.high[0])  # 1.0

    # 初始化 TD3 Agent 与回放池
    agent = TD3(action_dim=action_dim, max_action=max_action)
    replay_buffer = ReplayBuffer(state_shape=state_shape, action_dim=action_dim)

    # 训练变量
    obs, _ = env.reset()
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    os.makedirs("./models", exist_ok=True)

    print("🚀 开始 TD3 训练...")

    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1

        # 1. 动作选择 (预热期使用纯随机动作积累经验)
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs)
            # 添加高斯探索噪声
            noise = np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            action = (action + noise).clip(-max_action, max_action)

        # 2. 与环境交互
        next_obs, reward, done, truncated, info = env.step(action)

        # 处理 CARLA 的最大步数截断问题 (TimeLimit)
        done_bool = float(done) if not truncated else 0.0

        # 3. 存储经验
        replay_buffer.add(obs, action, reward, next_obs, done_bool)

        obs = next_obs
        episode_reward += reward

        # 4. 训练网络
        if t >= args.start_timesteps:
            agent.train(replay_buffer, args.batch_size)

        # 5. 回合结束处理
        if done or truncated:
            print(
                f"Total T: {t + 1} | Episode: {episode_num + 1} | Steps: {episode_timesteps} | Reward: {episode_reward:.2f}")

            # 定期保存模型
            if (episode_num + 1) % 50 == 0:
                agent.save(f"./models/td3_carla_ep{episode_num + 1}")

            obs, _ = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1


if __name__ == "__main__":
    main()