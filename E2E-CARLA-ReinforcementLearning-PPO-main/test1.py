import carla
import pygame
import argparse
import numpy as np
import time

# 假设你使用的是 sb3_contrib 的 RecurrentPPO 进行训练和测试
from sb3_contrib import RecurrentPPO
from Utils.HUD import HUD
from your_env_file import World  # 请将 your_env_file 替换为你保存 World 类的文件名


def get_args():
    parser = argparse.ArgumentParser(description='CARLA 200-Episode Testing')
    parser.add_argument('--host', default='127.0.0.1', help='CARLA host IP')
    parser.add_argument('--port', type=int, default=2000, help='CARLA host port')
    parser.add_argument('--waypoint_resolution', type=float, default=2.0)
    parser.add_argument('--waypoint_lookahead_distance', type=float, default=20.0)
    parser.add_argument('--desired_speed', type=float, default=25.0)
    parser.add_argument('--control_mode', default='RL')
    parser.add_argument('--FPS', type=int, default=20)
    parser.add_argument('--spawn_x', type=float, default=250.0, help='Town04 Straight Road X')
    parser.add_argument('--spawn_y', type=float, default=-10.0, help='Town04 Straight Road Y')
    parser.add_argument('--file_name', default='test_log.csv')
    return parser.parse_args()


def main():
    args = get_args()

    # 1. 初始化 Pygame 和 CARLA 客户端
    pygame.init()
    pygame.font.init()
    display = pygame.display.set_mode((640, 480), pygame.HWSURFACE | pygame.DOUBLEBUF)

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    # 确保加载正确的测试地图
    world = client.load_world('Town04')
    hud = HUD(640, 480)

    # 2. 实例化环境
    env = World(client, world, hud, args, visuals=True)

    # 3. 加载训练好的模型
    # 如果你还没有模型，想测试随机动作，可以把下面的 model 替换为 dummy 控制逻辑
    model_path = "path_to_your_trained_model.zip"
    try:
        model = RecurrentPPO.load(model_path, env=None)
        print(f"成功加载模型: {model_path}")
    except Exception as e:
        print(f"模型加载失败，将使用随机动作进行测试验证。错误信息: {e}")
        model = None

    # 4. 测试指标追踪
    total_episodes = 200
    success_count = 0
    collision_count = 0
    off_track_count = 0
    episode_rewards = []

    print("\n==================================================")
    print(f"🚀 开始执行 {total_episodes} 轮自动化测试...")
    print("==================================================\n")

    for episode in range(1, total_episodes + 1):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0

        # RNN 状态初始化
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)

        while not (done or truncated):
            # 处理 Pygame 事件，允许在中途关闭窗口退出测试
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("测试被用户手动终止。")
                    env.destroy()
                    pygame.quit()
                    return

            if model is not None:
                # 带有隐藏状态的动作推断
                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=True  # 测试时使用确定性策略，不加探索噪声
                )
                episode_starts = np.zeros((1,), dtype=bool)
            else:
                # 随机动作测试 (用于验证环境本身是否能无报错运行 200 轮)
                action = env.action_space.sample()

            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward

            # 渲染 (如果你的 HUD 包含图像渲染逻辑)
            hud.render(display)
            pygame.display.flip()

        # 5. 分析单个 Episode 的终止原因
        episode_rewards.append(ep_reward)

        if env.reward <= -150:  # 你的代码里碰撞扣 150
            collision_count += 1
            status = "💥 碰撞终止"
        elif env.reward >= 1000:  # 你的代码里达标奖 1000
            success_count += 1
            status = "✅ 成功避障"
        elif env.reward <= -100 and info.get("lateral_dist", 0) > 4.0:
            off_track_count += 1
            status = "⚠️ 偏离赛道"
        else:
            status = "🛑 其他原因终止 (超时/卡死/压线)"

        print(f"Episode: {episode:03d}/{total_episodes} | Status: {status} | Reward: {ep_reward:.2f} | Info: {info}")

    # 6. 测试报告汇总
    print("\n==================================================")
    print("📊 200 轮测试结果汇总")
    print("==================================================")
    print(f"总测试轮次: {total_episodes}")
    print(f"成功避让次数: {success_count} (成功率: {(success_count / total_episodes) * 100:.1f}%)")
    print(f"碰撞失败次数: {collision_count} (碰撞率: {(collision_count / total_episodes) * 100:.1f}%)")
    print(f"偏离赛道次数: {off_track_count}")
    print(f"平均奖励得分: {np.mean(episode_rewards):.2f}")
    print("==================================================\n")

    # 清理环境
    env.destroy()
    pygame.quit()


if __name__ == '__main__':
    main()