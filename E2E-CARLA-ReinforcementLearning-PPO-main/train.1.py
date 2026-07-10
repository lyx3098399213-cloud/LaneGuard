import carla
import time
from Utils.utils import *
from Utils.HUD import HUD as HUD
from World1 import World
import argparse
import logging
import pygame
import os

# ================= 训练环境配置 =================
# 如果你在没有显示器的服务器上训练，请取消下面这行的注释，启用无头模式
os.environ["SDL_VIDEODRIVER"] = "dummy"

os.environ["TQDM_DISABLE"] = "0"  # 确保 tqdm 不被关掉
os.environ["DISABLE_RICH"] = "1"  # 关闭 rich，只用普通 tqdm

from stable_baselines3.common.callbacks import CallbackList
from callbacks import *
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import RecurrentPPO

# 创建一个新的日志目录，用于存放本次恢复训练的日志
logdir = f"logs/resume_{int(time.time())}/"

if not os.path.exists(logdir):
    os.makedirs(logdir)


def game_loop(args):
    # ================= 核心修复：Pygame 初始化 =================
    pygame.init()
    pygame.font.init()
    # 创建一个 Pygame 窗口表面，这是 pygame.event.get() 正常工作的前提
    try:
        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF
        )
    except pygame.error as e:
        logging.error(f"Pygame display initialization failed: {e}")
        logging.info("If running on a headless server, please enable the 'dummy' video driver.")
        return
    # ==========================================================

    world = None
    try:
        # --- 1. 初始化 CARLA 环境 ---
        client = carla.Client(args.host, args.port)
        client.set_timeout(100.0)
        hud = HUD()

        # 加载并获取世界地图
        client.load_world(args.map)
        carla_world = client.get_world()

        # 应用同步模式设置
        carla_world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=1 / args.FPS))

        # 实例化 World 并包装在 Monitor 中
        world = World(client, carla_world, hud, args)
        world = Monitor(world, logdir)
        world.reset()

        # ==============================================================================
        # --- 2. 关键修改：加载已训练的模型而不是创建新模型 ---
        # ==============================================================================

        # [重要] 请在这里填入你之前保存的 .zip 文件的完整路径
        # 例如: "logs/1775186946/rl_model_296500_steps.zip"
        model_path = "logs/resume_1779704496/rl_model_361000_steps.zip"

        # 根据文件名填写已经训练的步数
        # 例如文件名是 rl_model_296500_steps.zip，这里就填 296500
        already_trained_steps =361000  # 请修改为你的实际步数

        print(f"正在加载模型: {model_path}")

        # 加载模型（绑定新环境，指定新的 tensorboard_log）
        model = RecurrentPPO.load(
            model_path,
            env=world,
            tensorboard_log=logdir,
            device='auto'  # 自动选择 GPU/CPU
        )

        # 可选：调整超参数继续训练
        # model.learning_rate = 0.0001  # 如果想用更小的学习率微调
        # model.ent_coef = 0.005        # 降低探索系数

        # --- 3. 设置回调函数 ---
        save_callback = SaveOnBestTrainingRewardCallback(
            check_freq=500,
            log_dir=logdir,
            verbose=1
        )
        tensor = TensorboardCallback()
        checkpoint = CheckpointCallback(
            save_freq=500,
            save_path=logdir,
            verbose=1
        )

        # --- 4. 计算剩余步数并开始训练 ---
        TOTAL_GOAL = 700000  # 目标总步数
        remaining_timesteps = TOTAL_GOAL - already_trained_steps

        if remaining_timesteps <= 0:
            print(f"训练步数已达到目标 {TOTAL_GOAL}，无需继续训练。")
            print("如果你想继续训练，请增大 TOTAL_GOAL 的值。")
            return

        print(f"目标总步数: {TOTAL_GOAL}")
        print(f"已完成步数: {already_trained_steps}")
        print(f"剩余步数: {remaining_timesteps}")
        print(f"新日志目录: {logdir}")
        print("=" * 60)

        # 开始训练
        model.learn(
            total_timesteps=remaining_timesteps,
            reset_num_timesteps=False,  # [关键] False 以延续 Tensorboard 曲线
            tb_log_name=f"PPO_resume",  # 新的日志名称
            progress_bar=True,
            callback=CallbackList([tensor, save_callback, checkpoint])
        )

        # 训练结束后保存最终模型
        final_model_path = f"{logdir}/final_model_{TOTAL_GOAL}_steps"
        model.save(final_model_path)
        print(f"训练完成！最终模型已保存至: {final_model_path}")

    except Exception as e:
        logging.error(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 清理资源
        if world is not None:
            world.destroy()
        pygame.quit()
        logging.info("训练结束，资源已清理")


# ==============================================================================
# -- main() (保持不变) ----------------------------------------------------------
# ==============================================================================

def main():
    argparser = argparse.ArgumentParser(
        description='CARLA RL Training Script - Resume Training')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '--map',
        metavar='NAME',
        default='Town04',
        help='simulation map (default: "Town04")')
    argparser.add_argument(
        '--spawn_x',
        metavar='x',
        default='-16.75',
        help='x position to spawn the agent')
    argparser.add_argument(
        '--spawn_y',
        metavar='y',
        default='-223.55',
        help='y position to spawn the agent')
    argparser.add_argument(
        '--random_spawn',
        metavar='RS',
        default='0',
        type=int,
        help='Random spawn agent')
    argparser.add_argument(
        '--vehicle_id',
        metavar='NAME',
        default='vehicle.tesla.model3',
        help='vehicle to spawn')
    argparser.add_argument(
        '--vehicle_wheelbase',
        metavar='NAME',
        type=float,
        default='2.89',
        help='vehicle wheelbase used for model predict control')
    argparser.add_argument(
        '--waypoint_resolution',
        metavar='WR',
        default='1',
        type=float,
        help='waypoint resulution for control')
    argparser.add_argument(
        '--waypoint_lookahead_distance',
        metavar='WLD',
        default='5.0',
        type=float,
        help='waypoint look ahead distance for control')
    argparser.add_argument(
        '--desired_speed',
        metavar='SPEED',
        default='30',
        type=float,
        help='desired speed for highway driving')
    argparser.add_argument(
        '--control_mode',
        metavar='CONT',
        default='PID',
        help='Controller')
    argparser.add_argument(
        '--planning_horizon',
        metavar='HORIZON',
        type=int,
        default='5',
        help='Planning horizon for MPC')
    argparser.add_argument(
        '--time_step',
        metavar='DT',
        default='0.15',
        type=float,
        help='Planning time step for MPC')
    argparser.add_argument(
        '--FPS',
        metavar='FPS',
        default='20',
        type=int,
        help='Frame per second for simulation')

    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        game_loop(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()