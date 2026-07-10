import pygame
import carla
from Utils.synch_mode import CarlaSyncMode
import Controller.PIDController as PIDController
import time
from Utils.utils import *
import math
import gymnasium as gym
from gymnasium import spaces
from Utils.HUD import HUD as HUD
import csv
import numpy as np
import random
import cv2


class World(gym.Env):
    def __init__(self, client, carla_world, hud, args, visuals=False):
        self.world = carla_world
        self.client = client
        self.map = self.world.get_map()
        self.hud = hud
        self.args = args
        self.waypoint_resolution = args.waypoint_resolution
        self.waypoint_lookahead_distance = args.waypoint_lookahead_distance
        self.desired_speed = args.desired_speed
        self.control_mode = args.control_mode
        self.controller = None
        self.control_count = 0.0
        self.random_spawn = 0
        self.world.on_tick(hud.on_world_tick)
        self.im_width = 640
        self.im_height = 480
        self.episode_start = 0
        self.visuals = visuals
        self.episode_reward = 0
        self.reward = 0.0
        self.player = None
        self.parked_vehicle = None
        self.slow_left_vehicle = None          # 左侧慢车
        self.collision_sensor = None
        self.camera_rgb = None
        self.lane_invasion = None
        self._autopilot_enabled = False
        self._control = carla.VehicleControl()
        self.max_dist = 4.5
        self.counter = 0
        self.frame = None
        self.delta_seconds = 1.0 / args.FPS
        self.last_v = 0
        self.last_x = 0                        # 前进方向记录
        self.last_y = 0                        # 横向坐标记录
        self.distance_parked = 100
        self.ttc_trigger = 1.0
        self.episode_counter = 0
        self.steer = 0.0
        self.filtered_steer = 0.0
        self.file_name = getattr(args, "file_name", "training_log.csv")
        self.save_list = []
        self.logger = False
        self.lane_change_buffer = 0
        self.prev_steer = 0.0

        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype="float")
        self.observation_space = spaces.Box(low=-0, high=255, shape=(128, 128, 1), dtype=np.uint8)

        self.global_t = 0
        self.max_steps = 1500
        self.weather_type = "Sunny"
        self.max_decel = 8.0

    # ================= 天气与物理 =================
    def set_weather_and_physics(self):
        weather_presets = {
            "Sunny": (carla.WeatherParameters.ClearNoon, 3.5, 8.0),
            "Rainy": (carla.WeatherParameters.HardRainNoon, 1.5, 4.5),
            "Cloudy": (carla.WeatherParameters.CloudySunset, 3.0, 7.0),
            "Wet": (carla.WeatherParameters.WetNoon, 2.0, 5.5)
        }
        self.weather_type = random.choice(list(weather_presets.keys()))
        weather_param, tire_friction, self.max_decel = weather_presets[self.weather_type]
        self.world.set_weather(weather_param)

        if self.player is not None:
            physics_control = self.player.get_physics_control()
            for wheel in physics_control.wheels:
                wheel.tire_friction = tire_friction
            self.player.apply_physics_control(physics_control)
        else:
            raise RuntimeError("Player vehicle is None.")

    def _update_spectator(self):
        """强制将模拟器视角锁定在障碍车上方"""
        if self.parked_vehicle is not None:
            spectator = self.world.get_spectator()
            # 获取障碍车的实时位置
            target_transform = self.parked_vehicle.get_transform()
            # 俯视视角：高度 30 米，垂直向下看
            # 你可以根据需要调整 Location(z=...) 的高度
            spectator.set_transform(carla.Transform(
                target_transform.location + carla.Location(z=30.0),
                carla.Rotation(pitch=-90, yaw=0, roll=0)
            ))
    # ================= 图像处理 =================
    def process_image(self, image_rgb):
        if image_rgb is None:
            return np.zeros(self.observation_space.shape, dtype=np.uint8)
        array = np.frombuffer(image_rgb.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image_rgb.height, image_rgb.width, 4))
        array = array[:, :, :3]
        gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (128, 128))
        return np.expand_dims(resized, axis=-1)

    # ================= 重置环境 =================
    def reset(self, seed=None):
        self.destroy()
        if self.counter == 1:
            self.lane_achieved = False
        # 1. 设置同步模式
        self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=1 / self.args.FPS))

        self.episode_reward = 0
        self.desired_speed = self.args.desired_speed
        self.episode_counter += 1
        self.counter = 0
        self.filtered_steer = 0.0
        self.lane_change_buffer = 0
        self.prev_steer = 0.0

        if self.logger and len(self.save_list) > 0:
            self.append_to_csv(file_name=self.file_name, data=self.save_list)
        self.save_list = []

        # 2. 创建 Actors (主车、障碍车等)
        self.create_actors()
        self.set_weather_and_physics()

        # 记录初始车道 ID
        self.start_lane_id = self.map.get_waypoint(self.player.get_location()).lane_id

        # 3. 初始视角设置：立刻锁定障碍物
        self._update_spectator()

        # 初始物理状态
        v_vec = self.player.get_velocity()
        t = self.player.get_transform()
        c_speed = math.sqrt(v_vec.x ** 2 + v_vec.y ** 2 + v_vec.z ** 2)
        frame, ts = self.hud.get_simulation_information()

        if self.controller is not None:
            self.controller.update_values(t.location.x, t.location.y, wrap_angle(t.rotation.yaw), c_speed, ts, frame)

        self.episode_start = time.time()
        self.clock = pygame.time.Clock()

        # 4. 传感器热身：使用 synch_mode 推进 10 帧，防止 Frame mismatch
        # 这一步非常重要，它让摄像头和碰撞传感器完成初始化并同步序列号
        for _ in range(10):
            self.synch_mode.tick(timeout=2.0)
            self._update_spectator()  # 保持视角跟随

        self.last_x = self.player.get_location().x
        self.last_y = self.player.get_location().y

        # 5. 预跑阶段：直到 TTC 进入触发范围
        ttc = self.time_to_collision()
        img = np.zeros(self.observation_space.shape, dtype=np.uint8)
        current_ttc_trigger = random.uniform(0.8, 2.5)

        while ttc > current_ttc_trigger:
            self.clock.tick_busy_loop(self.args.FPS)
            if self.parse_events(clock=self.clock, action=None):
                return img, {}

            # 慢车控制
            if self.slow_left_vehicle is not None:
                try:
                    self.slow_left_vehicle.apply_control(
                        carla.VehicleControl(throttle=0.8, steer=0.0, hand_brake=False))
                except:
                    pass

            # 🌟 核心：使用同步模式的 tick，同时更新上帝视角
            # synch_mode.tick 会自动调用 world.tick()，不要在外面额外调用了
            snapshot, image_rgb, lane, collision = self.synch_mode.tick(timeout=5.0)

            self._update_spectator()  # 持续锁定障碍物视角

            if image_rgb is not None:
                img = self.process_image(image_rgb)

            # 更新位置与 TTC
            v_loc = self.player.get_location()
            self.last_x = v_loc.x
            self.last_y = v_loc.y
            ttc = self.time_to_collision()

        return img, {}


    def tick(self, clock):
        self.hud.tick(self, clock)

    # ================= 环境步进 =================
    # ================= 环境步进 =================
    def step(self, action):
        self.reward = 0.0
        done = False
        truncated = False
        image = np.zeros(self.observation_space.shape, dtype=np.uint8)


        if self.slow_left_vehicle is not None:
            try:
                self.slow_left_vehicle.apply_control(
                    carla.VehicleControl(throttle=0.8, steer=0.0, hand_brake=False))
            except:
                pass

        if action is not None:
            self.counter += 1
            self.global_t += 1

            # 步数限制
            if self.counter >= self.max_steps:
                return getattr(self, "last_obs", image), 0.0, False, True, {}

            self.clock.tick_busy_loop(self.args.FPS)

            # 应用动作
            if self.apply_vehicle_control(action):
                return image, 0.0, False, False, {}

            # 2. 推进物理世界帧
            snapshot, image_rgb, lane_data, col_data = self.synch_mode.tick(timeout=10.0)

            # 🌟 关键补丁：每帧更新视角，锁定障碍物车
            self._update_spectator()

            if snapshot is None:
                return image, 0.0, False, True, {}

            if image_rgb is not None:
                image = self.process_image(image_rgb)
            self.last_obs = image

            # 获取当前状态
            vehicle_location = self.player.get_location()
            current_waypoint = self.map.get_waypoint(vehicle_location)
            vel = self.player.get_velocity()
            current_speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

            # 障碍物距离计算（仅 X 轴纵向距离）
            if self.parked_vehicle is not None:
                p_loc = self.parked_vehicle.get_location()
                dist_to_obs = p_loc.x - vehicle_location.x
            else:
                dist_to_obs = 100.0

            # 获取奖励组件
            cos_yaw, lateral_dist, col_flag, lane_flag, traveled, cur_lane_id = self.get_reward_comp(
                self.player, current_waypoint, col_data, lane_data
            )

            # 静止检测逻辑
            if not hasattr(self, 'stuck_counter'): self.stuck_counter = 0
            if current_speed < 0.2:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0

            # 计算奖励
            self.reward = self.reward_value(
                cos_yaw, lateral_dist, col_flag, lane_flag, traveled,
                dist_to_obs=dist_to_obs,
                current_speed=current_speed,
                steer=self.steer,
                current_waypoint=current_waypoint
            )
            self.episode_reward += self.reward

            # ---- 终止条件与保护期处理 ----

            # A. 碰撞终止
            if col_flag == 1:
                done = True
                self.reward -= 150
                print("Collision! Episode Terminated.")

            # B. 压线处理（修复无限重置 Bug）
            is_in_second_lane = hasattr(self, 'start_lane_id') and cur_lane_id != self.start_lane_id

            if lane_flag == 1:
                if is_in_second_lane:
                    if self.lane_change_buffer > 0:
                        self.reward -= 30
                        self.lane_change_buffer -= 1
                        print(f"Lane touch during buffer, remaining: {self.lane_change_buffer}")
                    else:
                        done = True
                        self.reward -= 100
                        print(f"Second Lane Invasion at Lane {cur_lane_id}! Hard Termination.")
                else:
                    self.reward -= 20
                    print("Crossing lane markings on original lane...")
            else:
                # 正常行驶时也要消耗 buffer
                if self.lane_change_buffer > 0:
                    self.lane_change_buffer -= 1

            # C. 🌟 启动保护期（仅在首次变道时触发一次）
            if not hasattr(self, 'buffer_triggered'): self.buffer_triggered = False

            if is_in_second_lane:
                if not self.buffer_triggered:
                    self.lane_change_buffer = 20
                    self.buffer_triggered = True  # 标记已触发，防止无限设为 20
                    print("Lane change detected! Buffer initialized.")
            else:
                # 如果回到了原车道，可以重置触发标记（可选）
                self.buffer_triggered = False

            # D. 偏离赛道终止
            if lateral_dist > 4.0:
                done = True
                self.reward -= 100
                print(f"Off track: {lateral_dist:.2f}m")

            # E. 卡死终止
            if self.stuck_counter > 120:
                done = True
                self.reward -= 100
                print("Vehicle Stuck! Episode Terminated.")

            # F. 目标达成
            if vehicle_location.x > float(self.args.spawn_x) + self.distance_parked + 15:
                done = True
                self.reward += 1000
                print("Goal reached! Mission Success.")

            # 更新历史记录
            self.last_x = vehicle_location.x
            self.last_y = vehicle_location.y
            self.last_v = current_speed

        info = {
            "x": vehicle_location.x,
            "y": vehicle_location.y,
            "speed": current_speed,
            "steer": self.steer,
            "lateral_dist": lateral_dist,
            "dist_to_obs": dist_to_obs,
            "ttc": self.time_to_collision()
        }

        return image, self.reward, done, truncated, info

    def get_reward_comp(self, vehicle, waypoint, collision, lane):
        v_loc = vehicle.get_location()
        w_loc = waypoint.transform.location

        # 1. 横向偏移：使用 Y 轴差值
        dist = abs(v_loc.y - w_loc.y)

        # 2. 航向差
        vh_yaw = correct_yaw(vehicle.get_transform().rotation.yaw)
        wp_yaw = correct_yaw(waypoint.transform.rotation.yaw)
        cos_yaw_diff = math.cos((vh_yaw - wp_yaw) * math.pi / 180.)

        # 3. 碰撞判定
        collision_flag = 0 if collision is None else 1

        # 4. 压线判定 (只要压了任何线就标记为 1)
        lane_flag = 1 if (lane is not None and lane.crossed_lane_markings) else 0

        # 🌟 5. 获取当前路点的车道 ID（这是缺失的第 6 个参数）
        current_lane_id = waypoint.lane_id

        # 6. 前进距离：X 轴增量
        traveled = v_loc.x - self.last_x

        # 必须返回 6 个值，顺序要和 step 接收的对应
        return cos_yaw_diff, dist, collision_flag, lane_flag, traveled, current_lane_id

    # ================= 奖励函数（保持早期简单版本） =================
    def reward_value(self, cos_yaw_diff, dist, collision, lane, traveled, dist_to_obs, current_speed, steer,
                     current_waypoint):
        """
        针对 Town04 超车任务优化的多目标奖励函数
        """
        # --- 1. 基础生存与进度奖励 ---
        reward = 0.2  # 基础时间步惩罚/生存奖

        # 状态判定：是否进入第二车道
        is_in_second_lane = False
        if hasattr(self, 'start_lane_id') and current_waypoint.lane_id != self.start_lane_id:
            is_in_second_lane = True

        # 进度乘子：在左侧车道行驶时，进度奖励权重更高 (1.2倍)，引导其留在左侧
        lane_multiplier = 1.2 if is_in_second_lane else 1.0
        r_progress = 2.0 * traveled * lane_multiplier

        # 速度权重因子：防止静止状态下刷分
        speed_factor = np.clip(current_speed / 2.0, 0.0, 1.0)

        # --- 2. 动态横向与航向约束 (Bug 5 修复) ---
        if is_in_second_lane:
            # 变道后的阶段
            if hasattr(self, 'lane_change_buffer') and self.lane_change_buffer > 15:
                # 变道刚刚发生的瞬间（前5帧）：显著降低横向约束，允许车辆物理跨线
                r_lateral = -1.0 * (dist ** 2) - 1.0 * abs(dist)
            else:
                # 变道稳定期：引入线性项解决微小漂移，强制锁定车道中心
                r_lateral = -5.0 * (dist ** 2) - 3.0 * abs(dist)
            # 提高变道后的航向对齐奖励
            r_yaw = 10.0 * (cos_yaw_diff - 1.0) * speed_factor
        else:
            # 原车道行驶：常规约束
            r_lateral = -2.0 * (dist ** 2) - 2.0 * abs(dist)
            r_yaw = 2.0 * (cos_yaw_diff - 1.0)

        # --- 3. 避障与变道方向引导 (引导左转是最好的) ---
        r_obs = 0.0
        r_direction = 0.0

        if is_in_second_lane:
            # 变道后：引导方向盘回正（奖励 abs(steer) 接近 0 的状态）
            r_obs = 8.0 * (1.0 - abs(steer)) * speed_factor
            # 惩罚不稳定的过度修正
            r_direction = -2.0 * abs(steer)
        else:
            # 变道前：探测到前方障碍物
            if 5.0 < dist_to_obs < 35.0:
                # 构建动态势场因子：距离越近，引导强度越大
                dist_factor = np.clip((35.0 - dist_to_obs) / 30.0, 0, 1)

                if steer < 0:
                    # 引导奖励：向左打方向盘 (steer < 0) 获得正分
                    r_direction = abs(steer) * 12.0 * dist_factor
                else:
                    # 负面惩罚：向右打方向盘或不作为扣分，迫使模型意识到左转是唯一出路
                    r_direction = -steer * 20.0 * dist_factor

                # “铁头娃”惩罚：如果距离很近了还不打方向 (abs(steer) < 0.05)，重罚
                if abs(steer) < 0.05:
                    r_obs = -15.0 * dist_factor
            else:
                # 正常行驶区间，抑制无意义的晃动
                if abs(steer) > 0.15:
                    r_obs = -3.0 * abs(steer)

        # --- 4. 变道成就奖励 (单次触发) ---
        r_bonus = 0.0
        if not hasattr(self, 'lane_achieved'): self.lane_achieved = False

        if is_in_second_lane and not self.lane_achieved:
            r_bonus = 80.0  # 给一个巨大的瞬间诱惑，鼓励跨越车道线
            self.lane_achieved = True
            print("RL Reward: Target Lane Achieved Bonus! +80")

        # --- 5. 动力学平滑惩罚 (抑制画龙) ---
        steer_diff = abs(steer - self.prev_steer)
        if steer_diff > 0.1:
            # 突变惩罚：对快速打方向的操作重罚
            r_smooth = -20.0 * steer_diff
        else:
            r_smooth = -2.0 * steer_diff

        # 更新历史记录供下一帧使用
        self.prev_steer = steer

        # --- 6. 静止惩罚 ---
        r_static = -10.0 if (self.counter > 60 and current_speed < 0.2) else 0.0

        # --- 总计 ---
        total = reward + r_progress + r_lateral + r_yaw + r_obs + r_direction + r_smooth + r_static + r_bonus

        return total

    # ================= TTC 计算 =================
    def time_to_collision(self):
        if self.player is None or self.parked_vehicle is None:
            return float('inf')
        v_vec = self.player.get_velocity()
        p_vec = self.parked_vehicle.get_velocity()
        p_loc = self.parked_vehicle.get_location()
        e_loc = self.player.get_location()
        rel_speed = math.sqrt(v_vec.x**2 + v_vec.y**2) - math.sqrt(p_vec.x**2 + p_vec.y**2)
        dist = math.sqrt((p_loc.x - e_loc.x)**2 + (p_loc.y - e_loc.y)**2)
        if rel_speed <= 0.001:
            return float('inf')
        return dist / rel_speed

    # ================= 动作控制（带轻微平滑） =================
    def apply_vehicle_control(self, action):
        try:
            raw_steer = float(action[0])
            raw_accel = float(action[1])
            self.filtered_steer = 0.5 * raw_steer + 0.5 * self.filtered_steer
            self.steer = self.filtered_steer

            self._control.steer = self.steer
            if raw_accel < 0:
                self._control.brake = abs(raw_accel)
                self._control.throttle = 0.0
            else:
                self._control.throttle = raw_accel
                self._control.brake = 0.0
            self.player.apply_control(self._control)
            self.control_count += 1
            return False
        except Exception as e:
            print(f"Control error: {e}")
            return True

    # ================= 事件 / PID 控制器处理 =================
    def parse_events(self, clock, action=None):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True

        if not self._autopilot_enabled and self.controller is not None:
            current_loc = self.player.get_location()
            vel_vec = self.player.get_velocity()
            current_speed = math.sqrt(vel_vec.x**2 + vel_vec.y**2 + vel_vec.z**2)
            frame, ts = self.hud.get_simulation_information()
            yaw = wrap_angle(self.player.get_transform().rotation.yaw)
            ready = self.controller.update_values(current_loc.x, current_loc.y, yaw, current_speed, ts, frame)

            if ready:
                waypoints = []
                if self.control_mode == "PID":
                    wp = self.map.get_waypoint(current_loc).next(self.waypoint_resolution)[0]
                    for _ in range(int(self.waypoint_lookahead_distance / self.waypoint_resolution)):
                        waypoints.append([wp.transform.location.x, wp.transform.location.y, self.desired_speed])
                        wp = wp.next(self.waypoint_resolution)[0]

                if action is not None:
                    # 修复坐标系：前进方向为 X，横向偏移为 Y
                    wp_rl = []
                    for i in range(1, 10):
                        wp_rl.append([current_loc.x + i * 2.0,                # X 轴前进
                                      current_loc.y + action[0] * i * 0.5,   # Y 轴横向偏移
                                      self.desired_speed])
                    self.controller.update_waypoints(wp_rl)
                else:
                    self.controller.update_waypoints(waypoints)

                self.controller.update_controls()
                self._control.throttle, self._control.steer, self._control.brake = self.controller.get_commands()
                self.player.apply_control(self._control)
                self.control_count += 1
            else:
                self._control.throttle = 0.0
                self._control.steer = 0.0
                self._control.brake = 1.0
                self.player.apply_control(self._control)

        return False

    # ================= 创建所有 Actor =================
    def create_actors(self):
        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle_blueprint = self.blueprint_library.filter('*vehicle*')
        model3_bp = self.vehicle_blueprint.filter('model3')[0]

        # ================= 1. 生成主车 (Player) =================
        spawn_loc = carla.Location(x=float(self.args.spawn_x), y=float(self.args.spawn_y), z=1.5)
        self.spawn_waypoint = self.map.get_waypoint(spawn_loc)
        spawn_trans = self.spawn_waypoint.transform
        spawn_trans.location.z = 1.5

        self.player = self.world.try_spawn_actor(model3_bp, spawn_trans)
        self.world.tick()

        if self.player is not None:
            print(f"✅ 主车生成成功! ID: {self.player.id}, 位置: {self.player.get_location()}")
            self.player.set_light_state(carla.VehicleLightState.Position)
        else:
            print("❌ 主车生成失败! 请检查 spawn_x 和 spawn_y 是否在路面上。")
            raise RuntimeError("Failed to spawn player vehicle.")

        # ================= 2. 生成传感器 =================
        cam_bp = self.blueprint_library.find('sensor.camera.rgb')
        cam_bp.set_attribute("image_size_x", str(self.im_width))
        cam_bp.set_attribute("image_size_y", str(self.im_height))

        self.camera_rgb = self.world.spawn_actor(
            cam_bp, carla.Transform(carla.Location(x=2, z=1)), attach_to=self.player)

        self.lane_invasion = self.world.spawn_actor(
            self.blueprint_library.find('sensor.other.lane_invasion'),
            carla.Transform(), attach_to=self.player)

        self.collision_sensor = self.world.spawn_actor(
            self.blueprint_library.find('sensor.other.collision'),
            carla.Transform(), attach_to=self.player)

        self.world.tick()
        self.synch_mode = CarlaSyncMode(self.world, self.camera_rgb, self.lane_invasion, self.collision_sensor)

        # ================= 3. 生成静止障碍车 (Parked Vehicle) =================
        # 🌟 改进逻辑：顺着路找 100 米，而不是单纯加 X 坐标
        ego_wp = self.map.get_waypoint(self.player.get_location())
        target_wps = ego_wp.next(self.distance_parked)  # 寻找前方 100 米的路点

        if target_wps:
            obs_transform = target_wps[0].transform
            obs_transform.location.z += 0.5  # 稍微抬高防止卡地

            self.parked_vehicle = self.world.try_spawn_actor(model3_bp, obs_transform)
            self.world.tick()

            if self.parked_vehicle is not None:
                print(f"✅ 静止障碍车生成成功! 位置: {self.parked_vehicle.get_location()}")
            else:
                print(f"⚠️ 静止障碍车生成失败! 坐标冲突或位置无效。坐标: {obs_transform.location}")
        else:
            print("❌ 无法找到前方 100 米的路点，地图可能已到尽头。")

        # ================= 4. 生成左侧慢车 (Slow Vehicle) =================
        if self.parked_vehicle is not None:
            obs_wp = self.map.get_waypoint(self.parked_vehicle.get_location())
            left_wp = obs_wp.get_left_lane()

            if left_wp is not None:
                # 在障碍车路点后面 35 米处生成
                slow_spawn_wps = left_wp.previous(35.0)
                if slow_spawn_wps:
                    slow_trans = slow_spawn_wps[0].transform
                    slow_trans.location.z += 0.5
                    self.slow_left_vehicle = self.world.try_spawn_actor(model3_bp, slow_trans)
                    self.world.tick()

                    if self.slow_left_vehicle is not None:
                        print(f"✅ 左侧慢车生成成功! 位置: {self.slow_left_vehicle.get_location()}")
                    else:
                        print("⚠️ 左侧慢车生成失败! 可能左侧车道被占用或无效。")
            else:
                print("ℹ️ 未发现左侧车道，取消生成慢车。")

        # ================= 5. 设置视角 (Spectator) =================
        spectator = self.world.get_spectator()
        if self.parked_vehicle is not None:
            # 上帝视角锁定在障碍物正上方
            target_t = self.parked_vehicle.get_transform()
            spectator.set_transform(carla.Transform(
                target_t.location + carla.Location(z=35.0),
                carla.Rotation(pitch=-90, yaw=180)
            ))
            print("🎬 视角已锁定在障碍车上方。")
        else:
            # 退而求其次，锁定主车
            target_t = self.player.get_transform()
            spectator.set_transform(carla.Transform(
                target_t.location + carla.Location(z=35.0),
                carla.Rotation(pitch=-90)
            ))
            print("🎬 障碍车缺失，视角回退到主车上方。")

        self.world.tick()
        if self.control_mode == "PID":
            self.controller = PIDController.Controller()
    # ================= 清理 =================
    def destroy(self):
        self.world.tick()
        for actor in [self.player, self.collision_sensor, self.camera_rgb,
                      self.lane_invasion, self.parked_vehicle, self.slow_left_vehicle]:
            if actor is not None:
                try:
                    actor.destroy()
                    self.world.tick()
                except:
                    pass

    # ================= 日志 =================
    def append_to_csv(self, file_name, data):
        with open(file_name, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data) if isinstance(data[0], list) else writer.writerow(data)

    def get_observation(self):
        pass

