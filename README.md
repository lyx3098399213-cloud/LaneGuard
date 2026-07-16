# 🚗LaneGuard
LaneGuard: A Spatiotemporal-Aware End-to-End Autonomous Evasive Steering And Emergency Braking Framework
LaneGuard is a deep reinforcement learning project built on the CARLA simulator, designed to enable autonomous driving in complex environments. The project focuses on end-to-end decision-making by capturing critical temporal dependencies in driving scenarios. It leverages time-series sensor data and ego-vehicle states as the primary observation inputs, and trains an agent using a custom RecurrentPPO implementation based on Stable Baselines3. The core perception and feature extraction module utilizes a unique LSTM-FCN architecture: the Fully Convolutional Network (FCN) branch operates serially, with SENet modules integrated directly inside each convolution layer to dynamically recalibrate channel-wise features. This architecture allows the agent to learn robust, adaptive driving strategies that effectively balance safety and efficiency.
![Figure 1: Overview of the LaneGuard framework.](https://raw.githubusercontent.com/lyx3098399213-cloud/LaneGuard/main/framwork.png)
# 1. Clone the repository
git clone https://github.com/yourusername/R-PPO.git
cd R-PPO

# 2. Create environment
conda env create -f environment.yml
conda activate R-PPO

# 3. Train the agent
python train.py --agent r-ppo

# 4. Evaluate the agent
python eval.py --checkpoint model/r_ppo/final_model.pth --output results/

✨ Project Characteristics

🚗 Task Type: Autonomous driving and navigation focused on the Town04 map, handling continuous dynamic environments and complex routing.

🎯 Training Objective: To achieve safe, smooth, and efficient driving control by jointly optimizing collision avoidance, route tracking, and driving stability through carefully tuned reward functions.

📷 Observation Inputs: Time-series sensor data and vehicle state (speed, position, etc.) are processed using a custom LSTM-FCN architecture. The FCN branch processes spatial features serially, with SENet modules embedded within each convolutional layer rather than concatenated at the end.

🧠 Default Algorithm: R-PPO (Recurrent Proximal Policy Optimization) built upon Stable Baselines3, which effectively models the problem as a partially observable Markov decision process using recurrent memory.

🔄 Optional Algorithms: Standard PPO, SAC, and DDPG are provided as baselines to facilitate ablation experiments and comparative studies.

⚙️ Automation: The training script automatically starts the CARLA server, eliminating the need to manually launch the simulator in advance.

📊 Evaluation Output: Supports TensorBoard training curve logs, detailed per-episode statistics (in CSV format), and feature visualization, facilitating analysis of model performance and training stability.

Repository Structure

Plaintext
R-PPO/
├── README.md
├── environment.yml
├── train.py
├── eval.py
├── default.yaml
├── r_ppo_agent.py
├── models/
│   └── lstm_fcn.py
├── carla_env.py
└── traffic_manager.py
The roles of each file are as follows:

train.py: The training entry point, responsible for argument parsing, environment instantiation, the main training loop, periodic evaluation, and model checkpoint saving.

eval.py: The offline evaluation entry point, which loads the experiment configuration and model weights from the log directory, and reports comprehensive evaluation metrics.

r_ppo_agent.py: The core agent implementation, integrating the RecurrentPPO algorithm from Stable Baselines3 with the custom policy networks.

models/lstm_fcn.py: Contains the implementation of the custom feature extractor, featuring the serial FCN branch with embedded SENet modules and the parallel LSTM sequence processor.

carla_env.py: The CARLA environment wrapper, defining the state and action spaces, custom reward computation, environment step methods, and termination conditions.

traffic_manager.py: Responsible for spawning, controlling, and cleaning up background vehicles to create complex multi-vehicle interaction scenarios.

default.yaml: The global hyperparameter configuration file, covering perception settings, Stable Baselines3 parameters, reward weights, and other training-related options.

environment.yml: The Conda environment specification, listing all dependencies including PyTorch, Stable Baselines3, Gymnasium, TensorBoard, and the CARLA Python API.

README.md: The project documentation, providing an overview, key features, installation instructions, training and evaluation guides.

Environment Setup

1. Basic Requirements

Operating System: Windows or Linux

GPU: Recommended to use NVIDIA GPU

Python: python=3.8 (or compatible version for Stable Baselines3)

CARLA: The current configuration uses Town04 as the default scenario.

2. Create Conda Environment
The project dependencies are already specified in environment.yml. It is recommended to create an isolated environment directly:

Bash
conda env create -f environment.yml
conda activate R-PPO
The core dependencies included in the environment file are:

stable-baselines3[extra]

pytorch

gymnasium

tensorboard

pyyaml

3. Install the CARLA Python API
First install and extract CARLA locally, then register the corresponding Python API into the current Conda environment. A common approach is as follows:

Bash
conda activate R-PPO
conda install -y conda-build
conda develop path/to/CARLA/PythonAPI/carla/dist/carla-<your_version>.egg
4. Setup CARLA_ROOT
⚠️ Important: The project relies on CARLA_ROOT to automatically start the simulator. Training cannot start properly if this environment variable is not set.

Example for Windows PowerShell:

PowerShell
$env:CARLA_ROOT="D:\CARLA_0.9.12"
Example for Linux Bash:

Bash
export CARLA_ROOT="/path/to/CARLA_0.9.12"
Task Description

This project is specifically designed for autonomous driving navigation using recurrent memory to handle complex, dynamic driving environments.

Map: Town04 by default.

Task: The ego vehicle navigates through the map, tracking designated routes while avoiding dynamic obstacles.

Traffic: Dynamic background vehicles are deployed via the Traffic Manager to create complex multi-vehicle interactions.

Control: Reinforcement learning outputs continuous control commands (target speed and steering angle); a low-level PID controller converts target speed to throttle/brake.

Termination: An episode terminates upon collision, severe route deviation, exceeding the maximum step limit, or successfully reaching the destination.

Key metrics: Success rate, collision rate, average speed, and completion time.

Rewards and Safety Objectives

The current reward design focuses on safe and efficient driving. The goal is not simply to maximize speed, but to teach the agent to balance safety and efficiency simultaneously using its recurrent memory of past states.

The main factors considered include:

Whether the ego vehicle maintains a safe distance from surrounding vehicles.

Whether the vehicle makes steady progress along the designated route.

Whether the control actions are smooth (avoiding abrupt throttle/brake/steering).

Penalties for lane invasions or collisions.

During training and evaluation, safety-related statistics are recorded, such as:

Success Rate (%)

Collision Rate (%)

Average Speed (m/s)

Completion Time (s)

Episode Reward

Failure Type (0=running, 1=collision, 2=timeout, 3=success)

Training

1. Train the Task
Run the following command in the project root directory:

Bash
conda activate R-PPO

# Train default task
python train.py --agent r-ppo
2. Common Arguments
The training script currently supports many arguments. The most important ones include:

Environment: --host, --port, --town, --fps

Network Architecture: --lstm_hidden_size, --fcn_channels

Decision (R-PPO): --lr, --gamma, --gae_lambda, --clip_range, --n_steps

Reward weights: --reward_speed, --penalty_collision, --penalty_steer

Training hyperparameters: --total_timesteps, --batch_size, --n_epochs

Runtime controls: --save_path, --checkpoint_freq, --tensorboard_log

3. Training Outputs
Each training run generates outputs under predefined directories for models and logs. The output structure is as follows:

Plaintext
model/
└── r_ppo/
    ├── checkpoint_*.pth
    └── final_model.pth
runs/
└── r_ppo/
    └── <run_id>/
model/: Stores all trained network checkpoints, including the LSTM-FCN feature extractor and the actor-critic policy weights.

runs/: Saves TensorBoard log files for training metric visualization.

Evaluation

1. Basic Evaluation Command

Bash
python eval.py --checkpoint model/r_ppo/final_model.pth --output results/
Notes:

--checkpoint: Path to the model checkpoint to evaluate.

--output: Directory to save evaluation results.

The evaluation script automatically:

Loads the trained model weights from the specified checkpoint.

Reconstructs the environment using the same Town04 configuration as training.

Runs episodes to compile reliable statistics.

2. Evaluation Outputs
The evaluation script generates the following contents:

Plaintext
results/
├── summary.csv              
└── detailed/               
    └── episode_*.csv
During evaluation, the following metrics are computed and printed:

Success Rate (%): Percentage of successfully completed episodes

Collision Rate (%): Percentage of episodes with collision

Average Speed (m/s): Mean speed over all successful episodes

Completion Time (s): Mean time to reach destination

Episode Reward: Cumulative reward per episode

Failure Type: 0=running, 1=collision, 2=timeout, 3=stuck, 4=success

Notes for Reproducibility

The environment step methods and reward functions have been specifically tuned for Town04. It is recommended to verify sensor spawn parameters if testing on other maps.

The project relies on CARLA_ROOT to automatically start the simulator. Training cannot start properly if this environment variable is not set.

The network heavily relies on the custom LSTM-FCN structure where SENet is deeply integrated into the serial convolution blocks. Ensure your PyTorch environment supports the required operations for these recalibration blocks.
