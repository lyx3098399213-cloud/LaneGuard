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
