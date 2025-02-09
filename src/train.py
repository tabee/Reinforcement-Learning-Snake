import argparse
import toml
from datetime import datetime
import torch.optim as optim
from snake_env import SnakeEnv
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import CallbackList, BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.dqn.policies import DQNPolicy

class ScoreLoggingCallback(BaseCallback):
    """
    Callback for logging the scores of episodes during training.

    Attributes:
        episode_scores (list): A list to store the scores of each episode.

    Methods:
        _on_step() -> bool:
            Called at each step of the environment. Collects the score from the
            'infos' dictionary if available and appends it to the episode_scores list.
        
        _on_rollout_end() -> None:
            Called at the end of each rollout. Calculates the average score of the
            collected episode scores and logs it using the internal logger. Resets
            the episode_scores list afterwards.
    """
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_scores = []  # List to store the scores of each episode

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "score" in info:
                self.episode_scores.append(info["score"])
        return True


    def _on_rollout_end(self) -> None:
        if self.episode_scores:
            avg_score = sum(self.episode_scores) / len(self.episode_scores)
            self.logger.record("rollout/avg_score", avg_score)
            self.episode_scores = []

def clean_toml_config(config):
    """
    Entfernt Schlüssel aus der Konfiguration, die nicht als Parameter für PPO zulässig sind.
    """
    excluded_keys = ["name"]
    return {k: v for k, v in config.items() if k not in excluded_keys}
def get_config_by_name(config_name=None):
    """
    Wählt eine Konfiguration basierend auf dem Namen aus der TOML-Datei aus.
    Falls kein Name angegeben wird, wird die erste Konfiguration verwendet.
    """
    if config_name:
        for cfg in data["configs"]:
            if cfg.get("name") == config_name:
                return cfg
        raise ValueError(f"Konfiguration '{config_name}' nicht gefunden!")
    
    return data["configs"][0]  # Fallback auf erste Konfiguration
# TOML-Konfiguration lesen
with open("./src/ppo_configs.toml", "r") as f:
    data = toml.load(f)
# TOML-Konfiguration laden
config_name = "config2"  # Hier den gewünschten Namen setzen oder als Argument übergeben
config = get_config_by_name(config_name)
print("\nUsing configuration:", config)


env = SnakeEnv() # Setups the environment
env_monitor = Monitor(env) # Monitor the environment

# Callback to save checkpoints during training
checkpoint_callback = CheckpointCallback(
    save_freq=5000,  # save a checkpoint every 5k steps
    save_path="./models/checkpoints_"+config.get("name")+"/",
    name_prefix="ppo_snake",
    verbose=1,
)
# Callback to evaluate the model during training
eval_callback = EvalCallback(
    env_monitor,
    best_model_save_path="./models/best_model_"+config.get("name")+"/",
    log_path="./logs/",
    eval_freq=10_000,  # Evaluate the model every 10k steps
    n_eval_episodes=3,  # Evaluate the model on 3 episodes
    deterministic=False,  # False for stochastic when less computation power
    render=False
)
# Callback to log the scores of episodes during training
score_callback = ScoreLoggingCallback(verbose=1)

# Callback list to combine all callbacks
callbacks = CallbackList([score_callback, checkpoint_callback, eval_callback])

def train_ppo(total_timesteps, model=None):
    """
    Train a Proximal Policy Optimization (PPO) model for the Snake environment.
    Parameters:
    total_timesteps (int): The total number of timesteps to train the model.
    model (PPO, optional): An existing PPO model to continue training. If None, a new model will be created.
    Returns:
    PPO: The trained PPO model.
    """
    if model is None:
        print("Train new model.")
    else:
        print("Continue training existing model.")
        
    check_env(env, warn=True)
    if model is None:
        ppo_config = clean_toml_config(config)
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/", **ppo_config)
    else:
        model.set_env(env)
        model.verbose = 1
        model.tensorboard_log = "./tensorboard/"
    
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=callbacks)
    model.save("./models/ppo_snake_"+config.get("name"))
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PPO model for Snake game.')
    parser.add_argument('--load', type=str, default=None, help='Path to the model to load')
    parser.add_argument('--timesteps', type=int, default=10_000, help='Number of timesteps to train the model')
    
    args = parser.parse_args()
    
    ppo_model = None
    
    if args.load != None:
        print("lodaded model: ", args.load)
        ppo_model = PPO.load("./models/" + args.load)
        
    if args.timesteps:
        print("timesteps: ", args.timesteps)
        ppo_model = train_ppo(total_timesteps=args.timesteps, model=ppo_model)
