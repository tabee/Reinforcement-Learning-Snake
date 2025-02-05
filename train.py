import torch.optim as optim
from snake_env import SnakeEnv
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.dqn.policies import DQNPolicy

class ScoreLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_scores = []  # Sammlung der Score-Werte

    def _on_step(self) -> bool:
        # In VecEnvs liegt in self.locals["infos"] eine Liste von info-Dicts vor.
        infos = self.locals.get("infos", [])
        for info in infos:
            if "score" in info:
                self.episode_scores.append(info["score"])
        return True

    def _on_rollout_end(self) -> None:
        if self.episode_scores:
            avg_score = sum(self.episode_scores) / len(self.episode_scores)
            # TensorBoard-Log über den internen Logger
            self.logger.record("rollout/avg_score", avg_score)
            self.episode_scores = []

def train_dqn(total_timesteps=10_000_000):
    env = SnakeEnv()
    check_env(env, warn=True)
    model = DQN(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.001,
        tensorboard_log="./tensorboard/",
        exploration_fraction=0.2,  # 20% der total_timesteps für Exploration
        exploration_final_eps=0.05  # Endwert für epsilon
    )
    score_callback = ScoreLoggingCallback(verbose=1)
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=score_callback)
    model.save("dqn_snake")
    return model

def train_ppo(total_timesteps=10_000_000):
    env = SnakeEnv()
    check_env(env, warn=True)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,   # Geringere Lernrate für stabileres Training
        n_steps=1024,           # Anzahl Schritte pro Update, passend für kurze Episoden
        batch_size=64,          # Mini-Batch-Größe
        n_epochs=10,            # Mehrfache Updates pro Rollout
        gamma=0.99,             # Diskontierungsfaktor
        gae_lambda=0.95,        # Vorteilsschätzung
        clip_range=0.2,         # Clipping der Policy-Updates
        ent_coef=0.01,          # Entropie-Koeffizient zur Förderung der Exploration
        vf_coef=0.5,            # Gewichtung der Wertfunktion
        max_grad_norm=0.5,      # Gradient Clipping
        tensorboard_log="./tensorboard/"
    )
    score_callback = ScoreLoggingCallback(verbose=1)
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=score_callback)
    model.save("ppo_snake")
    return model

def test(model):
    ''' Testet das trainierte Modell '''
    env = SnakeEnv()
    model = model
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        env.render()
        print("Reward:", reward, "Total Score:", env.score)

    print("Episode beendet. Total Reward:", total_reward)

if __name__ == "__main__":
    # Beispiel: Training und Test von DQN
    dqn_model = train_dqn(total_timesteps=10_000_000)
    test(dqn_model)
    
    # Beispiel: Training und Test von PPO
    ppo_model = train_ppo(total_timesteps=10_000_000)
    test(ppo_model)
