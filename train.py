from snake_env import SnakeEnv
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback

class ScoreLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(ScoreLoggingCallback, self).__init__(verbose)
        self.episode_scores = []  # Liste, um Score-Werte während eines Rollouts zu sammeln

    def _on_step(self) -> bool:
        # In VecEnvs wird in self.locals["infos"] ein Liste von info-Dictionaries bereitgestellt.
        infos = self.locals.get("infos", [])
        for info in infos:
            # Falls der Score im Info-Dictionary vorhanden ist, speichern wir ihn
            if "score" in info:
                self.episode_scores.append(info["score"])
        return True

    def _on_rollout_end(self) -> None:
        if self.episode_scores:
            avg_score = sum(self.episode_scores) / len(self.episode_scores)
            self.logger.record("rollout/score", avg_score)
            # Zurücksetzen für den nächsten Rollout
            self.episode_scores = []

def train_ppo(total_timesteps=500_000_000):
    env = SnakeEnv()
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.001)
    
    # Überprüfe die Umgebung
    check_env(env, warn=True)
    
    # Callback, um den Score während des Trainings zu loggen
    score_callback = ScoreLoggingCallback(verbose=1)
    
    # Trainiere das Modell
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=score_callback)
    
    # Speichere das trainierte Modell
    model.save("ppo_snake")
    
    return model

def train_dqn(total_timesteps=500_000_000):
    env = SnakeEnv()
    model = DQN("MlpPolicy", env, verbose=1, learning_rate=0.001)
    
    # Überprüfe die Umgebung
    check_env(env, warn=True)
    
    # Callback, um den Score während des Trainings zu loggen
    score_callback = ScoreLoggingCallback(verbose=1)
    
    # Trainiere das Modell
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=score_callback)
    
    # Speichere das trainierte Modell
    model.save("dqn_snake")
    
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

    dqn_model = train_dqn(100)
    test(dqn_model)
    
    ppo_model = train_ppo(1000000)
    test(ppo_model)


