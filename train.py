from snake_env import SnakeEnv
from stable_baselines3 import DQN
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

def main():
    env = SnakeEnv()
    # Überprüfe, ob die Umgebung dem Gym-Interface entspricht
    check_env(env, warn=True)

    model = DQN("MlpPolicy", env, verbose=1)
    
    # Erstelle den Score Logging Callback
    score_callback = ScoreLoggingCallback(verbose=1)
    
    # Trainiere für Zeitschritte und verwende dabei den Callback
    model.learn(total_timesteps=100_000, callback=score_callback)
    model.save("dqn_snake")

    # Testlauf des trainierten Modells:
    obs, _ = env.reset()  # Hier wird nur die Beobachtung entpackt
    done = False
    total_reward = 0

    while not done:
        action, _states = model.predict(obs)
        # Da step() jetzt 5 Werte zurückgibt, entpacken wir entsprechend:
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        env.render()
        print("Reward:", reward, "Total Score:", env.score)

    print("Episode beendet. Total Reward:", total_reward)

if __name__ == "__main__":
    main()

