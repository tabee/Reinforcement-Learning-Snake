from snake_env import SnakeEnv
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

def main():
    env = SnakeEnv()
    # Überprüfe, ob die Umgebung dem Gym-Interface entspricht
    check_env(env, warn=True)

    model = DQN("MlpPolicy", env, verbose=1)
    
    # Trainiere für 10.000 Zeitschritte
    model.learn(total_timesteps=1000000)
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

