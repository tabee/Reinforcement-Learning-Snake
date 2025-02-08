import argparse
from snake_env import SnakeEnv
from stable_baselines3 import PPO

# load the environment
env = SnakeEnv()

def calculate_average_score(model, num_episodes=100):
    """
    Test the model and return the average score over a specified number of episodes.

    Parameters:
    model (object): The trained model to be tested.
    num_episodes (int, optional): The number of episodes to run the test. Default is 1000.

    Returns:
    float: The average score obtained over the specified number of episodes.
    """
    total_scores = 0

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _states = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        total_scores += env.score

    avg_score = total_scores / num_episodes
    print("Average score:", avg_score, "over", num_episodes, "episodes.")
    return avg_score

def execute_test_episode(model):
    """
    Test the model and print a single episode with the total reward.

    Parameters:
    model (object): The trained model used to predict actions.

    Returns:
    None

    The function runs a single episode using the provided model, rendering the environment at each step,
    and prints the reward for each step along with the total score. At the end of the episode, it prints
    the total reward and total score.
    """
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

    print("Episode finished. Total Reward:", total_reward, ". Total Score:", env.score)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test a PPO model for Snake game.')
    parser.add_argument('--load', type=str, default="best_model/best_model", help='Path to the model to be loaded')
    
    args = parser.parse_args()
    if args.load:
        print("Loading model from:", args.load)
        ppo_model = PPO.load("./models/" + args.load)
        execute_test_episode(ppo_model)
        calculate_average_score(ppo_model)
