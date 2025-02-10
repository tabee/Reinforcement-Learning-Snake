import argparse
from snake_env import SnakeEnv
from stable_baselines3 import PPO

import argparse
from stable_baselines3 import PPO

AVAILABLE_MODELS = [
    "best_model_config0/best_model",
    "best_model_config1/best_model",
    "best_model_config2/best_model",
    "best_model_config3/best_model",
    "ppo_snake_config0",
    "ppo_snake_config1",
    "ppo_snake_config2",
    "ppo_snake_config3"
]

# load the environment
env = SnakeEnv()

def calculate_average_score(model, num_episodes=10):
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
    return f"Episode finished. Total Reward: {total_reward}, Total Score: {env.score}"

def test_model(model_path):
    print(f"Loading model from: {model_path}")
    try:
        ppo_model = PPO.load(f"./models/{model_path}")
        test_result = execute_test_episode(ppo_model)
        score_result = calculate_average_score(ppo_model, num_episodes=10000)
        return f"Model: {model_path}\n execute_test_episode {test_result}\n calculate_average_score {score_result}\n"
    except Exception as e:
        return f"Model: {model_path}\nError: {str(e)}\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a PPO model for Snake game.")
    parser.add_argument('--load', type=str, help="Path to the model to be loaded")
    parser.add_argument('--test_episode', action='store_true', help="Execute a test episode")
    parser.add_argument('--full_test', action='store_true', help="Test all available models and write results to a file")

    args = parser.parse_args()

    if args.full_test:
        with open("test_results.txt", "w") as file:
            for model in AVAILABLE_MODELS:
                result = test_model(model)
                file.write(result + "\n")
        print("Full test completed. Results saved to test_results.txt.")
    elif args.load:
        result = test_model(args.load)
        print(result)
    else:
        print("Error: Either --load or --full_test must be specified.")

