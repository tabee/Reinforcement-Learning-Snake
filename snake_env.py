import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class SnakeEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, grid_size=20, width=400, height=400):
        super(SnakeEnv, self).__init__()
        self.grid_size = grid_size
        self.width = width
        self.height = height
        self.cols = width // grid_size
        self.rows = height // grid_size

        # Aktionen: 0 = oben, 1 = rechts, 2 = unten, 3 = links
        self.action_space = spaces.Discrete(4)

        # Beobachtungsraum: Matrix (0 = leer, 1 = Schlangenkörper, 2 = Essen)
        self.observation_space = spaces.Box(low=0, high=2, 
                                            shape=(self.rows, self.cols), dtype=np.int32)
        self.reset()

    def reset(self, **kwargs):
        """Setzt die Umgebung zurück und gibt (observation, info) zurück."""
        self.snake = [(self.cols // 2, self.rows // 2)]  # Start in der Mitte
        self.direction = (1, 0)  # startet nach rechts
        self.done = False
        self.score = 0
        self._place_food()
        return self._get_observation(), {}

    def _place_food(self):
        """Plaziert das Essen an einer zufälligen, freien Stelle."""
        while True:
            self.food = (random.randint(0, self.cols - 1), random.randint(0, self.rows - 1))
            if self.food not in self.snake:
                break

    def step(self, action):
        """
        Führt einen Schritt aus und gibt (observation, reward, terminated, truncated, info) zurück.
        Terminierung (terminated) tritt bei Kollision auf.
        """
        # Bestimme neue Richtung
        if action == 0:
            new_direction = (0, -1)
        elif action == 1:
            new_direction = (1, 0)
        elif action == 2:
            new_direction = (0, 1)
        elif action == 3:
            new_direction = (-1, 0)
        else:
            new_direction = self.direction

        # Verhindere U-Turns (sofern Schlange länger als 1 Segment)
        if len(self.snake) > 1 and (new_direction[0] == -self.direction[0] and new_direction[1] == -self.direction[1]):
            new_direction = self.direction
        else:
            self.direction = new_direction

        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])

        # Prüfe auf Kollision (Wand oder Selbstkollision)
        if (new_head in self.snake or
            new_head[0] < 0 or new_head[0] >= self.cols or
            new_head[1] < 0 or new_head[1] >= self.rows):
            reward = -10
            terminated = True  # Spiel beendet wegen Kollision
            truncated = False
            return self._get_observation(), reward, terminated, truncated, {}

        # Schlange bewegen
        self.snake.insert(0, new_head)

        # Prüfe, ob Essen erreicht wurde
        if new_head == self.food:
            reward = 10
            self.score += 1
            self._place_food()
        else:
            reward = 0
            self.snake.pop()  # Entferne das letzte Segment

        # Kleine Schrittstrafe
        reward += -0.6

        terminated = False
        truncated = False

        return self._get_observation(), reward, terminated, truncated, {}

    def _get_observation(self):
        """Erzeugt eine Matrix als Beobachtung des aktuellen Zustands."""
        obs = np.zeros((self.rows, self.cols), dtype=np.int32)
        for (x, y) in self.snake:
            obs[y, x] = 1
        obs[self.food[1], self.food[0]] = 2
        return obs

    def render(self, mode='human'):
        """Einfache textbasierte Darstellung."""
        obs = self._get_observation()
        render_str = ""
        for row in obs:
            for cell in row:
                if cell == 0:
                    render_str += ". "
                elif cell == 1:
                    render_str += "S "
                elif cell == 2:
                    render_str += "F "
            render_str += "\n"
        print(render_str)

    def close(self):
        pass
