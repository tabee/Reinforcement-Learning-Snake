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

        # Beobachtungsraum: Wir nutzen einen 9-dimensionalen Feature-Vektor für das Training.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        self.reset()

    def reset(self, **kwargs):
        """Setzt die Umgebung zurück und gibt (observation, info) zurück."""
        self.snake = [(self.cols // 2, self.rows // 2)]  # Schlange startet in der Mitte
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
        Terminierung (terminated) tritt bei einer Kollision auf.
        """
        # Bestimme neue Richtung anhand der Aktion:
        if action == 0:
            new_direction = (0, -1)  # oben
        elif action == 1:
            new_direction = (1, 0)   # rechts
        elif action == 2:
            new_direction = (0, 1)   # unten
        elif action == 3:
            new_direction = (-1, 0)  # links
        else:
            new_direction = self.direction

        # Verhindere U-Turns (falls die Schlange länger als 1 Segment ist)
        if len(self.snake) > 1 and (new_direction[0] == -self.direction[0] and new_direction[1] == -self.direction[1]):
            new_direction = self.direction
        else:
            self.direction = new_direction

        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])

        # Prüfe auf Kollision mit eigenem Körper
        if new_head in self.snake:
            reward = -1.05   # Reward für Kollision mit eigenem Körper
            terminated = True
            truncated = False
            return self._get_observation(), reward, terminated, truncated, {}

        # Prüfe auf Kollision mit Wand
        if (new_head[0] < 0 or new_head[0] >= self.cols or
            new_head[1] < 0 or new_head[1] >= self.rows):
            reward = -1   # Reward für Kollision mit der Wand
            terminated = True
            truncated = False
            return self._get_observation(), reward, terminated, truncated, {}

        # Schlange bewegen
        self.snake.insert(0, new_head)
        if new_head == self.food:
            reward = 1
            self.score += 1
            self._place_food()
        else:
            reward = 0
            self.snake.pop()  # Entferne das letzte Segment, wenn kein Essen erreicht wurde

        # Füge eine kleine Schrittstrafe hinzu
        reward += -0.01
        terminated = False
        truncated = False
        info = {"score": self.score}

        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self):
        """
        Erzeugt einen 9-dimensionalen Feature-Vektor:
        [danger_ahead, danger_left, danger_right, food_dx, food_dy, dir_right, dir_down, dir_left, dir_up]
        """
        head = self.snake[0]

        def is_danger(cell):
            x, y = cell
            if x < 0 or x >= self.cols or y < 0 or y >= self.rows:
                return 1.0
            if cell in self.snake:
                return 1.0
            return 0.0

        # Gefahrenindikatoren
        danger_ahead = is_danger((head[0] + self.direction[0], head[1] + self.direction[1]))
        left_dir = (-self.direction[1], self.direction[0])
        danger_left = is_danger((head[0] + left_dir[0], head[1] + left_dir[1]))
        right_dir = (self.direction[1], -self.direction[0])
        danger_right = is_danger((head[0] + right_dir[0], head[1] + right_dir[1]))

        # Relative Position des Essens (normalisiert)
        food_dx = (self.food[0] - head[0]) / self.cols
        food_dy = (self.food[1] - head[1]) / self.rows

        # One-Hot-Encoding der aktuellen Richtung (Reihenfolge: [rechts, unten, links, oben])
        if self.direction == (1, 0):
            dir_onehot = [1, 0, 0, 0]
        elif self.direction == (0, 1):
            dir_onehot = [0, 1, 0, 0]
        elif self.direction == (-1, 0):
            dir_onehot = [0, 0, 1, 0]
        elif self.direction == (0, -1):
            dir_onehot = [0, 0, 0, 1]
        else:
            dir_onehot = [0, 0, 0, 0]

        obs_vector = np.array([danger_ahead, danger_left, danger_right, food_dx, food_dy] + dir_onehot, dtype=np.float32)
        return obs_vector

    def get_grid(self):
        """Gibt das Gitter (2D-Array) für Visualisierungszwecke zurück (0 = leer, 1 = Schlange, 2 = Essen)."""
        grid = np.zeros((self.rows, self.cols), dtype=np.int32)
        for (x, y) in self.snake:
            grid[y, x] = 1
        grid[self.food[1], self.food[0]] = 2
        return grid

    def render(self, mode='human'):
        """Einfache textbasierte Darstellung des Gitters."""
        grid = self.get_grid()
        render_str = ""
        for row in grid:
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
