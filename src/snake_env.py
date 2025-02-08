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
        self.action_space = spaces.Discrete(4) # Aktionen: 0 = oben, 1 = rechts, 2 = unten, 3 = links
        # Beobachtungsraum: Wir nutzen einen 9-dimensionalen Feature-Vektor für das Training.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        #### reward and penalty
        self.reward_for_food = 1.0
        self.penalty_for_small_steps = -0.01
        self.penalty_for_hit_wall = -1.0
        self.penalty_for_hit_body = -2.0
        ####      
        self.reset()

    def eats_food(self):
        return self.snake[0] == self.food

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

    def _new_direction(self, action):
        """
        Determine the new direction of the snake based on the given action.

        Args:
            action (int): The action to be taken, where:
                          0 -> move up
                          1 -> move right
                          2 -> move down
                          3 -> move left

        Returns:
            tuple: A tuple representing the new direction (dx, dy).
                   If the action is not recognized, returns the current direction.
        """
        if action == 0:
            return (0, -1) # move up
        elif action == 1:
            return (1, 0) # move right
        elif action == 2:
            return (0, 1) # move down
        elif action == 3:
            return (-1, 0) # move left
        else:
            return self.direction # no change

    def _is_uturn(self, new_direction):
        """
        Check if the snake is making a U-turn.

        Args:
            new_direction (tuple): The new direction of the snake.

        Returns:
            bool: True if the snake is making a U-turn, False otherwise.
        """
        return (new_direction[0] == -self.direction[0] and new_direction[1] == -self.direction[1])

    def _hit_wall(self, head):
        """
        Check if the snake's head has hit the wall.

        Args:
            head (tuple): The (x, y) coordinates of the snake's head.

        Returns:
            bool: True if the snake's head has hit the wall, False otherwise.
        """
        return (head[0] < 0 or head[0] >= self.cols or
                head[1] < 0 or head[1] >= self.rows)

    def _hit_body(self, head):
        """
        Check if the snake's head has hit its body.

        Args:
            head (tuple): The (x, y) coordinates of the snake's head.

        Returns:
            bool: True if the snake's head has hit its body, False otherwise.
        """
        return head in self.snake[1:]
        
    def _eat_food(self, head):
        """
        Check if the snake's head has eaten the food.

        Args:
            head (tuple): The (x, y) coordinates of the snake's head.

        Returns:
            bool: True if the snake's head has eaten the food, False otherwise.
        """
        return head == self.food
        
    def step(self, action):
        # Bestimme neue Richtung und vermeide U-Turns
        new_direction = self._new_direction(action)
        if len(self.snake) > 1 and self._is_uturn(new_direction):
            new_direction = self.direction
        else:
            self.direction = new_direction

        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        
        # Kollisionsprüfungen vor dem Einfügen
        if self._hit_wall(new_head):
            reward = self.penalty_for_hit_wall
            self.done = True
            terminated = True
            truncated = False
            return self._get_observation(), reward, terminated, truncated, {}
        if self._hit_body(new_head):
            reward = self.penalty_for_hit_body
            self.done = True
            terminated = True
            truncated = False
            return self._get_observation(), reward, terminated, truncated, {}

        # Kein Kollisionsfehler: Neuer Kopf einfügen
        self.snake.insert(0, new_head)
        
        if self._eat_food(new_head):
            reward = self.reward_for_food
            self.score += 1
            self._place_food()
        else:
            reward = 0
            self.snake.pop()  # Schwanzsegment entfernen

        terminated = False
        truncated = False
        return self._get_observation(), reward, terminated, truncated, {}

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
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for i, (x, y) in enumerate(self.snake):
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                # Wenn die Schlange tot ist, Kopf als 'X' markieren (Wert 4)
                if i == 0:
                    grid[y, x] = 4 if self.done else 3
                else:
                    grid[y, x] = 1
            else:
                print(f"Index out of bounds: x={x}, y={y}")
        if 0 <= self.food[0] < self.grid_size and 0 <= self.food[1] < self.grid_size:
            grid[self.food[1], self.food[0]] = 2
        else:
            print(f"Food index out of bounds: x={self.food[0]}, y={self.food[1]}")
        return grid

    def render(self, mode='human'):
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
                elif cell == 3:
                    render_str += "H "
                elif cell == 4:
                    render_str += "X "  # Kopf als 'X', wenn tot
            render_str += "\n"
        print(render_str)

    def close(self):
        pass
