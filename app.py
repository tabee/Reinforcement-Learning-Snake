from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import time
from stable_baselines3 import DQN
from snake_env import SnakeEnv  # Deine Gym-Umgebung

app = Flask(__name__)
socketio = SocketIO(app)

# Lade das trainierte Modell (sorge dafür, dass dqn_snake.zip existiert)
model = DQN.load("dqn_snake")

# Erstelle die Umgebung
env = SnakeEnv()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('start_test')
def start_test():
    obs, _ = env.reset()  # Entpacke das Tupel und nutze nur die Beobachtung
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        socketio.emit('state_update', {
            'state': obs.tolist(),
            'score': env.score
        })
        time.sleep(0.1)
    socketio.emit('episode_end', {'score': env.score})


if __name__ == '__main__':
    socketio.run(app, debug=True)

