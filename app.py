from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import time
from stable_baselines3 import DQN, PPO
from snake_env import SnakeEnv

app = Flask(__name__)
socketio = SocketIO(app)

# Erstelle die Umgebung
env = SnakeEnv()

@app.route('/')
def index():
    return render_template('index.html')

# Socket-Event, um das Modell zu wechseln
@socketio.on('switch_model')
def switch_model(model_name):
    global model
    if model_name == "DQN":
        model = DQN.load("dqn_snake")
    elif model_name == "PPO":
        model = PPO.load("ppo_snake")
    emit('model_switched', {'model': model_name})

@socketio.on('start_test')
def start_test():
    obs, _ = env.reset()  # Hier wird nur die Beobachtung (Feature-Vektor) entpackt
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # Für die Visualisierung: Hole das Gitter (2D-Array) statt des Feature-Vektors
        grid = env.get_grid()
        socketio.emit('state_update', {
            'state': grid.tolist(),
            'score': env.score
        })
        time.sleep(0.1)
    socketio.emit('episode_end', {'score': env.score})

if __name__ == '__main__':
    socketio.run(app, debug=True)
