from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import time
from stable_baselines3 import DQN, PPO
from snake_env import SnakeEnv

app = Flask(__name__)
socketio = SocketIO(app)

# Erstelle die Umgebung
env = SnakeEnv()

def _direction_to_text(direction):
    if direction == (0, -1):
        return "up"
    elif direction == (1, 0):
        return "right"
    elif direction == (0, 1):
        return "down"
    elif direction == (-1, 0):
        return "left"
    return "unknown"

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('start_test')
def start_test(data):
    model_name = data.get('model', 'ppo_snake_config0')  # Standardmodell, falls nichts übergeben wird
    model_path = f"./models/{model_name}"
    
    try:
        model = PPO.load(model_path)
    except Exception as e:
        emit('error', {'message': f'Fehler beim Laden des Modells: {str(e)}'})
        return

    obs, _ = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        grid = env.get_grid()
        
        socketio.emit('state_update', {
            'state': grid.tolist(),
            'score': env.score,
            'direction': _direction_to_text(env.direction),
        })
        time.sleep(0.1)

    socketio.emit('episode_end', {'score': env.score})

if __name__ == '__main__':
    socketio.run(app, debug=True)
