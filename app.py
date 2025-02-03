from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import random
import time

app = Flask(__name__)
socketio = SocketIO(app)

GRID_SIZE = 20
WIDTH, HEIGHT = 400, 400

# Globale Spielvariablen
game_running = False

def reset_game():
    global snake, direction, food, game_over, game_running
    snake = [{'x': WIDTH // 2, 'y': HEIGHT // 2}]
    direction = {'dx': GRID_SIZE, 'dy': 0}
    food = {
        'x': random.randint(0, (WIDTH - GRID_SIZE) // GRID_SIZE) * GRID_SIZE,
        'y': random.randint(0, (HEIGHT - GRID_SIZE) // GRID_SIZE) * GRID_SIZE
    }
    game_over = False
    game_running = True

def move_snake():
    global game_over, snake, food, game_running
    print("Starte move_snake() Task")
    while not game_over:
        head = {'x': snake[0]['x'] + direction['dx'], 'y': snake[0]['y'] + direction['dy']}
        print("Neuer Kopf:", head)
        if (head in snake or 
            head['x'] < 0 or head['x'] >= WIDTH or 
            head['y'] < 0 or head['y'] >= HEIGHT):
            print("Kollision erkannt. Game Over!")
            game_over = True
            game_running = False
            socketio.emit('game_over')
            break

        snake.insert(0, head)
        if head == food:
            print("Essen gefunden! Generiere neues Essen.")
            food = {
                'x': random.randint(0, (WIDTH - GRID_SIZE) // GRID_SIZE) * GRID_SIZE,
                'y': random.randint(0, (HEIGHT - GRID_SIZE) // GRID_SIZE) * GRID_SIZE
            }
        else:
            snake.pop()
        print("Aktuelle Snake:", snake)
        print("Aktuelles Essen:", food)
        socketio.emit('game_state', {'snake': snake, 'food': food})
        time.sleep(0.1)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('start_game')
def start_game():
    global game_running
    if not game_running:
        reset_game()
        print("Spiel gestartet!")
        socketio.start_background_task(move_snake)

@socketio.on('change_direction')
def change_direction(data):
    global direction
    new_dx, new_dy = data['dx'], data['dy']
    # Verhindert U-Turns
    if (new_dx, new_dy) != (-direction['dx'], -direction['dy']):
        direction = {'dx': new_dx, 'dy': new_dy}

if __name__ == '__main__':
    socketio.run(app, debug=True)
