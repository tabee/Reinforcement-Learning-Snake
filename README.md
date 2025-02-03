# Reinforcement-Learning-Snake

Dieses Projekt implementiert eine Snake-Umgebung gemäss dem OpenAI Gym Standard und trainiert einen Reinforcement Learning Agenten (DQN) mit stable-baselines3.

## Projektübersicht

![Bildschirmfoto vom 2025-02-04 00-35-31](https://raw.githubusercontent.com/tabee/Reinforcement-Learning-Snake/refs/heads/main/Bildschirmfoto%20vom%202025-02-04%2000-35-31.png)

Dieses Projekt besteht aus einer Snake-Umgebung, die mit Gymnasium erstellt wurde, und einem DQN-Agenten, der mit stable-baselines3 trainiert wird. Die Umgebung und der Agent werden verwendet, um das Verhalten der Schlange zu steuern und zu optimieren.

## Verzeichnisstruktur

```bash
__pycache__/
.gitignore
app.py
LICENSE
README.md
requirements.txt
snake_env.py
templates/
    index.html
train.py
```

- `app.py`: Startet eine Flask-Webanwendung zur Visualisierung des Snake-Spiels.
- `snake_env.py`: Implementiert die Snake-Umgebung gemäss dem Gym-Standard.
- `train.py`: Skript zum Trainieren des DQN-Agenten.
- `templates/index.html`: HTML-Datei für die Visualisierung des Spiels.
- `requirements.txt`: Liste der Python-Abhängigkeiten.
- `LICENSE`: Lizenzinformationen.
- `README.md`: Diese Datei.

## Installation

Installiere die Abhängigkeiten mit:

```bash
pip install -r requirements.txt
```

## Start Training

Starte das Training des DQN-Agenten mit:

```bash
python train.py
```

## Start Demo

Starte die Flask-Webanwendung zur Visualisierung des Spiels mit:

```bash
python app.py
```

## Nutzung

1. Öffne einen Webbrowser und gehe zu `http://127.0.0.1:5000/`.
2. Klicke auf den Button "Testlauf starten", um das Spiel zu starten.
3. Beobachte die Schlange, wie sie sich basierend auf dem trainierten Modell bewegt.

## Fehlerbehebung

- Stelle sicher, dass alle Abhängigkeiten korrekt installiert sind.
- Überprüfe, ob die Datei `dqn_snake.zip` im Verzeichnis vorhanden ist, bevor du die Demo startest.
- Bei Problemen mit Flask oder SocketIO, überprüfe die Versionen und Kompatibilität der Pakete in `requirements.txt`.

## Lizenz

Dieses Projekt steht unter der MIT-Lizenz. Siehe die [LICENSE](LICENSE) Datei für weitere Details.
