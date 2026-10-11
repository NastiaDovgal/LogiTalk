import csv
from random import randint
import joblib
import pandas as pd
from pygame import *

from collect_data import collect_data

# Ініціалізація Pygame та шрифтів
font.init()
my_font = font.SysFont('Arial', 24)

FPS = 60
PLAYER_SPEED = 5
PLAYER_SIZE = 50
OBSTACLE_SIZE = 50

obstacle_speed = 3
is_game_over = False

window_size = 800, 600
window = display.set_mode(window_size)
clock = time.Clock()

# Завантаження навченої моделі ШІ
model = joblib.load('game_model.pkl')

# Режими керування: 'HUMAN' або 'AI'
control_mode = 'HUMAN'


def spawn_obstacle(count, obstacles):
  for _ in range(count):
    obstacles.append(
        Rect(
            randint(0, window_size[0] - OBSTACLE_SIZE),
            randint(-600, -100),
            OBSTACLE_SIZE,
            OBSTACLE_SIZE,
        )
    )
  return obstacles


player = Rect(0, window_size[1] - PLAYER_SIZE, 50, 50)
obstacles = spawn_obstacle(5, [])

frame_count = 0

while True:
  for e in event.get():
    if e.type == QUIT:
      quit()
    elif e.type == KEYDOWN:
      # Натисніть клавішу 'M' для змінення режиму керування
      if e.key == K_m:
        control_mode = 'AI' if control_mode == 'HUMAN' else 'HUMAN'

  keys = key.get_pressed()

  if not is_game_over:
    window.fill('light green')

    action = 'wait'

    # --- РЕЖИМ 1: Керує людина (HUMAN) ---
    if control_mode == 'HUMAN':
      if keys[K_d]:
        player.x += PLAYER_SPEED
        action = 'right'
      elif keys[K_a]:
        player.x -= PLAYER_SPEED
        action = 'left'

      # Збираємо дані ТІЛЬКИ в ручному режимі
      frame_count += 1
      if frame_count % 10 == 0:
        collect_data(player, obstacles, action)

    # --- РЕЖИМ 2: Керує ШІ (AI) ---
    elif control_mode == 'AI':
      if obstacles:
        # Знаходимо найближчу перешкоду
        nearest_obstacle = max(obstacles, key=lambda o: o.y)

        # Формуємо стан гри з тими самими ознаками, на яких навчалася модель
        state = pd.DataFrame([{
            'player_x': player.x,
            'obstacle_x': nearest_obstacle.x,
            'obstacle_y': nearest_obstacle.y,
            'distance_x': nearest_obstacle.x - player.x,
            'distance_y': player.y - nearest_obstacle.y,
        }])

        # Отримуємо прогноз дії від моделі
        predicted_action = model.predict(state)[0]

        # Виконуємо прогнозовану дію
        if predicted_action == 'right':
          player.x += PLAYER_SPEED
        elif predicted_action == 'left':
          player.x -= PLAYER_SPEED

    # Обмеження руху гравця в межах вікна
    player.x = max(0, min(window_size[0] - PLAYER_SIZE, player.x))

    # Малювання гравця
    draw.rect(window, 'sky blue', player)

    # Оновлення та малювання перешкод
    for obstacle in obstacles[:]:
      obstacle.y += obstacle_speed
      draw.rect(window, 'red', obstacle)

      if obstacle.colliderect(player):
        is_game_over = True

      if obstacle.y >= window_size[1]:
        obstacles.remove(obstacle)

    if not obstacles:
      obstacles = spawn_obstacle(randint(4, 8), obstacles)

    # Відображення поточного режиму керування на екрані
    mode_text = my_font.render(
        f'Mode: {control_mode} (Press M to switch)', True, (0, 0, 0)
    )
    window.blit(mode_text, (10, 10))

  if is_game_over:
    window.fill('red')
    game_over_text = my_font.render(
        f'GAME OVER in {control_mode} mode! Press R to restart.',
        True,
        (255, 255, 255),
    )
    window.blit(game_over_text, (100, 250))

    if keys[K_r]:
      is_game_over = False
      obstacles = spawn_obstacle(5, [])
      player.x = 0

  display.update()
  clock.tick(FPS)