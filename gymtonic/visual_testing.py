import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import gymtonic.envs


if __name__ == "__main__":
    # Crear el entorno con el tamaño de 7x7
    env = gym.make('gymtonic/GridTarget-v0', n_rows=7, n_columns=7, render_mode='human')

    # Especifica manualmente la ruta del modelo .zip
    model_path = '/home/rl/gymtonic/ppo_gridd_target_model/ppo_gridd_target_model_2024-12-01_22-27-01.zip'  # Cambia esta ruta por la correcta

    # Comprobar si el archivo existe
    if not os.path.isfile(model_path):
        print(f"El archivo {model_path} no existe. Exiting.")
        exit()

    # Cargar el modelo
    print(f"Cargando el modelo desde {model_path}...")
    model = PPO.load(model_path, env=env)

    # Evaluar el modelo
    print("Evaluando el modelo...")
    avg_reward, avg_steps = evaluate_policy(model, env, n_eval_episodes=5, deterministic=False, render=True)
    
    # Mostrar el resultado de la evaluación
    print(f"Avg Reward: {avg_reward}, Avg Steps: {avg_steps}")