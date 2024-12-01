import os
import gymnasium as gym
import numpy as np
import random
import json
from stable_baselines3 import PPO
from datetime import datetime
import gymtonic.envs
from stable_baselines3.common.evaluation import evaluate_policy


def train_manual_ppo(env, model, n_steps):
    """
    Entrena un modelo PPO por el número de pasos especificados y evalúa su desempeño usando la función evaluate_policy de stable-baselines3.
    """
    # Entrenar el modelo
    model.learn(total_timesteps=n_steps)

    # Evaluar el modelo usando evaluate_policy de stable-baselines3
    avg_reward, avg_steps = evaluate_policy(model, env, n_eval_episodes=1)
    print(f"Avg Reward: {avg_reward}, Avg Steps: {avg_steps}")

    return model, avg_reward


if __name__ == "__main__":
    # Configuración del entorno
    env = gym.make('gymtonic/GridTarget-v0', n_rows=7, n_columns=7)

    # Establece la semilla para reproducibilidad
    seed = 42
    random.seed(seed)
    env.reset(seed=seed)

    # Número de pasos de entrenamiento
    n_steps = 50_000  # Puedes ajustar este valor según lo necesites

    # Mejores conjuntos de hiperparámetros para PPO
    best_params = {
        "learning_rate": 0.001,  # Velocidad de aprendizaje comúnmente efectiva
        "batch_size": 64,  # Tamaño de batch estándar para estabilidad
        "n_steps": 1024,  # Pasos de actualización para equilibrar velocidad y estabilidad
        "gamma": 0.99,  # Alto descuento para un horizonte largo
        "gae_lambda": 0.95,  # Promedio estándar para GAE
        "ent_coef": 0.01,  # Incentiva la exploración sin ser muy alto
        "vf_coef": 0.5,  # Importancia del error del valor
        "max_grad_norm": 0.5,  # Previene grandes actualizaciones de gradientes
    }

    # Rutas relativas para modelos y logs
    current_dir = os.path.dirname(os.path.realpath(__file__))  # Directorio actual
    model_dir = os.path.join(current_dir, 'ppo_gridd_target_model')
    log_dir = os.path.join(current_dir, 'logs')
    os.makedirs(model_dir, exist_ok=True)  # Crear el directorio si no existe

    # Crear el directorio para los resultados si no existe
    training_results_dir = os.path.join(current_dir, 'training_results')
    os.makedirs(training_results_dir, exist_ok=True)  # Crear el directorio si no existe

    # Obtener la fecha actual para nombres únicos
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Buscar el modelo más reciente en el directorio
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".zip")]
    if model_files:
        model_files.sort(reverse=True)
        latest_model_path = os.path.join(model_dir, model_files[0])
        print(f"Found existing model: {latest_model_path}. Continuing training from this model.")
        model = PPO.load(latest_model_path, env=env)
    else:
        print("No existing model found. Creating a new model...")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=best_params["learning_rate"],
            batch_size=best_params["batch_size"],
            n_steps=best_params["n_steps"],
            gamma=best_params["gamma"],
            gae_lambda=best_params["gae_lambda"],
            ent_coef=best_params["ent_coef"],
            vf_coef=best_params["vf_coef"],
            max_grad_norm=best_params["max_grad_norm"],
            tensorboard_log=log_dir,
            verbose=2,
        )

    # Entrenar el modelo
    print("Training the model...")
    model, avg_reward = train_manual_ppo(env, model, n_steps)
    
    # Guardar el modelo actualizado con la fecha actual
    model_path = os.path.join(model_dir, f"ppo_gridd_target_model_{date_str}.zip")
    model.save(model_path)
    print(f"Model saved at {model_path}")

    # Guardar las recompensas promedio en un archivo JSON
    results = {
        "model_name": f"ppo_gridd_target_model_{date_str}.zip",
        "final_reward": avg_reward,
    }
    results_path = os.path.join(training_results_dir, f"training_results_{date_str}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Training results saved at {results_path}")

