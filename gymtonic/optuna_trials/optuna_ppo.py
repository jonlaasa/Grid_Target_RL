import os
import gymnasium as gym
import random
import optuna
from optuna.samplers import TPESampler
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import gymtonic.envs
import json

# Función objetivo para optimización con Optuna
def objective(trial):
    # Hiperparámetros para PPO optimizados por Optuna
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.8, 0.995)
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.2)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    gae_lambda = trial.suggest_float("gae_lambda", 0.85, 0.99)
    vf_coef = trial.suggest_float("vf_coef", 0.1, 1.0)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    # Crear el entorno con Monitor
    env = gym.make('gymtonic/GridTarget-v0', n_rows=8, n_columns=10)
    env = Monitor(env)

    # Configurar PPO con los hiperparámetros sugeridos
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        ent_coef=ent_coef,
        clip_range=clip_range,
        gae_lambda=gae_lambda,
        vf_coef=vf_coef,
        batch_size=batch_size,
        verbose=0,
    )

    # Entrenar el modelo
    model.learn(total_timesteps=30_000)

    # Evaluar el modelo usando la función de Stable-Baselines3
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=2, deterministic=True)
    
    return mean_reward


if __name__ == "__main__":
    # Crear el entorno base (solo para reproducibilidad)
    seed = 42
    random.seed(seed)
    env = gym.make('gymtonic/GridTarget-v0', n_rows=6, n_columns=6)
    env.reset(seed=seed)



    # Configuración de Optuna
    study_name = "gridtarget_ppo"

    # Obtener el directorio donde se está ejecutando el script y subir un nivel
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)  # Subir un nivel

    # Crear el directorio para los resultados de Optuna dentro del directorio padre
    output_dir = os.path.join(parent_dir, "optuna", study_name)
    os.makedirs(output_dir, exist_ok=True)

    # Especificar la ruta de la base de datos dentro de la carpeta de resultados
    storage_file = f"sqlite:///{os.path.join(output_dir, 'optuna_gridtarget.db')}"
    tpe_sampler = TPESampler(seed=seed)

    study = optuna.create_study(
        sampler=tpe_sampler,
        direction="maximize",
        study_name=study_name,
        storage=storage_file,
        load_if_exists=True,
    )

    # Ejecutar la optimización
    n_trials = 2
    print(f"Buscando los mejores hiperparámetros en {n_trials} pruebas...")
    study.optimize(objective, n_trials=n_trials)

    # Guardar los mejores resultados en un archivo JSON
    best_trials = sorted(study.get_trials(), key=lambda t: t.value, reverse=True)[:1]
    best_trials_params = {
        f"trial_{i+1}": {
            "value": trial.value,
            "params": trial.params,
            "number": trial.number,
        }
        for i, trial in enumerate(best_trials)
    }

    # Guardar los mejores resultados en la carpeta de Optuna
    best_trials_dir = os.path.join(parent_dir, "best_trial_ppo")
    os.makedirs(best_trials_dir, exist_ok=True)

    with open(os.path.join(best_trials_dir, "best_trials.json"), "w") as f:
        json.dump(best_trials_params, f, indent=4)

    # Visualizar los resultados y guardar los gráficos
    optuna.visualization.plot_optimization_history(study).write_html(
        os.path.join(best_trials_dir, "optimization_history.html")
    )
    optuna.visualization.plot_param_importances(study).write_html(
        os.path.join(best_trials_dir, "param_importances.html")
    )

    print("Optimización completada.")