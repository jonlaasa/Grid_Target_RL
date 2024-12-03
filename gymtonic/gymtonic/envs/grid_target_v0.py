import gymnasium as gym
from gymnasium.core import ObsType, ActType, SupportsFloat
from gymnasium import spaces
from gymtonic.envs.grid_base import GridBaseEnv
import numpy as np
import pybullet as p
import pybullet_data
import time
from typing import Any

class GridTargetEnv(GridBaseEnv):
    """
    GridTargetEnv class represents an environment where an agent moves on a grid to reach a target.
    Args:
        n_rows (int): Number of rows in the grid (axis y). Default is 5.
        n_columns (int): Number of columns in the grid . Default is 5.
        smooth_movement (bool): Flag indicating whether smooth movement is enabled. Default is False.
        render_mode (str): Rendering mode for visualization. Default is None.
    """
    def calculate_manhattan(self, agent_pos, target_pos):
         return np.abs(agent_pos[0] - target_pos[0]) + np.abs(agent_pos[1] - target_pos[1])
    
    def calculate_euclidean(self, agent_pos, target_pos):
        # Calcular la diferencia absoluta en las coordenadas x e y
        diff = np.abs(agent_pos - target_pos)
        # Aplicar la fórmula de la distancia euclidiana con diferencias absolutas
        distance = np.linalg.norm(diff)

        return distance

    def __init__(self, n_rows=5, n_columns=5, smooth_movement=False, render_mode=None):
        super(GridTargetEnv, self).__init__(n_rows=n_rows, n_columns=n_columns,
                                        smooth_movement=smooth_movement, render_mode=render_mode)
        
        self.max_rows = 10  # Tamaño máximo del grid PARA SOLUCIONAR EL OBS SPACE CUANTO AUMENTEMOS EL GRID
        self.max_columns = 10
        self.n_rows = n_rows
        self.n_columns = n_columns

        # Redefine the observation space        
        # Multidiscrete observation space of the size of the board with 3 possible values (empty, agent or target)
        self.observation_space = spaces.MultiDiscrete(np.array([3] * self.max_rows * self.max_columns, dtype=np.int32))
        self.steps_taken = 0  # Agregar esta línea
        self.max_steps = self.max_columns * self.max_rows
        self.manhattan_to_objective = -1
        self.score = 0
        self.score_limit = 10 ## Intially set to 10
        self.phase = 1
        self.episode = 1
        

        print(f"Starting PHASE {self.phase}... Grid Size={self.n_columns}X{self.n_rows}")
        print(f"Starting EPISODE {self.episode}...")
        
        # Initial target position
        self.target_pos = np.array([self.n_columns-1, self.n_rows-1], dtype=np.int32)

        if self.render_mode == 'human':
            p.setAdditionalSearchPath(pybullet_data.getDataPath())        
            self.pybullet_target_id = p.loadURDF("duck_vhacd.urdf", [0, 0, 0], useFixedBase=False, globalScaling=7, baseOrientation=[0.5, 0.5, 0.5, 0.5])
            # Mass 0 for the target to avoid collision physics
            p.changeDynamics(self.pybullet_target_id, -1, mass=0)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        self.agent_pos = np.array([0, 0], dtype=np.int32)
        
        self.steps_taken = 0  # RESET THE STEPS TAKEN BY THE AGENT
        self.x_random_obj = np.random.randint(0, self.n_columns)   #  INITIALLY RANDOM
        self.y_random_obj = np.random.randint(0, self.n_rows)

        if self.episode == 1: ### SET TO A CERTAIN POSITION
            self.score_limit = 10
            self.x_random_obj = self.n_columns - 1
            self.y_random_obj = self.n_rows - 1

        if self.episode == 2: ### RANDOMIZE
            self.score_limit = 20 # We want the agent to train better in this phase because it used to go to the previous obj_position
            self.x_random_obj = np.random.randint(0, self.n_columns)  # Genera un valor entre 0 y n_columns - 1
            self.y_random_obj = np.random.randint(0, self.n_rows)

        self.target_pos = [self.x_random_obj, self.y_random_obj]
        
        self.manhattan_to_objective_initial = self.calculate_manhattan(self.agent_pos, self.target_pos)
        self.manhattan_to_objective_actual = self.calculate_manhattan(self.agent_pos, self.target_pos)

        if self.render_mode == 'human':
            self.update_visual_objects()
        
        obs = self.get_observation()
        info = {}
        return obs, info
    
    def update_visual_objects(self, force_teletransport = False):
        super().update_visual_objects(force_teletransport=force_teletransport)
        if self.render_mode == 'human':
            # Do a fancy rotation effect
            self.rotate_target_a_little()

    def rotate_target_a_little(self):
        # Rotate the target a little bit each time for a nice effect
        _, orientation = p.getBasePositionAndOrientation(self.pybullet_target_id)
        euler_angles = p.getEulerFromQuaternion(orientation)
        euler_angles = [euler_angles[0], euler_angles[1], euler_angles[2] - 0.15]
        quaternion = p.getQuaternionFromEuler(euler_angles)
        p.resetBasePositionAndOrientation(self.pybullet_target_id,[self.target_pos[0], self.target_pos[1], self.floor_height + 0.1], quaternion)
        self.render()


    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        
        self.steps_taken += 1
        obs, reward, terminated, truncated, info = super().step(action)
        
        reward = self.calculate_reward()
        
        if self.is_target_reached():
            terminated = True
            self.score += 1
            print(f"Target reached! Score: {self.score}")

            if self.score == self.score_limit: ## Se acabo el episodio, ha llegado al SCORE
                self.episode += 1 # Entrar en siguente episodio
                self.score = 0 # Score set to 0

                if self.episode == 3:   # Si el episodio es 3 == EMPEZAMOS SIGUENTE PHASE + ampliar grid!
                    self.phase += 1
                    self.episode = 1
                    
                    if (self.n_columns == 9 | self.n_rows == 9):  # Limite de grid de 10x10
                        print("ACABOOOO!")
                        self.n_columns = 8
                        self.n_rows = 8
                    self.n_columns += 1   # MODIFICAMOS EL ROWS Y EL COLUMNS PERO TAMBIEN EL OBSERVATION SPACE!
                    self.n_rows += 1
                    self.observation_space = spaces.MultiDiscrete(np.array([3] * self.n_rows * self.n_columns, dtype=np.int32))
                    print(f"Starting PHASE {self.phase}... Grid Size={self.n_columns}X{self.n_rows}")
                
                print(f"Starting EPISODE {self.episode}...")

            self.reset()  # Reinicia el entorno si se alcanza el objetivo

        # Reiniciar el entorno si se supera el límite de pasos
        if self.steps_taken >= self.max_steps:
            truncated = True
            print("Max steps reached!")
            self.reset()

        return obs, reward, terminated, truncated, info

    
    def calculate_reward(self):
        # Calcular la distancia de Manhattan entre el agente y el objetivo
        # Si el agente ha llegado al objetivo
        if self.is_target_reached():
            reward = 100

        else:
            real_manhattan=self.calculate_manhattan(self.agent_pos, self.target_pos)
            
            if self.manhattan_to_objective_actual > real_manhattan:
                reward = 5
                self.manhattan_to_objective_actual = real_manhattan

            elif self.manhattan_to_objective_actual == real_manhattan:
                reward = -1

            else:
                reward =-2
                self.manhattan_to_objective_actual = real_manhattan


        reward -= 0.1  # Penalización por cada paso para reducir ineficiencia
        return reward


    def get_observation(self):
        observation = np.zeros((self.max_columns, self.max_rows), dtype=np.int32)
        observation[self.agent_pos[0], self.agent_pos[1]] = 1
        observation[self.target_pos[0], self.target_pos[1]] = 2
        return observation.flatten()


    def is_target_reached(self):
        return np.array_equal(self.agent_pos, self.target_pos)

# Example usage
if __name__ == "__main__":
    env = gym.make('gymtonic/GridTarget-v0', n_rows=6, n_columns=10)
    obs, info = env.reset()
    n_episodes = 10
    for _ in range(n_episodes):
        for _ in range(100): # Maximum 100 steps
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = env.step(action)
            if env.render_mode == 'human':
                time.sleep(0.1)
            if terminated or truncated:
                break

        obs, info = env.reset()    
    
    env.close()
