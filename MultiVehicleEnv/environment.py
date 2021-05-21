import gym
from gym import spaces
import numpy as np

# environment for all vehicles in the multi-vehicle world
# currently code assumes that no vehicles will be created/destroyed at runtime!
class MultiVehicleEnv(gym.Env):
    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_reward = False):

        self.world = world
        self.vehicles = self.world.vehicles
        # set required vectorized gym env property
        self.vehicles_number = len(self.world.vehicles)
        self.shared_reward = shared_reward

        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback

        self.total_time = 0.0

        # action spaces
        self.action_space = []
        for vehicle in self.vehicles:
            self.action_space.append(spaces.Discrete(len(vehicle.discrete_table)))

        # observation space
        self.observation_space = []
        for vehicle in self.vehicles:
            obs_dim = len(observation_callback(vehicle, self.world))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
    # set env action for a particular vehicles
    def _set_action(self, action, vehicle):
        assert isinstance(action,int)
        [ctrl_vel_b,ctrl_phi] = vehicle.discrete_table[action]
        vehicle.state.ctrl_vel_b = ctrl_vel_b
        vehicle.state.ctrl_phi = ctrl_phi

    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        self.vehicles = self.world.vehicles
        # set action for each vehicles
        for i, vehicle in enumerate(self.vehicles):
            self._set_action(action_n[i], vehicle)
        # advance world state
        self.world.step()
        # record observation for each vehicles
        for vehicle in self.vehicles:
            obs_n.append(self._get_obs(vehicle))
            reward_n.append(self._get_reward(vehicle))
            done_n.append(self._get_done(vehicle))

            info_n['n'].append(self._get_info(vehicle))

        # all vehicles get total reward in cooperative case
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [reward] * self.vehicles_number
        self.world.dumpGUI()
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        # reset world
        self.reset_callback(self.world)
        self.world.total_time = 0.0
        # reset renderer
        # record observations for each vehicles
        obs_n = []
        self.vehicles = self.world.vehicles
        for vehicles in self.vehicles:
            obs_n.append(self._get_obs(vehicles))
        self.world.dumpGUI()
        return obs_n

    # get info used for benchmarking
    def _get_info(self, vehicles):
        if self.info_callback is None:
            return {}
        return self.info_callback(vehicles, self.world)

    # get observation for a particular vehicles
    def _get_obs(self, vehicles):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(vehicles, self.world)

    # get dones for a particular vehicles
    # unused right now -- vehicles are allowed to go beyond the viewing screen
    def _get_done(self, vehicles):
        if self.done_callback is None:
            return False
        return self.done_callback(vehicles, self.world)

    # get reward for a particular vehicles
    def _get_reward(self, vehicles):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(vehicles, self.world)

