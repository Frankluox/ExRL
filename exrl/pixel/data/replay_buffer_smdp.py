from typing import Iterable, Optional, Union
import collections
import copy
import pickle

try:
    import gym
except ImportError:
    import gymnasium as gym
import jax
import numpy as np
from flax.core import frozen_dict

from exrl.pixel.data.dataset import Dataset, DatasetDict


def _init_replay_dict(obs_space: gym.Space, capacity: int) -> Union[np.ndarray, DatasetDict]:
    if isinstance(obs_space, gym.spaces.Box):
        return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
    if isinstance(obs_space, gym.spaces.Dict):
        data_dict = {}
        for key, value in obs_space.spaces.items():
            data_dict[key] = _init_replay_dict(value, capacity)
        return data_dict
    raise TypeError()


def _slice_replay_storage(storage: Union[np.ndarray, DatasetDict], end: int) -> Union[np.ndarray, DatasetDict]:
    if isinstance(storage, np.ndarray):
        return storage[:end].copy()
    if isinstance(storage, dict):
        return {key: _slice_replay_storage(value, end) for key, value in storage.items()}
    raise TypeError()


def _resize_replay_storage(storage, capacity, size):
    if isinstance(storage, np.ndarray):
        resized = np.empty((capacity, *storage.shape[1:]), dtype=storage.dtype)
        # Only populated rows are meaningful. Copying the uninitialized tail
        # touches tens of GB for a 100k dual-image replay during resume.
        resized[:size] = storage[:size]
        return resized
    if isinstance(storage, dict):
        return {key: _resize_replay_storage(value, capacity, size) for key, value in storage.items()}
    raise TypeError()


class ReplayBufferSMDP(Dataset):
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, capacity: int):
        self.observation_space = observation_space
        self.action_space = action_space
        self.capacity = int(capacity)

        print("making SMDP replay buffer of capacity", self.capacity)

        observations = _init_replay_dict(self.observation_space, self.capacity)
        next_observations = _init_replay_dict(self.observation_space, self.capacity)
        actions = np.empty((self.capacity, *self.action_space.shape), dtype=self.action_space.dtype)
        next_actions = np.empty((self.capacity, *self.action_space.shape), dtype=self.action_space.dtype)
        rewards = np.empty((self.capacity,), dtype=np.float32)
        masks = np.empty((self.capacity,), dtype=np.float32)
        discount = np.empty((self.capacity,), dtype=np.float32)
        durations = np.empty((self.capacity,), dtype=np.float32)
        env_ids = np.empty((self.capacity,), dtype=np.int32)
        traj_ids = np.empty((self.capacity,), dtype=np.int64)
        traj_steps = np.empty((self.capacity,), dtype=np.int32)
        mc_returns = np.empty((self.capacity,), dtype=np.float32)
        mc_return_valid = np.empty((self.capacity,), dtype=np.float32)

        self.data = {
            'observations': observations,
            'next_observations': next_observations,
            'actions': actions,
            'next_actions': next_actions,
            'rewards': rewards,
            'masks': masks,
            'discount': discount,
            'durations': durations,
            'env_ids': env_ids,
            'traj_ids': traj_ids,
            'traj_steps': traj_steps,
            'mc_returns': mc_returns,
            'mc_return_valid': mc_return_valid,
        }

        self.size = 0
        self._traj_counter = 0
        self._start = 0
        self.traj_bounds = {}
        self.completed_traj_ids = set()
        self.streaming_buffer_size = None

    def __len__(self) -> int:
        return self.size

    def length(self) -> int:
        return self.size

    def increment_traj_counter(self):
        if self._traj_counter not in self.traj_bounds:
            self.traj_bounds[self._traj_counter] = (self._start, self.size)
        self.completed_traj_ids.add(int(self._traj_counter))
        self._start = self.size
        self._traj_counter += 1

    def allocate_traj_id(self) -> int:
        traj_id = int(self._traj_counter)
        self._traj_counter += 1
        return traj_id

    def finish_traj(self, traj_id: int):
        traj_id = int(traj_id)
        if traj_id in self.traj_bounds:
            self.completed_traj_ids.add(traj_id)

    def _indices_for_traj(self, traj_id: int) -> np.ndarray:
        bounds = self.traj_bounds[int(traj_id)]
        if isinstance(bounds, tuple):
            start, end = bounds
            return np.arange(int(start), int(end), dtype=np.int64)
        return np.asarray(bounds, dtype=np.int64)

    def get_random_trajs(self, num_trajs: int):
        candidates = [traj_id for traj_id in sorted(self.completed_traj_ids) if traj_id in self.traj_bounds]
        candidates = [traj_id for traj_id in candidates if self._indices_for_traj(traj_id).size > 0]
        if not candidates:
            candidates = [traj_id for traj_id in sorted(self.traj_bounds) if self._indices_for_traj(traj_id).size > 0]
        if not candidates:
            raise ValueError("No non-empty trajectories available")
        which_trajs = np.random.choice(np.asarray(candidates), int(num_trajs), replace=True)
        observations_list = []
        next_observations_list = []
        actions_list = []
        rewards_list = []
        terminals_list = []
        masks_list = []
        discount_list = []
        durations_list = []

        for traj_idx in which_trajs:
            indices = self._indices_for_traj(int(traj_idx))
            indices = indices[np.argsort(indices)]

            obs_dict_curr_traj = {}
            for key in self.data['observations']:
                obs_dict_curr_traj[key] = self.data['observations'][key][indices]
            observations_list.append(obs_dict_curr_traj)

            next_obs_dict_curr_traj = {}
            for key in self.data['next_observations']:
                next_obs_dict_curr_traj[key] = self.data['next_observations'][key][indices]
            next_observations_list.append(next_obs_dict_curr_traj)

            actions_list.append(self.data['actions'][indices])
            rewards_list.append(self.data['rewards'][indices])
            terminals_list.append(1 - self.data['masks'][indices])
            masks_list.append(self.data['masks'][indices])
            discount_list.append(self.data['discount'][indices])
            durations_list.append(self.data['durations'][indices])

        return {
            'observations': observations_list,
            'next_observations': next_observations_list,
            'actions': actions_list,
            'rewards': rewards_list,
            'terminals': terminals_list,
            'masks': masks_list,
            'discount': discount_list,
            'durations': durations_list,
        }

    def insert(
        self,
        data_dict: DatasetDict,
        env_id: int = 0,
        traj_id: Optional[int] = None,
        traj_step: Optional[int] = None,
    ):
        if self.size == self.capacity:
            observations = _init_replay_dict(self.observation_space, self.capacity)
            next_observations = _init_replay_dict(self.observation_space, self.capacity)
            actions = np.empty((self.capacity, *self.action_space.shape), dtype=self.action_space.dtype)
            next_actions = np.empty((self.capacity, *self.action_space.shape), dtype=self.action_space.dtype)
            rewards = np.empty((self.capacity,), dtype=np.float32)
            masks = np.empty((self.capacity,), dtype=np.float32)
            discount = np.empty((self.capacity,), dtype=np.float32)
            durations = np.empty((self.capacity,), dtype=np.float32)
            env_ids = np.empty((self.capacity,), dtype=np.int32)
            traj_ids = np.empty((self.capacity,), dtype=np.int64)
            traj_steps = np.empty((self.capacity,), dtype=np.int32)
            mc_returns = np.empty((self.capacity,), dtype=np.float32)
            mc_return_valid = np.empty((self.capacity,), dtype=np.float32)

            data_new = {
                'observations': observations,
                'next_observations': next_observations,
                'actions': actions,
                'next_actions': next_actions,
                'rewards': rewards,
                'masks': masks,
                'discount': discount,
                'durations': durations,
                'env_ids': env_ids,
                'traj_ids': traj_ids,
                'traj_steps': traj_steps,
                'mc_returns': mc_returns,
                'mc_return_valid': mc_return_valid,
            }

            for key in data_new:
                if isinstance(self.data[key], np.ndarray):
                    self.data[key] = np.concatenate((self.data[key], data_new[key]), axis=0)
                elif isinstance(self.data[key], dict):
                    for subkey in self.data[key]:
                        self.data[key][subkey] = np.concatenate((self.data[key][subkey], data_new[key][subkey]), axis=0)
                else:
                    raise TypeError()
            self.capacity *= 2

        traj_id = int(self._traj_counter if traj_id is None else traj_id)
        traj_step = int(self.size - self._start if traj_step is None else traj_step)
        data_dict = dict(data_dict)
        data_dict.setdefault('env_ids', np.asarray(env_id, dtype=np.int32))
        data_dict.setdefault('traj_ids', np.asarray(traj_id, dtype=np.int64))
        data_dict.setdefault('traj_steps', np.asarray(traj_step, dtype=np.int32))
        data_dict.setdefault('mc_returns', np.asarray(0.0, dtype=np.float32))
        data_dict.setdefault('mc_return_valid', np.asarray(0.0, dtype=np.float32))

        insert_index = int(self.size)
        for key, value in data_dict.items():
            if key not in self.data:
                continue
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    self.data[key][subkey][self.size] = subvalue
            else:
                self.data[key][self.size] = value
        self.size += 1
        bounds = self.traj_bounds.setdefault(traj_id, [])
        if isinstance(bounds, tuple):
            start, end = bounds
            bounds = list(range(int(start), int(end)))
            self.traj_bounds[traj_id] = bounds
        bounds.append(insert_index)

    def compute_action_stats(self):
        actions = self.data['actions']
        return {'mean': actions.mean(axis=0), 'std': actions.std(axis=0)}

    def normalize_actions(self, action_stats):
        action_stats = copy.deepcopy(action_stats)
        action_stats['mean'][-1] = 0
        action_stats['std'][-1] = 1
        self.data['actions'] = (self.data['actions'] - action_stats['mean']) / action_stats['std']
        self.data['next_actions'] = (self.data['next_actions'] - action_stats['mean']) / action_stats['std']

    def sample(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[np.ndarray] = None,
    ) -> frozen_dict.FrozenDict:
        if self.streaming_buffer_size:
            indices = np.random.randint(0, self.streaming_buffer_size, batch_size)
        elif indx is not None:
            indices = indx
        else:
            indices = np.random.randint(0, self.size, batch_size)

        data_dict = {}
        for key in self.data:
            if keys is not None and key not in keys:
                continue
            if isinstance(self.data[key], np.ndarray):
                data_dict[key] = self.data[key][indices]
            elif isinstance(self.data[key], dict):
                data_dict[key] = {}
                for subkey in self.data[key]:
                    data_dict[key][subkey] = self.data[key][subkey][indices]
            else:
                raise TypeError()
        return frozen_dict.freeze(data_dict)

    def get_iterator(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[np.ndarray] = None,
        queue_size: int = 2,
    ):
        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(batch_size, keys, indx)
                queue.append(jax.device_put(data))

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)

    def save(self, filename):
        size = int(self.size)
        save_dict = {
            'data': _slice_replay_storage(self.data, size),
            'size': size,
            '_traj_counter': self._traj_counter,
            '_start': self._start,
            'traj_bounds': self.traj_bounds,
            'completed_traj_ids': sorted(self.completed_traj_ids),
            'saved_capacity': size,
        }
        with open(filename, 'wb') as f:
            pickle.dump(save_dict, f, protocol=4)

    def restore(self, filename):
        with open(filename, 'rb') as f:
            save_dict = pickle.load(f)
        self.data = save_dict['data']
        self.size = save_dict['size']
        self._traj_counter = save_dict['_traj_counter']
        self._start = save_dict['_start']
        self.traj_bounds = save_dict['traj_bounds']
        if isinstance(self.data['actions'], np.ndarray):
            loaded_capacity = int(self.data['actions'].shape[0])
        else:
            raise TypeError("Replay actions storage must be a numpy array")
        self.completed_traj_ids = set(save_dict.get('completed_traj_ids', []))
        self._ensure_optional_fields(loaded_capacity)

        target_capacity = max(int(self.capacity), loaded_capacity)
        self.capacity = loaded_capacity
        if target_capacity > loaded_capacity:
            self._expand_to_capacity(target_capacity)

    def _ensure_optional_fields(self, capacity: int):
        optional_specs = {
            'env_ids': (np.int32, 0),
            'traj_ids': (np.int64, 0),
            'traj_steps': (np.int32, 0),
            'mc_returns': (np.float32, 0.0),
            'mc_return_valid': (np.float32, 0.0),
        }
        for key, (dtype, fill_value) in optional_specs.items():
            if key not in self.data:
                self.data[key] = np.full((int(capacity),), fill_value, dtype=dtype)

    def _expand_to_capacity(self, target_capacity: int):
        target_capacity = int(target_capacity)
        if target_capacity <= int(self.capacity):
            return
        self.data = _resize_replay_storage(self.data, target_capacity, int(self.size))
        self.capacity = target_capacity

    def compute_mc_returns_in_place(self, gamma: float) -> int:
        if self.size <= 0:
            return 0
        self._ensure_optional_fields(int(self.capacity))
        gamma = float(gamma)
        chronological = np.arange(int(self.size), dtype=np.int64)
        returns_by_traj = {}
        valid_by_traj = {}
        count = 0
        for idx in chronological[::-1]:
            traj_id = int(self.data['traj_ids'][idx])
            if traj_id in self.completed_traj_ids and traj_id not in valid_by_traj:
                returns_by_traj[traj_id] = 0.0
                valid_by_traj[traj_id] = True
            if not bool(valid_by_traj.get(traj_id, False)):
                self.data['mc_returns'][idx] = 0.0
                self.data['mc_return_valid'][idx] = 0.0
                continue
            ret = float(self.data['rewards'][idx]) + (gamma ** float(self.data['durations'][idx])) * float(
                returns_by_traj.get(traj_id, 0.0)
            )
            returns_by_traj[traj_id] = ret
            self.data['mc_returns'][idx] = ret
            self.data['mc_return_valid'][idx] = 1.0
            count += 1
        return int(count)
