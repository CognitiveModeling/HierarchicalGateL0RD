import io
import pickle
from collections import abc
from collections.abc import Iterable

import numpy as np
from forwardable import forwardable, def_delegators
from sklearn.model_selection import train_test_split

class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "rolloutbuffer":
            renamed_module = "utils.rolloutbuffer"

        return super(RenameUnpickler, self).find_class(renamed_module, name)


def rollout_load(file_obj):
    return RenameUnpickler(file_obj).load()


def rollout_loads(pickled_bytes):
    file_obj = io.BytesIO(pickled_bytes)
    return rollout_load(file_obj)


@forwardable()
class Rollout(object):
    allowed_fields = {"observations", "next_observations", "actions", "rewards", "dones",
                      "costs", "env_states", "model_states", "successes"}

    def_delegators("_data", "__len__, __iter__, __getitem__, __repr__")

    def __init__(self, field_names, transitions: Iterable, strict_field_names=True):
        transitions = list(transitions)  # Generators might cause problem in inspection of first element

        if strict_field_names:
            for name in field_names:
                if name not in self.allowed_fields:
                    raise NameError(f"Field name {name} not expected. Only {self.allowed_fields} are allowed!")

        if not transitions:
            raise ValueError("No data given!")

        self.dtype = [(name, "f8", np.array(item).shape) for name, item in zip(field_names, transitions[0])]
        self._data = np.array(transitions, dtype=self.dtype)

    @classmethod
    def from_dict(cls, **kwargs):
        field_names = kwargs.keys()
        transitions = zip(*kwargs.values())
        return cls(field_names, transitions)

    def cost_to_go(self, t, discount=1.0):
        return sum([self._data["rewards"][i] * discount ** (t - i) for i in range(t, len(self._data))])


# noinspection PyAbstractClass
@forwardable()
class _CustomList(abc.Sequence):
    """ A list-like structure with three additional features
    1) Maintains a maximum total size (sum of sizes of elements)
    2) Remembers how many elements were added last
    3) Remembers whether it was 'recently'  modified"""

    def_delegators("_list", "__len__, __iter__, __getitem__")

    def __init__(self, array_list, max_size=None):
        array_list = array_list or []

        if not isinstance(array_list, (list, tuple, Iterable)):
            raise TypeError("Expected data argument to be a list/tuple")
        array_list = list(array_list)

        self.max_size = max_size
        self._list = []
        self._ensure_data_fits(new_list=array_list)

        self._list.extend(array_list)
        self.modified = True
        self.number_of_latest_data_elems_added = len(array_list)
        self.list_number_of_latest_data_elems_added = [len(array_list)] if len(array_list) else []

    @property
    def _total_size(self):
        return sum([len(elem) for elem in self._list])

    def _resolve_overflow(self, overflow):
        if overflow <= 0:
            return
        idx = 0
        space_freed = 0
        while space_freed < overflow:
            assert idx < len(self._list)

            space_freed += len(self._list[idx])
            idx += 1

        self._list = self._list[idx:]
        self.modified = True

    def _ensure_data_fits(self, new_list):
        if self.max_size is not None:
            total_length = sum(len(item) for item in new_list)
            if total_length > self.max_size:
                raise ValueError(
                    f"Attempting to add data with size {total_length}"
                    f" bigger than max capacity of _CustomList {self.max_size}"
                )
            overflow = total_length + self._total_size - self.max_size
            self._resolve_overflow(overflow)

    def extend(self, other_list):
        self._ensure_data_fits(other_list)
        self.number_of_latest_data_elems_added = len(other_list)
        self._list.extend([item for item in other_list])
        self.list_number_of_latest_data_elems_added.append(self.number_of_latest_data_elems_added)
        self.modified = True

    def append(self, item):
        self.extend([item])


# noinspection PyAbstractClass
@forwardable()
class RolloutBuffer(abc.Sequence):
    def_delegators("rollouts", "__len__, __iter__, __getitem__, append, extend")

    def __init__(self, *, rollouts=None, max_size=None):
        self.rollouts = _CustomList(rollouts, max_size=max_size)
        self._last_flat = None
        self.reported_flatten_warning = False

        try:
            if self.rollouts:
                _ = self.flat
        except Exception as e:
            raise TypeError(f"Concatenating rollouts failed with error {e}")

    def common_field_names(self):
        all_field_names = [[d[0] for d in r.dtype] for r in self.rollouts]
        for f in Rollout.allowed_fields:
            if sum([1 for fields in all_field_names if f not in fields]) == 0:
                yield f

    @property
    def flat(self):
        if self.rollouts.modified:
            try:
                self._last_flat = np.concatenate(self.rollouts)
            except TypeError:
                # use only common fields
                common_fields = list(self.common_field_names())
                if not self.reported_flatten_warning:
                    print(f"Flatten of RolloutBuffer with different fields, use only {common_fields}")
                    self.reported_flatten_warning = True
                self._last_flat = np.concatenate([r[common_fields] for r in self.rollouts])
            self.rollouts.modified = False
        return self._last_flat

    def split(self, train_size=None, test_size=None, shuffle=True):
        train_size = train_size or 1.0 - test_size or None
        test_size = test_size or 1.0 - train_size or None

        if train_size is None and test_size is None:
            raise ValueError("At least one of train_size, test_size must be specified")

        train_rollouts, test_rollouts = train_test_split(
            self.rollouts, train_size=train_size, test_size=test_size, shuffle=shuffle
        )

        return RolloutBuffer(rollouts=train_rollouts), RolloutBuffer(rollouts=test_rollouts)

    def as_array(self, key):
        """ returns for the given field key an array of shape: rollouts, time, dim(of field) """
        try:
            return np.concatenate([item[key][None, ...] for item in self], axis=0)
        except Exception as e:
            raise TypeError(
                f"Turning rollout structure into numpy array failed." f" Rollouts of unequal length? Error: {e}"
            )

    @property
    def latest_rollouts(self):
        last_rollouts = self.rollouts[-self.rollouts.number_of_latest_data_elems_added:]
        return RolloutBuffer(rollouts=last_rollouts)

    def last_n_iterations(self, num_iter=1):
        return self.last_n_rollouts(sum(self.rollouts.list_number_of_latest_data_elems_added[-num_iter:]))

    def last_n_rollouts(self, last_n=1):
        last_rollouts = self.rollouts[-last_n:]
        return RolloutBuffer(rollouts=last_rollouts)

    def n_iterations(self, start_iter=0, end_iter=1):
        return self.n_rollouts(start_n=sum(self.rollouts.list_number_of_latest_data_elems_added[:start_iter]),
                               end_n=sum(self.rollouts.list_number_of_latest_data_elems_added[:end_iter]))

    def n_rollouts(self, start_n=0, end_n=1):
        rollouts = self.rollouts[start_n:end_n]
        return RolloutBuffer(rollouts=rollouts)

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.flat[item]
        if isinstance(item, tuple) and all(isinstance(sub_item, str) for sub_item in item):
            return tuple([self.flat[sub_item] for sub_item in item])
        if isinstance(item, Iterable) and all(isinstance(sub_item, int) or np.isscalar(sub_item) for sub_item in item):
            return tuple([self.rollouts[sub_item] for sub_item in item])
        else:
            return self.rollouts[item]

    @property
    def mean_avg_reward(self):
        if not self.rollouts:
            return None
        return np.mean(self.flat["rewards"])

    @property
    def mean_max_reward(self):
        if not self.rollouts:
            return None
        return np.mean([np.max(rollout["rewards"]) for rollout in self.rollouts])

    @property
    def mean_return(self):
        if not self.rollouts:
            return None
        return np.mean([np.sum(rollout["rewards"]) for rollout in self.rollouts])

    @property
    def std_return(self):
        if not self.rollouts:
            return None
        if len(self.rollouts) == 1:
            return 0
        else:
            return np.std([np.sum(rollout["rewards"]) for rollout in self.rollouts])

    @property
    def is_empty(self):
        if self.rollouts._list == []:
            return True
        else:
            return False
