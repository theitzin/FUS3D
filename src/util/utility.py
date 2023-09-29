import inspect
import subprocess
from collections import OrderedDict

import numpy as np
import torch

def get_device(visible_devices=None):
    force_cpu = (visible_devices is not None and len(visible_devices) == 0)
    if torch.cuda.is_available() and not force_cpu:
        info = subprocess.check_output(['nvidia-smi', '--format=csv', '--query-gpu=memory.free'])
        gpu_mem = [int(line.split()[0]) for line in info.decode('utf-8').split('\n')[1:-1]]
        gpu_ranking = np.argsort(-np.array(gpu_mem))
        if visible_devices is not None:
            gpu_ranking = [r for r in gpu_ranking if r in visible_devices]

        gpu_index = int(gpu_ranking[0])
        device = 'cuda:%d' % gpu_index
        torch.cuda.set_device(gpu_index)
        print('device is %s (%d MiB free)' % (device, gpu_mem[gpu_index]))
    else:
        device = 'cpu'
        print('device is cpu')

    return torch.device(device)

def is_debugging():
    for frame in inspect.stack():
        if frame[1].endswith('pydevd.py'):
            print('debugger detected')
            return True
    return False

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

class DataTable:
    def __init__(self, data, spec, dtype=None):
        '''
        2d numpy or torch array. Columns can be indexed with string names specified by spec.
        '''
        indices = [i for key in spec for i in spec[key]]
        assert min(indices) >= 0 and max(indices) < data.shape[1]

        self.spec = spec.copy()
        self.data = self._asarray(data, dtype=dtype)

    @staticmethod
    def _asarray(data, dtype=None):
        if dtype is None and hasattr(data, 'dtype'):
            return data  # no change needed

        if dtype is not None and str(dtype).startswith('torch.'):
            return torch.as_tensor(data, dtype=dtype)
        else:
            return np.asarray(data, dtype=dtype)

    @staticmethod
    def _reshape(data, shape):
        if torch.is_tensor(data):
            return torch.reshape(data, shape)
        else:
            return np.reshape(data, shape)

    @staticmethod
    def _empty_like(data, shape):
        # data is dict contains either numpy array or torch tensors
        # first infer the type by creating an array / tensor from list of scalars of each dict entry
        entry_dummy = next(iter(data.values()))
        is_tensor = torch.is_tensor(entry_dummy)
        if is_tensor:
            type_dummy = torch.tensor([torch.zeros(1, dtype=data[key].dtype) for key in data])
            return torch.empty(shape, dtype=type_dummy.dtype, device=entry_dummy.device)
        else:
            type_dummy = np.array([np.zeros(1, dtype=data[key].dtype) for key in data])
            return np.empty(shape, dtype=type_dummy.dtype)

    @staticmethod
    def _cat(data, *args, axis=0, **kwargs):
        assert len(data) > 0
        if torch.is_tensor(data[0]):
            return torch.cat(data, dim=axis, *args, **kwargs)
        else:
            return np.concatenate(data, axis=axis, *args, **kwargs)

    @staticmethod
    def from_dict(data, spec=None, **kwargs):
        # Data is a dict with lists of same length such that they can be packed into a single (2d) table.
        # Dict keys are used as DataTable spec.

        # check if arguments are valid
        rows = [len(data[key]) for key in data]
        assert len(set(rows)) == 1
        n_rows = rows[0]

        if spec is None:
            assert rows[0] > 0
            data = OrderedDict([(key, DataTable._reshape(DataTable._asarray(data[key]), (n_rows, -1))) for key in data])
            n_cols = sum(data[key].shape[1] for key in data)

            # compute spec
            col_indices = list(range(n_cols))
            spec = {}
            for key in data:
                spec[key] = [col_indices.pop(0) for _ in range(data[key].shape[1])]
        else: # if spec is given then the case n_rows=0 can be handled as well
            data = OrderedDict([(key, DataTable._reshape(DataTable._asarray(data[key]), (-1, len(spec[key])))) for key in data])
            n_cols = sum(data[key].shape[1] for key in data)

        table_data = DataTable._empty_like(data, (n_rows, n_cols))
        for key in data:
            table_data[:, spec[key]] = data[key]

        return DataTable(table_data, spec, **kwargs)

    def to(self, device):
        if torch.is_tensor(self.data):
            self.data = self.data.to(device)
        return self

    def to_numpy(self):
        if torch.is_tensor(self.data):
            self.data = self.data.detach().cpu().numpy()
        return self

    def select(self, item):
        # same as __getitem__ but return DataTable if indexed by spec
        if not (type(item) == str or (type(item) == list and str in map(type, item))): # row selector
            return self[item]

        data = self[item]
        col_indices = list(range(data.shape[1]))
        spec = {}
        for key in item:
            spec[key] = [col_indices.pop(0) for _ in self.spec[key]]
        return DataTable(data, spec)

    def append(self, table):
        assert len(self.spec.keys() & table.spec.keys()) == 0

        n_cols = self.data.shape[1]
        self.data = DataTable._cat((self.data, table.data), axis=1)
        for key in table.spec:
            self.spec[key] = [n_cols + i for i in table.spec[key]]
        return self

    def copy(self):
        if torch.is_tensor(self.data):
            return DataTable(self.data.clone(), self.spec.copy())
        else:
            return DataTable(self.data.copy(), self.spec.copy())

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        if type(item) == str or (type(item) == list and str in map(type, item)):
            # use self.spec if object is indexed with string or list containing string
            if type(item) == list:
                indices = [i for key in item for i in self.spec[key]]
            else:
                indices = self.spec[item]
            return self.data[:, indices]
        else:
            # otherwise index data directly (this allows row indexing with slices or boolean arrays)
            # in this case we return another datatable. unexpected things can happen if you try to index columns here.
            selected_data = self.data[item]
            if len(selected_data.shape) == 1:
                selected_data = DataTable._reshape(selected_data, (1, selected_data.shape[0]))
            return DataTable(selected_data, self.spec)

    # def __setitem__(self, item, value):
    #     Implement later. Tricky to do properly... (problem arises when __getitem__ return value is scalar?

def recursive_operation(obj, op):
    if isinstance(obj, dict):
        return {k: recursive_operation(v, op) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_operation(v, op) for v in obj]
    elif isinstance(obj, tuple):
        return tuple([recursive_operation(v, op) for v in obj])
    elif isinstance(obj, DataTable):
        return DataTable(op(obj.data), spec=obj.spec)
    else:
        try:
            return op(obj)
        except:
            return obj
