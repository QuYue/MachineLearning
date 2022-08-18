# Basic
import os
import sys
import json
import datetime
import copy

# Add path
if __package__ is None:
    os.chdir(os.path.dirname(__file__))
    sys.path.append("..")

# %% Classes
class BatchRecord():
    """
    A Record for one batch.
    
    example:
    --------
    a = BatchRecord(2)
    a['key1'] = 1
    a['key2'] = [0,1,2,3]

    a['key2'].append(3)
    a['key2'].extend([4,5,6])
    """
    def __init__(self, size=0, index=0):
        self.info = dict()
        self.size = size
        self.index = index
    
    def __getitem__(self, key):
        return self.info[key]

    def __setitem__(self, key, value):
        self.info[key] = value
    
    def __len__(self):
        return (len(self.info), self.size)
    
    def keys(self):
        return list(self.info.keys())

    @property
    def shape(self):
        # (number of keys, batch size)
        return (len(self.info), self.size)

    def list_keys(self):
        list_keys = []
        keys = self.keys()
        for k in keys:
            if isinstance(self.info[k], list):
                list_keys.append(k)
        return list_keys

    def num_keys(self):
        num_keys = []
        keys = self.keys()
        for k in keys:
            if not isinstance(self.info[k], list):
                num_keys.append(k)
        return num_keys
    
    def __repr__(self) -> str:
        s = self.shape
        return f"BatchRecord[{self.index}](keys:{s[0]}, size:{s[1]})"


class Record(BatchRecord):
    def __init__(self, size=0, index=0):
        super().__init__(size, index)
        self.time = datetime.datetime.now()
        self.batch_num = 0

    @property
    def shape(self):
        # (number of keys, batch_size, batch_num)
        return (len(self.keys()), self.size, self.batch_num)
    
    def tojson(self):
        record_json = dict()
        record_json["class"] = "Record"
        record_json["time"] = self.time.strftime("%Y-%m-%d %H:%M:%S")
        record_json["size"] = self.size
        record_json["index"] = self.index
        record_json["batch_num"] = self.batch_num
        record_json["info"] = self.info
        return record_json

    def add_batch(self, batch_record: BatchRecord):
        self.batch_num += 1
        self.size += batch_record.size
        list_keys = batch_record.list_keys()
        list_keys = dict(zip(list_keys, range(len(list_keys))))
        num_keys = batch_record.num_keys()
        num_keys = dict(zip(num_keys, range(len(num_keys))))

        for k in batch_record.keys():
            if k in list_keys:
                if k in self.info:
                    self.info[k].extend(batch_record[k])
                else:
                    self.info[k] = batch_record[k]
            elif k in num_keys:
                if k in self.info:
                    self.info[k].append(batch_record[k])
                else:
                    self.info[k] = [batch_record[k]]
    
    def aggregate(self, key_dict):
        if not isinstance(key_dict, dict):
            raise ValueError("The type of argument 'key_dict' should be dict.")
        for k, v in key_dict.items():
            if k in self.info:
                if isinstance(v, str):
                    if v == "sum":
                        self.info[k] = sum(self.info[k])
                    elif v == "mean":
                        self.info[k] = sum(self.info[k]) / len(self.info[k])
                    elif v == "mean_size":
                        self.info[k] = sum(self.info[k]) / self.size 
                    else:
                        raise ValueError("The value of dict should be 'mean', 'mean_size', 'sum'.")
                elif hasattr(v, "__call__"):
                    self.info[k] = v(self.info[k])
            else:
                raise ValueError(f"The key '{k}' is not in this Record.")
    
    def __repr__(self) -> str:
        s = self.shape
        return f"Record[{self.index}](keys:{s[0]}, size:{s[1]}, batch_num:{s[2]})"

    def datashow(self, d, dpoint=2):
        if isinstance(d, int):
            template = "{}"
        elif isinstance(d, float):
            template = "{:." + str(dpoint) + "f}"
        else:
            template = "{:." + str(dpoint) + "f}"
        try:
            p = template.format(d)
        except:
            p = "{}".format(d)
        return p

    def print_str(self, keys=[], sep=' | ', isfinal=True, dpoint=2):
        if (not isinstance(keys, list)) and (not isinstance(keys, tuple)):
            str = f"{k}: {self.datashow(self.info[keys], dpoint)}"
            str += sep if isfinal else ''
        else:
            str = ''
            if len(keys) == 0:
                keys = self.keys()
            for k in keys:
                str += f"{k}: {self.datashow(self.info[k], dpoint)}" + sep
            if (not isfinal) and len(sep) > 0:
                str = str[:-len(sep)]
        return str

    def print(self, keys=[], sep=' | ', isfinal=True, dpoint=2):
        str = self.print_str(keys, sep, isfinal, dpoint)
        print(str)

    def print_all_str(self, sep=' | ', isfinal=True, dpoint=2):
        return self.print_str([], sep, isfinal, dpoint)

    def print_all(self, sep=' | ', isfinal=True, dpoint=2):
        self.print([], sep, isfinal, dpoint)
    

# %%
class Recorder():
    def __init__(self, data=[]):
        self.info = data
        self.max_show = 5

    def tolist(self):
        return self.info
    
    def __len__(self):
        return len(self.info)
    
    @property
    def shape(self):
        def count(a):
            dim = []
            if isinstance(a, list):
                dim.append(len(a))
                if len(a)>0:
                    dim += count(a[0])
            return dim
        dim = count(self.info)
        return tuple(dim)

    @property
    def dim(self):
        return len(self.shape)

    def __get_slice(self, n, a):
        if isinstance(n, int):
            return a[n]
        elif isinstance(n, list):
            L = []
            for i in n:
                if isinstance(i, int): L.append(a[i])
            return L
        elif isinstance(n, slice):
            return a[n]
        else:
            assert False, f"Recorder indices must be integers or list or slices, not {type(n)}."
    
    def __select4get(self, data, inputs, dim=0):
        data = self.__get_slice(inputs[0], data)
        if len(inputs) > 1:
            if isinstance(inputs[0], int):
                L = self.__select4get(data, inputs[1:], dim+1)
            else:
                L = []
                for d in data:
                    L.append(self.__select4get(d, inputs[1:], dim+1))
            data = L
        return data

    def __select4set(self, command, k, v, dim=0):
        if len(k) == 0:
            exec(command+'=v')
            return None
        if isinstance(k[0], int):
            self.__select4set(command+f"[{k[0]}]", k[1:], v, dim)
        elif isinstance(k[0], list):
            a = 'v'
            for i in range(dim):
                a += '[:]'
            for j, i in enumerate(k[0]):
                self.__select4set(command+f"[{i}]", k[1:], eval(a+f"[{j}]"), dim)

    def __getitem__(self, *inputs):
        if isinstance(inputs[0], tuple):
            inputs = inputs[0]
        assert self.dim >= len(inputs), f"The shape of Recorder is {self.shape}, but your indices is {inputs}, of which dimension is larger."
        data = self.__select4get(self.info, inputs)
        if isinstance(data, list):
            return Recorder(data)
        else:
            return data

    def __setitem__(self, k, v):
        shape1 = self.shape
        if not isinstance(k, tuple):
            k = [k]
        assert self.dim >= len(k), f"The shape of Recorder is {shape1}, but your indices is {k}, of which dimension is larger."
        if len(k) == self.dim:
            check = True
            for i in range(len(k)):
                if not isinstance(k[i],int): 
                    check = False
                    break
            if check:
                if isinstance(v, Recorder):
                    assert v.dim==0,  f"The shape of Split Recorder is {tuple()}, but the shape of data is {v.shape}, of which dimension is larger."
                else:
                    v = Recorder(v)
        shape = []
        for i in shape1:
            shape.append(list(range(i)))
        for i, index in enumerate(k):
            shape[i] = self.__get_slice(index, shape[i])
        new_shape = [len(i) for i in shape if isinstance(i, list) and len(i)>0]
        if isinstance(v, list):
            v = Recorder(v)
        try:
            shape2 = v.shape
        except:
            assert False, "Recorder can only be assigned to a data with 'shape' attribute."
        check = True
        if len(new_shape) == len(shape2):
            for i, j in zip(new_shape, shape2):
                if i != j: check=False
        else:
            check = False
        assert check, f"The shape of the sliced Holder is {tuple(new_shape)}, but the shape of input data is {shape2}."
        command = 'self.info'
        v = v.tolist()
        self.__select4set(command, shape, v)

    def append(self, new_Recorder, dim=0):
        def app(a, b, n=0):
            if n==dim:
                return a + b
            else:
                L = []
                for i in range(len(a)):
                    L.append(app(a[i], b[i], n+1))
                return L
        
        assert isinstance(new_Recorder, Recorder) or isinstance(new_Recorder, list), f"The first input of Function 'append' must be a Recorder or a List, not {type(new_Recorder)}."
        shape1 = self.shape
        if isinstance(new_Recorder, list):
            new_Recorder = Recorder(new_Recorder)
        shape2 = new_Recorder.shape
        assert self.dim == new_Recorder.dim, f"The shape of Recorder is {shape1}, but the shape of new Recorder, which need to be append, is {shape2}, can not be concatenated."
        assert self.dim > dim, f"The second input of Function 'append' is dimension for concatenation, the shape of Recorders are {shape1} and {shape2}, but dimension input by your is {dim}."
        check = True
        for i in range(len(shape1)):
            if i != dim and shape1[i]!=shape2[i]:
                check = False
        assert check,  f"Can not concatenate. Because the shape of Recorders are {shape1} and {shape2}, the dimension for concatenation is {dim}."
        data = app(self.info, new_Recorder.info)
        return Recorder(data)

    def concat(self, Recorder_list, dim=0):
        assert isinstance(Recorder_list, list), f"The first input of Function 'concat' must be a List, not {type(Recorder_list)}."
        d = self
        for i in Recorder_list:
            d = d.append(i, dim)
        return d

    def new_axis(self, dim):
        def add_axis(data, dim=0, new_dim=0):
            if dim == new_dim:
                data = [data]
            else:
                L = []
                for i in data:
                    L.append(add_axis(i, dim+1, new_dim))
                data = L
            return data
        assert isinstance(dim, int) and 0<=dim<=self.dim, f"The dimension of Recorder is {self.dim}, but the new axis input by you is {dim}, which is bigger."
        data = add_axis(self.info,0,dim)
        return Recorder(data)

    def pprint(self, data, max_dim=1, dim=0):
        p = ''
        bigger = False
        if len(data)>self.max_show+1:
            data = data[:self.max_show]+data[-1:]
            bigger = True
        length = len(data)
        if dim < max_dim-1:
            for i in range(length):
                if i==length-1 and bigger:
                    p += f"{' '*(dim+1)}...\n"
                if i != 0:
                    p += f"{' '*(dim+1)}"
                p += '['
                p += self.pprint(data[i], max_dim, dim+1)
                p += '],\n'
        else:
            for i in range(length):
                if i==length-1 and bigger:
                    p=p[:-1]
                    p+='...,'
                p += f"{data[i]}, "
        p = p[:-2]
        return p
    
    def __repr__(self):
        p = f"{self.__class__.__name__} with shape {self.shape}\n"
        p += '['
        p += self.pprint(self.info, max_dim=self.dim, dim=0, )
        p += ']'
        return p

    def query(self, key):
        def query_recorder(data, key, dim=0):
            if isinstance(data, list) and len(data)>0:
                L = []
                for i in data:
                    L.append(query_recorder(i, key, dim+1))
                return L
            elif isinstance(data, Record):
                return data[key]
            else:
                return data
        return query_recorder(self.info, key, 0)

    @property
    def index(self):
        def query_index(data, dim=0):
            if isinstance(data, list) and len(data)>0:
                L = []
                for i in data:
                    L.append(query_index(i, dim+1))
                return L
            elif isinstance(data, Record):
                return data.index
            else:
                return data
        return query_index(self.info, 0)

    def tojson(self):
        def iter(data):
            for i, d in enumerate(data):
                if isinstance(d, list):
                    iter(d)
                elif isinstance(d, Record):
                    data[i] = d.tojson()
        info = copy.deepcopy(self.info)
        iter(info)
        return info

    def save(self, filename):
        info = self.tojson()
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(info, f)

    def load_json(self, json):
        def iter(data):
            for i, d in enumerate(data):
                if isinstance(d, list):
                    iter(d)
                elif isinstance(d, dict) and d['class'] == 'Record':
                    temp = Record(d["size"], d["index"])
                    temp.size, temp.batch_num = d["size"], d["batch_num"]
                    temp.info = d["info"]
                    data[i] = temp
        info = json
        iter(info)
        self.info = info


# %% Functions
def read_json(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        d = json.load(f)
    recorder = Recorder([])
    recorder.load_json(d)
    return recorder

#%%
def Recorder_nones(shape):
    def create_none(s):
        L = []
        if len(s) == 1:
            L = [None] * s[0]
        else:
            for i in range(s[0]):
                L.append(create_none(s[1:]))
        return L
    d = Recorder(create_none(shape))
    return d

# %% Main Function
if __name__ == '__main__':
    a1 = BatchRecord(5, 0)
    a2 = BatchRecord(5, 1)
    a1['A'] = 1
    a1['B'] = [1,2,3]
    a1['C'] = [1,2]
    a1['D'] = [1,2,3]

    a2['A'] = 2
    a2['B'] = [3,4,5]
    a2['C'] = [3,4,5]
    a2['D'] = [2]

    b = Record()
    b.add_batch(a1)
    b.add_batch(a2)

    b.aggregate({'A': 'sum', 'B': 'mean', 'C': 'mean_size', 'D': lambda x: x[0]})

    b.print_all(sep=' | ', dpoint=3, isfinal=False)
    a = Recorder([[b,b],[b,b]])
    a.query('D')

# %%

