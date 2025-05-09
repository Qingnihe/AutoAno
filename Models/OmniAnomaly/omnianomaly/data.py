
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class SlidingWindowDataset(Dataset):

    def __init__(self, values, window_size, scores=None, if_wrap=False):
        self._values = values
        self._window_size = window_size
        if scores is not None:
            self.scores = scores[window_size:]

            score_th = np.percentile(self.scores, 80)
            # print(f"self.scores:{self.scores.tolist()} score_th:{score_th}")
            # print(f"self.scores:{self.scores.shape}")
            self.remove_index = np.where(self.scores>score_th)[0]
            print(f"remove_index:{self.remove_index.tolist()} {len(self.remove_index)}")
  
        else:
            self.scores = None

        if type(values)!=list:
            self._strided_values = self._to_windows(self._values)
        else:
            if if_wrap:
                for i in range(len(self._values)):
                    self._values[i] = np.concatenate([self._values[i][-window_size+1:], self._values[i]], axis=0)
                    # print(f"self._values[i].shape:{self._values[i].shape}")
            self._strided_values = self._to_windows_list(self._values)
            # print(f"self._strided_values:{self._strided_values.shape}")
        

    def __len__(self):
        return np.size(self._strided_values, 0)

    def _to_windows(self, values):
        return np.lib.stride_tricks.as_strided(values,
                                               shape=(np.size(values, 0) - self._window_size + 1, self._window_size, np.size(values, 1)),
                                               strides=(values.strides[-2], values.strides[-2], values.strides[-1]))
    def _to_windows_list(self, values_list):
        stride_list = []
        for values in values_list:
            s = np.lib.stride_tricks.as_strided(values,
                                                shape=(np.size(values, 0) - self._window_size + 1, self._window_size, np.size(values, 1)),
                                                strides=(values.strides[-2], values.strides[-2], values.strides[-1]))   
            if self.scores is not None:
                s = np.delete(s, self.remove_index, axis=0)
                # print(f"s:{s.shape}")
            stride_list.append(s)
        # print(f"len:{len(stride_list)} shape:{stride_list[0].shape[0]}")
        return np.concatenate(stride_list, axis=0)

    def __getitem__(self, index):
        return np.copy(self._strided_values[index]).astype(np.float32)


class _IndexSampler:

    def __init__(self, length, shuffle, drop_last, batch_size):
        self._index = np.arange(length)
        if shuffle:
            np.random.shuffle(self._index)
        self._pos = 0
        self._drop_last = drop_last
        self._batch_size = batch_size
        self._length = length

    def next(self):
        if self._pos + self._batch_size <= self._length:
            data = self._index[self._pos:self._pos+self._batch_size]
        elif self._pos >= self._length:
            raise StopIteration()
        elif self._drop_last:
            raise StopIteration()
        else:
            data = self._index[self._pos:]
        self._pos += self._batch_size
        return data


class SlidingWindowDataLoader(DataLoader):

    def __init__(self, dataset, batch_size, shuffle=False, drop_last=False):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        self._dataset = dataset
        self._shuffle = shuffle
        self._sampler = None

    def __next__(self):
        return torch.stack(tuple(torch.from_numpy(data) for data in self._dataset[self._sampler.next()]), 0)

    def __iter__(self):
        self._sampler = _IndexSampler(
            length=len(self._dataset),
            shuffle=self._shuffle,
            drop_last=self.drop_last,
            batch_size=self.batch_size
        )
        return self

    def __len__(self) -> int:
        if self.drop_last:
            return len(self._dataset) // self.batch_size
        else:
            return (len(self._dataset) + self.batch_size - 1) // self.batch_size
