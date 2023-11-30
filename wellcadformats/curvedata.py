import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class WAW(object):

    """Open .waw file.

    Args:
        file_obj (file-like object): optional open file-like object

    Attributes:
        names (list): Log names (str)
        units (list): Log units (str)
        data (numpy.ndarray): numpy array of data
        df (pandas.DataFrame): dataframe of data
        index (numpy.ndarray): reference data for the log (i.e. depth)

    """

    def __init__(self, filename=None, file_obj=None):
        self._units = {}
        if filename is not None or file_obj is not None:
            self.read(filename=filename, file_obj=file_obj)

    def read(self, file_obj=None, filename=None):
        if not filename is None:
            file_obj = open(filename, mode="r")
        for i, line in enumerate(file_obj.readlines()):
            line = line.strip("\n")
            if i == 0:
                names = line.split(",")
            elif i == 1:
                units = [u.strip() for u in line.split(",")]
            else:
                break
        for name, unit in zip(names, units):
            self.set_unit(name, unit)

        file_obj.seek(0)
        data = np.loadtxt(file_obj, skiprows=2, delimiter=",")
        self.df = pd.DataFrame(data, columns=names)

        if not filename is None:
            file_obj.close()

    def set_unit(self, name, unit):
        self._units[name] = unit

    def get_unit(self, name):
        if name in self._units:
            return self._units[name]
        elif name in self.df.columns:
            self._units[name] = ""
            return self.get_unit(name)
        else:
            raise KeyError(f"Well log '{name}' not known")

    @classmethod
    def from_df(cls, df, units=None):
        self = cls()
        self.df = df
        assert units is None or isinstance(units, dict)
        for log_name, log_unit in units.items():
            if log_name in self.df.columns:
                self.set_unit(log_name, log_unit)
        return self

    def to_file(self, file_obj):
        file_obj.write(",".join(self.df.columns) + "\n")
        file_obj.write(",".join([self.get_unit(col) for col in self.df.columns]) + "\n")
        self.df.to_csv(file_obj, header=False)

    def to_lasio_object(self):
        from lasio import LASFile

        las = LASFile()
        las.set_data_from_df(self.df.set_index(self.df.columns[0]))
        for curve in las.curves:
            curve.unit = self.get_unit(curve.mnemonic)
        return las

    def to_las(self, *args, **kwargs):
        las = self.to_lasio_object()
        return las.write(*args, **kwargs)
