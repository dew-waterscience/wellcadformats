import glob
import os

from .arraydata import WAF, WAI
from .curvedata import WAW

from .wa import Comments

__version__ = "0.1"


class UnknownFormat(IOError):
    pass


def read(filename):
    name, ext = os.path.splitext(filename)
    ext = ext.lower()
    if ext == "waf":
        return WAF(filename)
    elif ext == "wai":
        return WAI(filename)
    elif ext == "waw":
        return WAW(filename=filename)
    else:
        raise UnknownFormat("Cannot read in %s files" % ext)


class MultipleExport(object):
    def __init__(self, path=None):
        self.datasets = []
        if not path is None:
            self.read(path)

    def read(self, filename):
        try:
            dset = read(fn)
            self.datasets.append(dset)
        except UnknownFormat:
            pass
        self.refresh()

    def readpath(self, path):
        fns = glob.glob(os.path.join(path, "*.w??"))
        self.readfiles(fns)

    def readfiles(self, *fns):
        for fn in fns:
            self.read(fn)

    # def refresh(self):
    #     import pandas as pd

    #     df = pd.DataFrame()
    #     for dset in self.datasets:
    #         if isinstance(dset, WAW):
    #             self.waw_datasets.append(dset)
    #     for i, dset in enumerate(self.waw_datasets):
    #         if not index_key in indices:
    #             indices[index_key] = 1
    #         else:
    #             indices[index_key] += 1
