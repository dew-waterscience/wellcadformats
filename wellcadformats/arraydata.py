import os

import numpy as np
import matplotlib.pyplot as plt



class WAF(object):

    def __init__(self, fn=None):
        if fn:
            self.load(fn)

    @property
    def vampl(self):
        return self.__vampl

    @vampl.setter
    def vampl(self, value):
        if value is None:
            value = np.nanmax(self.data)
        self.__vampl = value

    def load(self, fn):
        self.fn = fn
        alldata = np.loadtxt(open(fn, mode="r"), skiprows=2, delimiter=",")
        depths = alldata[:, 0]
        with open(fn, mode="r") as f:
            line = f.readline()
        times = np.array([float(s.split()[0]) for s in line.split(",")[1:]])
        data = alldata[:, 1:]
        self.generate(data, depths, times)

    def generate(self, data, depths, times, parent=None, vampl=None):
        self.data = data
        self.depths = depths
        self.times = times
        self.n = len(self.depths)
        self.m = len(self.times)

        self.cmap = plt.cm.gray
        self.vampl = vampl
        if parent:
            self.vampl = parent.vampl
            self.cmap = parent.cmap

    def plot_amplitude_hist(self, fig=None, ax=None):
        if ax is None:
            if fig is None:
                fig = plt.figure()
            ax = fig.add_subplot(111)
        amplitudes = ax.hist(self.data.ravel(), bins=30, log=True)

    def print_depths(self):
        print('%s %s %s' % ("depth range", depths[:3], "..."))
        print('%s %s %s' % (depths[self.n-3:], "n =", self.n))
        print('%s %s %s %s %s %s' % ("time range", times[:3], "...", times[self.m-3:], "m =", self.m))
        print(str(self.data.shape))

    def imshow(self, vampl=None, fig=None, figsize=None, ax=None, **kws):
        if ax is None:
            if figsize:
                fig = plt.figure(figsize=figsize)
            elif fig is None: 
                fig = plt.figure()
            ax = fig.add_subplot(111)
        if vampl:
            self.vampl = vampl
        defkws = dict(
            cmap=self.cmap, vmin=-1*self.vampl, vmax=self.vampl,
            interpolation="nearest", origin="upper",
            # left, right, bottom, top
            aspect="auto", extent=[
                self.times[0], self.times[-1],
                self.depths[-1], self.depths[0]])
        defkws.update(**kws)
        ax.imshow(self.data, **defkws)

    def extract(self, drange=None, trange=None):
        if drange is None:
            drange = (None, None)
        if trange is None:
            trange = (None, None)
        d0, d1 = drange
        if d0 is None:
            d0i = 0
        else:
            d0i = np.argmin((self.depths - d0) ** 2)
        if d1 is None:
            d1i = len(self.depths)
        else:
            d1i = np.argmin((self.depths - d1) ** 2)
        t0, t1 = trange
        if t0 is None:
            t0i = 0
        else:
            t0i = np.argmin((self.times - t0) ** 2)
        if t1 is None:
            t1i = len(self.times)
        else:
            t1i = np.argmin((self.times - t1) ** 2)
        # print('%s %s %s %s' % (d0, d1, d0i, d1i))
        # print('%s %s %s %s' % (t0, t1, t0i, t1i))
        depths = self.depths[d0i: d1i]
        times = self.times[t0i: t1i]
        new = WAF()
        new.generate(self.data[d0i: d1i, t0i: t1i], depths, times, parent=self)
        return new

    def htrace(self, depth, retdepth=False):
        di = np.argmin((self.depths - depth) ** 2)
        if retdepth:
            return self.depths[di], self.data[di, :]
        else:
            return self.data[di, :]

    def vtrace(self, time, rettime=False, interptime=True):
        ti = np.argmin((self.times - time) ** 2)
        if interptime:
            if ti < len(self.times) - 1:
                tiextra = ti + 1
            elif ti > 0:
                tiextra = ti - 1
            ti1, ti2 = sorted([ti, tiextra])
            damp = self.data[:, ti2] - self.data[:, ti1]
            dt = self.times[ti2] - self.times[ti1]
            plusamp = (damp / dt) * (time - self.times[ti1])
            return self.data[:, ti1] + plusamp
        else:
            if rettime:
                return self.times[ti], self.data[:, ti]
            else:
                return self.data[:, ti]




class WAI(object):

    def __init__(self, fn=None):
        self.__vmin = None
        self.__vmax = None
        if fn:
            self.load(fn)

    @property
    def vmax(self):
        return self.__vmax

    @vmax.setter
    def vmax(self, value):
        if value is None:
            value = np.nanmax(self.data)
        self.__vmax = value

    @property
    def vmin(self):
        return self.__vmin

    @vmin.setter
    def vmin(self, value):
        if value is None:
            value = np.nanmin(self.data)
        self.__vmin = value

    def load(self, fn):
        self.fn = fn
        alldata = np.loadtxt(open(fn, mode="r"), skiprows=2, delimiter=",")
        depths = alldata[:, 0]
        with open(fn, mode="r") as f:
            line = f.readline()
            line2 = f.readline()
        # Check for unit. Hardcoded as 0.1 us for now.
        data = alldata[:, 1:] / 10
        nsegments = data.shape[1]
        azimuths = np.linspace(0, 360, nsegments)
        self.generate(data, depths, azimuths)

    def generate(self, data, depths, azimuths, parent=None, vampl=None):
        self.data = data
        self.depths = depths
        self.azimuths = azimuths
        self.n = len(self.depths)
        self.m = len(self.azimuths)

        self.cmap = plt.cm.gray
        self.vampl = vampl
        if parent:
            self.vampl = parent.vampl
            self.cmap = parent.cmap

    def plot_amplitude_hist(self, fig=None, ax=None):
        if ax is None:
            if fig is None:
                fig = plt.figure()
            ax = fig.add_subplot(111)
        amplitudes = ax.hist(self.data.ravel(), bins=30, log=True)

    def print_depths(self):
        print('%s %s %s' % ("depth range", self.depths[:3], "..."))
        print('%s %s %s' % (self.depths[self.n-3:], "n =", self.n))
        print('%s %s %s' % ("azimuths range", self.azimuths[:3], "..."))
        print('%s %s %s' % (self.azimuths[self.m-3:], "m =", self.m))
        print(str(self.data.shape))

    def imshow(
            self, vmin=None, vmax=None, vampl=None, fig=None, ax=None, **kws):
        if ax is None:
            if fig is None:
                fig = plt.figure()
            ax = fig.add_subplot(111)
        if vampl:
            vmin = -1 * vampl
            vmax = vampl
        defkws = dict(
            cmap=self.cmap, vmin=vmin, vmax=vmax, interpolation="nearest",
            origin="upper", aspect="auto", extent=[
                self.azimuths[0], self.azimuths[-1],
                self.depths[-1], self.depths[0]])
        defkws.update(**kws)
        ax.imshow(self.data, **defkws)

    def plot_tt_slice(self, depth, fig=None, ax=None, **kws):
        if ax is None:
            if fig is None:
                fig = plt.figure()
            ax = fig.add_subplot(111, polar=True)
        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi / 2.)
        actualdepth, data = self.htrace(depth, retdepth=True)
        thetas = np.deg2rad(self.azimuths)
        ax.plot(thetas, data, **kws)
        ax.set_rlim(self.vmin, self.vmax)
        return ax

    def extract(self, drange=None, azrange=None):
        if drange is None:
            drange = (None, None)
        if azrange is None:
            azrange = (None, None)
        d0, d1 = drange
        if d0 is None:
            d0i = 0
        else:
            d0i = np.argmin((self.depths - d0) ** 2)
        if d1 is None:
            d1i = len(self.depths) - 1
        else:
            d1i = np.argmin((self.depths - d1) ** 2)
        t0, t1 = azrange
        if t0 is None:
            t0i = 0
        else:
            t0i = np.argmin((self.azimuths - t0) ** 2)
        if t1 is None:
            t1i = len(self.azimuths) - 1
        else:
            t1i = np.argmin((self.azimuths - t1) ** 2)
        print('%s %s %s %s' % (d0, d1, d0i, d1i))
        print('%s %s %s %s' % (t0, t1, t0i, t1i))
        depths = self.depths[d0i: d1i]
        azimuths = self.azimuths[t0i: t1i]
        new = FWSWaf()
        new.generate(
            self.data[d0i: d1i, t0i: t1i], depths, azimuths, parent=self)
        return new

    def htrace(self, depth, retdepth=False):
        di = np.argmin((self.depths - depth) ** 2)
        if retdepth:
            return self.depths[di], self.data[di, :]
        else:
            return self.data[di, :]

    def vtrace(self, time, rettime=False, interptime=True):
        ti = np.argmin((self.azimuths - azimuth) ** 2)
        if interptime:
            if ti < len(self.azimuths) - 1:
                tiextra = ti + 1
            elif ti > 0:
                tiextra = ti - 1
            ti1, ti2 = sorted([ti, tiextra])
            damp = self.data[:, ti2] - self.data[:, ti1]
            dt = self.azimuths[ti2] - self.azimuths[ti1]
            plusamp = (damp / dt) * (time - self.azimuths[ti1])
            return self.data[:, ti1] + plusamp
        else:
            if rettime:
                return self.azimuths[ti], self.data[:, ti]
            else:
                return self.data[:, ti]