import glob
import sys
import os
import logging

import numpy as np
import pandas as pd

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

import pyqtgraph as pg

# from __feature__ import snake_case, true_property

from . import arraydata


logger = logging.getLogger(__name__)


class SonicViewerWindow(QMainWindow):
    def __init__(self):
        super(SonicViewerWindow, self).__init__()
        # fns = sorted(glob.glob("RX*Wide Band.waf"))
        fns = sorted(glob.glob("*pass2_*.waf"))
        tab_widget = QTabWidget()
        self.setCentralWidget(tab_widget)
        for fn in fns:
            print("Loading %s..." % fn)
            log = SonicLog(fn)
            tab_widget.addTab(log, fn)
        self.showMaximized()


class SonicLog(QWidget):
    def __init__(self, waf_fn):
        super(SonicLog, self).__init__()

        self.waf = arraydata.WAF(waf_fn)
        # self.setWindowTitle(waf_fn)
        self.waf_fn = waf_fn
        self.vdl = VDL(self.waf)
        self.time_plot = pg.PlotWidget()
        self.vertical_plot_panel = QWidget()
        layout1 = QVBoxLayout()
        self.vertical_plot_panel.setLayout(layout1)
        self.vertical_plot = pg.PlotWidget()
        self.export_button = QPushButton("Export CSV")
        layout1.addWidget(self.vertical_plot)
        layout1.addWidget(self.export_button)

        self.topbottom_splitter = QSplitter()
        self.leftright_splitter = QSplitter()
        self.topbottom_splitter.setOrientation(Qt.Vertical)
        self.leftright_splitter.setOrientation(Qt.Horizontal)

        layout = QHBoxLayout()
        layout.addWidget(self.leftright_splitter)
        self.setLayout(layout)

        self.leftright_splitter.addWidget(self.vertical_plot_panel)
        self.leftright_splitter.addWidget(self.topbottom_splitter)

        self.topbottom_splitter.addWidget(self.vdl)
        self.topbottom_splitter.addWidget(self.time_plot)

        self.rois = []
        self.add_roi(DepthSlice(self.vdl, self.time_plot))
        self.add_roi(FixedWindow(200, 50, self.vdl, self.vertical_plot, self.time_plot))
        self.export_button.clicked.connect(self.export)

        self.vdl.set_vampl(500)
        self.vdl.sigVamplChanged.connect(self.set_vampl)

    def add_roi(self, roi):
        roi.show()
        self.rois.append(roi)

    def set_vampl(self, value):
        value = abs(value) * 2
        self.time_plot.getPlotItem().setYRange(value * -1, value, padding=0)

    def export(self):
        fixed_window = self.rois[1]  # very hard-coded thingy
        t0 = fixed_window.t0
        width = fixed_window.width
        fn = self.waf_fn.replace(
            ".waf", "_%.0f+%.0fus.csv" % (fixed_window.t0, fixed_window.width)
        )
        depths, times, values = fixed_window.data()
        depths, times, values = resample([depths, times], [depths, values])
        keys = [
            "Depth",
            "Time (%.0f-%.0f us)" % (t0, t0 + width),
            "Amplitude (%.0f-%.0f us)" % (t0, t0 + width),
        ]
        data = {
            "Depth": depths,
            "Time (%.0f-%.0f us)" % (t0, t0 + width): times,
            "Amplitude (%.0f-%.0f us)" % (t0, t0 + width): values,
        }
        pd.DataFrame(data, columns=keys).to_csv(fn, index=False)


def resample(*pairs):
    new_pairs = []
    for pair in pairs:
        i = np.argsort(pair[0])
        p0 = pair[0][i]
        p1 = pair[1][i]
        new_pairs.append((p0, p1))
    print(new_pairs)
    index_array = np.asarray(new_pairs[0][0])
    min_gradients = [np.min(index_array)]
    for pair in new_pairs[1:]:
        index_array = np.append(index_array, pair[0])
        min_gradients.append(np.min(np.gradient(pair[0])))
    print(min_gradients)
    index_reg = np.arange(
        np.min(index_array), np.max(index_array), np.min(min_gradients) / 5.0
    )
    data_regs = []
    for index_array, data in new_pairs:
        data_reg = np.interp(index_reg, index_array, data, left=np.nan, right=np.nan)
        data_regs.append(data_reg)
    return [index_reg] + data_regs


class DepthSlice(object):
    def __init__(self, vdl, time_plot, depth=None):
        self.vdl = vdl
        self.time_plot = time_plot
        if depth is None:
            depth = np.mean(vdl.waf.depths)
        self.depth = depth
        self.vdl_line = pg.InfiniteLine(
            depth, angle=0, movable=True, bounds=(vdl.waf.depths[0], vdl.waf.depths[-1])
        )
        logger.debug(f"self.vdl.waf.times.shape = {self.vdl.waf.times.shape}")
        logger.debug(f"self.trace.shape = {self.trace.shape}")
        self.trace_line = pg.PlotDataItem(self.vdl.waf.times, self.trace)
        self.vdl_line.sigPositionChanged.connect(self.update)

    @property
    def trace(self):
        depth, trace_data = self.vdl.waf.htrace(self.depth)
        return trace_data

    def update(self):
        self.depth = self.vdl_line.value()
        self.trace_line.setData(self.vdl.waf.times, self.trace)
        self.time_plot.setTitle(str(self.depth))

    def show(self):
        self.vdl.addItem(self.vdl_line)
        self.time_plot.addItem(self.trace_line)

    def hide(self):
        self.vdl.removeItem(self.vdl_line)
        self.time_plot.removeItem(self.trace_line)


class FixedWindow(object):
    def __init__(self, t0, width, vdl, vertical_plot, time_plot):
        self.vdl = vdl
        self.vertical_plot = vertical_plot
        self.time_plot = time_plot

        self.vdl_roi = pg.LinearRegionItem(
            values=[t0, t0 + width], orientation=pg.LinearRegionItem.Vertical
        )
        self.time_plot_roi = pg.LinearRegionItem(
            values=[t0, t0 + width], orientation=pg.LinearRegionItem.Vertical
        )
        depths, times, values = self.data()
        self.statistic_value_curve = pg.PlotDataItem(values, depths)
        self.statistic_time_curve = pg.PlotDataItem(times, depths, pen="r")

        self.vdl_roi.sigRegionChanged.connect(self.moved_vdl_roi)
        self.time_plot_roi.sigRegionChanged.connect(self.moved_time_plot_roi)
        self.vdl_roi.sigRegionChanged.connect(self.update)

    @property
    def t0(self):
        return min(self.vdl_roi.getRegion())

    @property
    def t1(self):
        return max(self.vdl_roi.getRegion())

    @property
    def width(self):
        return self.t1 - self.t0

    def update(self):
        depths, times, values = self.data()
        self.statistic_value_curve.setData(values, depths)
        self.statistic_time_curve.setData(times, depths)

    def data(self, statistic="min(positive)"):
        try:
            self.window_waf = self.vdl.waf.extract(trange=self.vdl_roi.getRegion())
        except ValueError:
            return [], [], []
        if statistic == "max(positive)":
            seek_data = self.window_waf.data
            return_data = self.window_waf.data
            func = np.argmax
        if statistic == "min(positive)":
            seek_data = self.window_waf.data * -1
            return_data = self.window_waf.data
            func = np.argmax
        if statistic == "max(abs)":
            seek_data = np.abs(self.window_waf.data)
            return_data = np.abs(self.window_waf.data)
            func = np.argmax
        time_args = func(seek_data, axis=1)
        times = np.asarray([self.window_waf.times[ti] for ti in time_args])
        values = []
        for i in range(len(time_args)):
            values.append(return_data[i, time_args[i]])
        values = np.asarray(values)
        # print("vdl_waf = %s" % (self.vdl.waf.depths.shape, ))
        # print("window_waf = %s" % (self.window_waf.depths.shape, ))
        return self.window_waf.depths, times, values

    def moved_vdl_roi(self):
        self.time_plot_roi.setRegion(self.vdl_roi.getRegion())

    def moved_time_plot_roi(self):
        self.vdl_roi.setRegion(self.time_plot_roi.getRegion())

    def show(self):
        self.vdl.addItem(self.vdl_roi)
        self.time_plot.addItem(self.time_plot_roi)
        self.vertical_plot.addItem(self.statistic_value_curve)
        self.vertical_plot.invertY()
        self.vdl.addItem(self.statistic_time_curve)

    def hide(self):
        self.vdl.removeItem(self.vdl_roi)
        self.time_plot.removeItem(self.time_plot_roi)
        self.vertical_plot.removeItem(self.statistic_value_curve)
        self.vdl.removeItem(self.statistic_time_curve)


class VDL(pg.ImageView):
    sigVamplChanged = Signal(float)

    def __init__(self, waf):
        super(VDL, self).__init__(view=pg.PlotItem())
        self.waf = waf
        self.view.removeItem(self.roi)
        self.view.removeItem(self.normRoi)
        self.ui.roiBtn.hide()
        hist_region = self.getHistogramWidget().item.region
        hist_region.lines[0].sigPositionChanged.connect(self.hist_drag_0_slot)
        hist_region.lines[1].sigPositionChanged.connect(self.hist_drag_1_slot)
        x0 = waf.times[0]
        x1 = waf.times[-1]
        y1 = waf.depths[-1]
        y0 = waf.depths[0]
        # print("VDL WAF data shape = %s" % (waf.data.shape, ))
        xscale = (x1 - x0) / waf.data.shape[1]
        yscale = (y1 - y0) / waf.data.shape[0]
        self.setImage(waf.data, pos=[x0, y0], scale=[xscale, yscale])
        self.view.setAspectLocked(False)

    def hist_drag_0_slot(self):
        hist_region = self.getHistogramWidget().item.region
        value = hist_region.lines[0].value()
        hist_region.lines[1].setValue(value * -1)
        self.sigVamplChanged.emit(abs(value))

    def hist_drag_1_slot(self):
        hist_region = self.getHistogramWidget().item.region
        value = hist_region.lines[1].value()
        hist_region.lines[0].setValue(value * -1)
        self.sigVamplChanged.emit(abs(value))

    def set_vampl(self, value):
        value = abs(value)
        hist = self.getHistogramWidget()
        hist.setLevels(value * -1, value)


def main():
    logging.basicConfig(level=logging.DEBUG)
    pg.setConfigOptions(imageAxisOrder="row-major")
    app = QApplication([])
    # app.setStyleSheet("QWidget { font-size: 0.9em; font-family: Verdana;}")
    # window = SonicViewerWindow(sys.argv[1])
    window = SonicViewerWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
