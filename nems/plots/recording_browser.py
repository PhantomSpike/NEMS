#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 09:25:00 2018

@author: svd
"""

#!/usr/bin/env python

# embedding_in_qt5.py --- Simple Qt4 application embedding matplotlib canvases
#
# Copyright (C) 2005 Florent Rougon
#               2006 Darren Dale
# with Updates from @boxcontrol
# <http://www.boxcontrol.net/embedding-matplotlib-plot-on-pyqt5-gui.html>
# <https://github.com/boxcontrol/matplotlibPyQt5>
#
# This file is an example program for matplotlib. It may be used and
# modified with no restriction; raw copies as well as modified versions
# may be distributed without limitation.

import sys
import random
import copy
import types
import traceback
from functools import wraps
import pandas as pd

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np

import PyQt5.QtCore as qc
import PyQt5.QtGui as qg
import PyQt5.QtWidgets as qw

from numpy import arange, sin, pi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

#import nems_db.db as nd
#import nems_db.xform_wrappers as nw
import nems.xforms as xforms
import nems.plots.api as nplt
from nems.recording import Recording
import nems.signal
from nems.plots.utils import ax_remove_box

class RecordingPlotWrapper():
    # TODO: Not using this anymore?
    def __init__(self, recording):

        test_signals=['stim','resp','pred']

        self.recording = recording
        self.plot_signals = test_signals
        self.start_time = 0
        self.stop_time = 10

    def get_segment(self, signal):

        fs = self.recording[signal].fs
        start_bin = int(self.start_time * fs)
        stop_bin = int(self.stop_time * fs)

        d = recording[signal].as_continuous()[:,start_bin:stopbin]

        return d


def show_exceptions(*args):
    if len(args) == 0 or isinstance(args[0], types.FunctionType):
        args = []

    @qc.pyqtSlot(*args)
    def slotdecorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                func(*args)
            except:
                print("\n*** Uncaught Exception in PyQt slot ***")
                traceback.print_exc()
        return wrapper

    return slotdecorator


class MyMplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        # tweak size of axes in window
        pos1 = self.axes.get_position() # get the original position
        pos2 = [0.075, 0.1, 0.9, 0.8]
        self.axes.set_position(pos2) # set a new position

        # We want the axes cleared every time plot() is called
        self.axes.hold(False)

        self.compute_initial_figure()

        #
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, qw.QSizePolicy.Expanding,
                                   qw.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass


class MyStaticMplCanvas(MyMplCanvas):
    """Simple canvas with a sine plot."""
    def compute_initial_figure(self):
        t = arange(0.0, 3.0, 0.01)
        s = sin(2*pi*t)
        self.axes.plot(t, s)


class MyDynamicMplCanvas(MyMplCanvas):
    """A canvas that updates itself every second with a new plot."""
    def __init__(self, *args, **kwargs):
        MyMplCanvas.__init__(self, *args, **kwargs)
        timer = qc.QTimer(self)
        timer.timeout.connect(self.update_figure)
        timer.start(1000)

    def compute_initial_figure(self):
        self.axes.plot([0, 1, 2, 3], [1, 2, 0, 4], 'r')

    def update_figure(self):
        # Build a list of 4 random integers between 0 and 10 (both inclusive)
        l = [random.randint(0, 10) for i in range(4)]

        self.axes.plot([0, 1, 2, 3], l, 'r')
        self.draw()


class NemsCanvas(MyMplCanvas):
    """A canvas that updates itself every second with a new plot."""

    def __init__(self, recording=None, signal='stim', parent=None,
                 *args, **kwargs):
        MyMplCanvas.__init__(self, *args, **kwargs)
        self.recording = recording
        self.signal = signal
        self.parent = parent
        print("creating canvas: {}".format(signal))
        sig_array = self.recording[self.signal].as_continuous()
        # Chop off end of array (where it's all nan'd out after processing)
        # TODO: Make this smarter incase there are intermediate nans?
        no_nans = sig_array[:, ~np.all(np.isnan(sig_array), axis=0)]
        self.max_time = no_nans.shape[-1] / self.recording[self.signal].fs

    def compute_initial_figure(self):
        pass

    def update_figure(self):
        p = self.parent

        c_count = self.recording[self.signal].shape[0]
        fs = self.recording[self.signal].fs
        start_bin = int(p.start_time * fs)
        stop_bin = int(p.stop_time * fs)

        # skip some channels, get names
        channel_names=self.recording[self.signal].chans[:c_count]
        skip_channels = ['baseline']
        if channel_names is not None:
            keep = np.array([(n not in skip_channels) for n in channel_names])
            channel_names = [channel_names[i] for i in range(c_count) if keep[i]]
        else:
            keep = np.ones(c_count, dtype=bool)
            channel_names = None

        d = self.recording[self.signal].as_continuous()[keep, start_bin:stop_bin]

        point = (isinstance(self.recording[self.signal],
                            nems.signal.PointProcess))
        tiled = (isinstance(self.recording[self.signal],
                           nems.signal.TiledSignal)
                 or 'stim' in self.recording[self.signal].name)
        if point:
            self.axes.imshow(d, aspect='auto', cmap='Greys',
                             interpolation='nearest', origin='lower')
            self.axes.get_yaxis().set_visible(False)
        elif tiled:
            self.axes.imshow(d, aspect='auto', origin='lower')
        else:
            t = np.linspace(p.start_time, p.stop_time, d.shape[1])
            self.axes.plot(t, d.T)
            if (channel_names is not None) and len(channel_names)>1:
                self.axes.legend(channel_names, frameon=False)

        self.axes.set_ylabel(self.signal)
        self.axes.autoscale(enable=True, axis='x', tight=True)
        ax_remove_box(self.axes)
        self.draw()

        if point or tiled:
            tick_labels = self.axes.get_xticklabels()

            #new_labels = [round((t.get_position()[0]+start_bin)/fs)
            #              if t.get_text() else ''
            #              for t in tick_labels]
            new_labels = ['']*len(tick_labels)
            self.axes.set_xticklabels(new_labels)
            self.draw()


class ApplicationWindow(qw.QMainWindow):

    recording=None
    signals = []
    plot_list = []
    start_time = 0
    display_duration = 10.0
    minimum_duration = 0.001
    stop_time = 10
    time_slider = None

    plot_width = 6
    plot_height = 4
    plot_dpi = 100

    def __init__(self, recording, signals=['stim', 'resp'], cellid=None,
                 modelname=None):
        qw.QMainWindow.__init__(self)

        self.recording=recording
        self.signals=signals
        self.cellid=cellid
        self.modelname=modelname

        self.setAttribute(qc.Qt.WA_DeleteOnClose)

        self.file_menu = qw.QMenu('&File', self)
        self.file_menu.addAction('&Quit', self.fileQuit,
                                 qc.Qt.CTRL + qc.Qt.Key_Q)
        self.file_menu.addAction('Add Signal', self.add_signal,
                                 qc.Qt.CTRL + qc.Qt.Key_A)
        self.file_menu.addAction('Remove Signal', self.remove_signal,
                                 qc.Qt.CTRL + qc.Qt.Key_R)
        self.file_menu.addAction('Screenshot', self.save_screenshot,
                                 qc.Qt.CTRL + qc.Qt.Key_S)
        self.menuBar().addMenu(self.file_menu)

        self.help_menu = qw.QMenu('&Help', self)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.help_menu)

        self.help_menu.addAction('&About', self.about)

        self.main_widget = qw.QWidget(self)
        self.outer_layout = qw.QVBoxLayout()
        header = ''
        if modelname is not None:
            header += 'Model: %s\n' % modelname
        if cellid is not None:
            header += 'Cellid: %s' % cellid
        self.setWindowTitle("application main window")
        self.outer_layout.addWidget(qw.QLabel(header))


        # mpl panels
        signals = [value for value in signals if value in recording.signals]
        self.plot_list = [self._default_plot_instance(s) for s in signals]
        self.plot_layout = qw.QVBoxLayout()
        self.plot_layout.setSpacing(25)
        [self.plot_layout.addWidget(p) for p in self.plot_list]

        # Slider for plot view windows
        self._update_max_time()
        self.time_slider = qw.QScrollBar(orientation=1)
        self.time_slider.setRange(0, self.max_time-self.display_duration)
        self.time_slider.valueChanged.connect(self.scroll_all)
        self.plot_layout.addWidget(self.time_slider)

        self.outer_layout.addLayout(self.plot_layout)


        # Set zoom / display range for plot views
        self.display_range = qw.QLineEdit()
        self.display_range.setValidator(
                qg.QDoubleValidator(self.minimum_duration, 10000.0, 4)
                )
        self.display_range.textChanged.connect(self.set_display_range)
        self.display_range.setText(str(self.display_duration))

        # Increment / Decrement zoom
        plus = qw.QPushButton('Zoom Out')
        plus.clicked.connect(self.increment_display_range)
        minus = qw.QPushButton('Zoom In')
        minus.clicked.connect(self.decrement_display_range)

        range_layout = qw.QHBoxLayout()
        [range_layout.addWidget(w) for w in [self.display_range, plus, minus]]
        self.outer_layout.addLayout(range_layout)


        # control buttons
        qbtn = qw.QPushButton('Quit', self)
        qbtn.clicked.connect(self.close)

        qbtn2 = qw.QPushButton('Test', self)
        qbtn2.clicked.connect(self.print_signals)

        control_layout = qw.QHBoxLayout()
        control_layout.addWidget(qbtn)
        control_layout.addWidget(qbtn2)
        self.outer_layout.addLayout(control_layout)

        self.main_widget.setLayout(self.outer_layout)
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)
        self.statusBar().showMessage("Welcome to the NEMS recording browser", 2000)

    # Plot Window adjusters
    #@show_exceptions('bool')
    def scroll_all(self):
        self.start_time = self.time_slider.value()
        self.stop_time = self.start_time + self.display_duration

        # don't go past the latest time of the biggest plot
        # (should all have the same max most of the time)
        self._update_max_time()
        if self.stop_time >= self.max_time:
            self.stop_time = self.max_time
            self.start_time = max(0, self.max_time - self.display_duration)

        [p.update_figure() for p in self.plot_list]

    def _update_max_time(self):
        self.max_time = max([p.max_time for p in self.plot_list])

    #@show_exceptions('bool')
    def set_display_range(self):
        duration = float(self.display_range.text())
        if not duration:
            print("Duration not set to a valid value. Please enter a"
                  "a number > 0")
            return
        self.display_duration = duration
        self._update_range()

    #@show_exceptions('bool')
    def increment_display_range(self):
        self.display_duration += 1
        self.display_range.setText(str(self.display_duration))
        self._update_range()

    #@show_exceptions('bool')
    def decrement_display_range(self):
        self.display_duration -= 1
        self.display_range.setText(str(self.display_duration))
        self._update_range()

    def _update_range(self):
        self.time_slider.setRange(0, self.max_time-self.display_duration)
        self.scroll_all()

    # Add / Remove plots

    #@show_exceptions('bool')
    def print_signals(self):
        '''
        For testing/development, verify plot_list, layout and signals
        match up.
        '''
        for s in self.signals:
            print("signal: %s" % s)
        for p in self.plot_list:
            print("plot: %s" % p.signal)

    #@show_exceptions('bool')
    def add_signal(self):
        n = len(self.plot_list)
        valid_signals = [s for s in self.recording.signals
                         if s not in self.signals]
        idx, ok = qw.QInputDialog.getInt(self, "Which plot position?", "Index",
                                         n, 0, n, 1)
        if not ok:
            return
        s, ok = qw.QInputDialog.getItem(self, "Name of the signal to plot?",
                                        "Signals:", valid_signals, 0, False)
        if not ok:
            return

        self.signals.insert(idx, s)
        self.plot_list.insert(idx, self._default_plot_instance(s))
        if idx in [-1, len(self.plot_list)]:
            layout_idx = idx-1
        else:
            layout_idx = idx
        self.plot_layout.insertWidget(layout_idx, self.plot_list[idx])
        self.main_widget.update()
        self.scroll_all()

    #@show_exceptions('bool')
    def remove_signal(self):
        s, ok = qw.QInputDialog.getItem(self, "Name of the signal to remove?",
                                        "Signals:", self.signals, 0, False)
        if not ok:
            return

        idx = self.signals.index(s)
        self.signals.pop(idx)
        self.plot_list.pop(idx)
        self.plot_layout.itemAt(idx).widget().deleteLater()
        self.main_widget.update()
        self.scroll_all()

    #@show_exceptions('bool')
    def _default_plot_instance(self, s):
        return NemsCanvas(self.recording, s, self, self.main_widget,
                          width=self.plot_width, height=self.plot_height,
                          dpi=self.plot_dpi)

    def resp_as_spikes(self):
        pass

    def resp_as_psth(self):
        pass

    @show_exceptions('bool')
    def save_screenshot(self):
        raise ValueError("Screenshots not working quite yet...")
        file, ok = qw.QFileDialog.getSaveFileName(
                self, "Filepath to save screenshot?",
                filter="PNG(*.png);; JPEG(*.jpg)"
                )
        if not ok:
            return

        screenshot = qw.QApplication.primaryScreen().grabWindow(0)
        if file[-3:] == 'png':
            screenshot.save(file, 'png')
        elif file[-3:] == 'jpg':
            screenshot.save(file, 'jpg')

    def closeEvent(self):
        print('closing recording browser')

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

    def about(self):
        qw.QMessageBox.about(self, "About",
  """NEMS recording browser

  Implemented based on embedding_in_qt5.py example (Copyright 2015 BoxControL)
  and
  http://matplotlib.org/examples/user_interfaces/embedding_in_qt4.html
  """
  )


class PandasModel(qc.QAbstractTableModel):
    def __init__(self, df = pd.DataFrame(), parent=None):
        qc.QAbstractTableModel.__init__(self, parent=parent)
        self._df = df

    def headerData(self, section, orientation, role=qc.Qt.DisplayRole):
        if role != qc.Qt.DisplayRole:
            return qc.QVariant()

        if orientation == qc.Qt.Horizontal:
            try:
                return self._df.columns.tolist()[section]
            except (IndexError, ):
                return qc.QVariant()
        elif orientation == qc.Qt.Vertical:
            try:
                # return self.df.index.tolist()
                return self._df.index.tolist()[section]
            except (IndexError, ):
                return qc.QVariant()

    def data(self, index, role=qc.Qt.DisplayRole):
        if role != qc.Qt.DisplayRole:
            return qc.QVariant()

        if not index.isValid():
            return qc.QVariant()

        return qc.QVariant(str(self._df.ix[index.row(), index.column()]))

    def setData(self, index, value, role):
        row = self._df.index[index.row()]
        col = self._df.columns[index.column()]
        if hasattr(value, 'toPyObject'):
            # PyQt4 gets a QVariant
            value = value.toPyObject()
        else:
            # PySide gets an unicode
            dtype = self._df[col].dtype
            if dtype != object:
                value = None if value == '' else dtype.type(value)
        self._df.set_value(row, col, value)
        return True

    def rowCount(self, parent=qc.QModelIndex()):
        return len(self._df.index)

    def columnCount(self, parent=qc.QModelIndex()):
        return len(self._df.columns)

    def sort(self, column, order):
        colname = self._df.columns.tolist()[column]
        self.layoutAboutToBeChanged.emit()
        self._df.sort_values(colname, ascending= order == qc.Qt.AscendingOrder, inplace=True)
        self._df.reset_index(inplace=True, drop=True)
        self.layoutChanged.emit()



def pandas_table_test():

    data = {'a': [1, 2, 3], 'b': ['dog','cat','ferret']}
    df = pd.DataFrame.from_dict(data)
    w = qw.QWidget()

    def loadFile(self):
        fileName, _ = qw.QFileDialog.getOpenFileName(w, "Open File", "", "CSV Files (*.csv)");
        pathLE.setText(fileName)
        df = pd.read_csv(fileName)
        model = PandasModel(df)
        pandasTv.setModel(model)


    hLayout = qw.QHBoxLayout()
    pathLE = qw.QLineEdit(w)
    hLayout.addWidget(pathLE)
    loadBtn = qw.QPushButton("Select File", w)
    hLayout.addWidget(loadBtn)
    loadBtn.clicked.connect(loadFile)

    vLayout = qw.QVBoxLayout(w)
    vLayout.addLayout(hLayout)

    pandasTv = qw.QTableView()
    model = PandasModel(df)
    pandasTv.setModel(model)
    vLayout.addWidget(pandasTv)

    w.show()
    w.raise_()
    return w


def browse_recording(rec, signals=['stim', 'resp'], cellid=None,
                     modelname=None):
    aw = ApplicationWindow(recording=rec, signals=signals,
                           cellid=cellid, modelname=modelname)
    _window_startup(aw)

    return aw


def browse_context(ctx, rec='val', signals=['stim', 'resp'], rec_idx=0):
    rec = ctx[rec]
    if isinstance(rec, list):
        rec = rec[rec_idx]
    meta = ctx['modelspecs'][0][0]['meta']
    cellid = meta.get('cellid', None)
    modelname = meta.get('modelname', None)

    aw = browse_recording(rec, signals, cellid, modelname)

    return aw


def _window_startup(aw):
    aw.setWindowTitle("NEMS data browser")
    aw.show()
    aw.scroll_all()

    aw.raise_()

    return aw


#batch = 289
#modelname = "ozgf.fs100.ch18-ld-sev_dlog-wc.18x2.g-fir.2x15-lvl.1-dexp.1_init-basic"
#cellid = 'TAR010c-21-4'
#xf, ctx = nw.load_model_baphy_xform(cellid, batch, modelname, eval_model=True,
#                                    only=0)
#
#browse_context(ctx, rec='val', signals=['stim', 'resp'], rec_idx=0)
