from __future__ import annotations

import warnings
from numbers import Number
from typing import Optional, Tuple, List, Union

import matplotlib.pyplot as plt
import numpy as np

__all__ = ['Figure']

_DataType = Union[np.ndarray, List, Number]
_StrSeq = List[str]


def _to_numpy(data: Optional[_DataType]) -> Optional[np.ndarray]:
    if data is None:
        return data

    if isinstance(data, int or float):
        data = [data]
    return np.array(data)


def _is_notebook():
    import sys

    return "ipykernel" in sys.modules


def _is_display_available():
    import os

    return os.getenv('DISPLAY') is not None


class Figure(object):
    _default_tick_params = dict(direction="in",
                                grid_alpha=0.5,
                                grid_linestyle="--",
                                labelsize="large")

    def __init__(self,
                 figsize: Tuple[int, int],
                 boxes: Optional[Tuple[int, int]] = None):
        """ Create Figure with size of `figsize` and boxes of `boxes` ::

        >>> fig = Figure((5, 5))
        >>> fig.bar(...)

        """

        self._figure = plt.figure(figsize=figsize)
        self._ax_pos = 0
        self._boxes = (1, 1) if boxes is None else boxes
        self._num_boxes = np.prod(self._boxes)
        self._ax: Optional[plt.Axes] = None
        # initialize
        self.next()

    # global APIs
    @property
    def fig(self) -> plt.Figure:
        """ Direct access to `fig` ::

        """

        return self._figure

    def save(self,
             path: str,
             dpi: int,
             no_tight_layout: bool = False) -> None:
        """ Save figure to `path` with given `dpi`
        """

        if not no_tight_layout:
            self.fig.tight_layout()
        self.fig.savefig(path, dpi=dpi)

    def show(self):
        if _is_notebook():
            return self.fig

        if _is_display_available():
            self.fig.show()
        else:
            warnings.warn('Display is not available, so this is noop')

    def global_legend(self,
                      loc: Optional[str] = None,
                      fontsize: Optional[int] = None) -> Figure:
        self.fig.legend(loc=loc, fontsize=fontsize)
        return self

    # APIs for each box (ax)
    @property
    def ax(self) -> plt.Axes:
        """ Direct access to `ax`
        """

        return self._ax

    def next(self) -> Figure:
        """ Increment the current box (ax)
        """

        if self._ax_pos == self._num_boxes:
            # note that __init__ calls next(self)
            warnings.warn('All boxes are already occupied, so `self.next()` cannot be used!')
        else:
            self._ax_pos += 1
            self._ax = self.fig.add_subplot(*self._boxes, self._ax_pos)
            self._ax.tick_params('both', **self._default_tick_params)
        return self

    def _annotate_bar(self,
                      bar: list,
                      offset: float,
                      fontsize: int,
                      alpha: float) -> None:
        for b in bar:
            height = b.get_height()
            self.ax.annotate(str(height),
                             xy=(b.get_x() + b.get_width() / 2, height + offset),
                             ha='center',
                             fontsize=fontsize,
                             alpha=alpha)

    def bar(self,
            *_data,
            width: Optional[float] = None,
            colors: Optional[List[str]] = None,
            labels: Optional[List[str]] = None,
            alpha: Optional[float] = None,
            add_annotate: bool = False,
            annotate_offset: float = 0.01,
            annotate_fontsize: Optional[int] = None,
            annotate_alpha: Optional[float] = None) -> Figure:
        """ Vertical bars

        """

        _data = [_to_numpy(d) for d in _data]
        num_types = len(_data)
        indices = np.arange(_to_numpy(_data[0]).shape[-1])
        width = 1 / (1 + num_types) if width is None else width

        if num_types > 1:
            assert all([_data[0].shape == d.shape for d in _data])

        if colors is None:
            colors = [f'C{i}' for i in range(num_types)]
        else:
            assert len(colors) == num_types

        if labels is None:
            labels = [None for _ in range(num_types)]
        else:
            assert len(labels) == num_types

        for i, d in enumerate(_data):
            std = None
            if d.ndim == 2:
                std = d.std(axis=0)
                d = d.mean(axis=0)
            bar = self.ax.bar(indices + i * width, d, align='edge', yerr=std,
                              width=width, alpha=alpha, color=colors[i],
                              label=labels[i])
            if add_annotate:
                self._annotate_bar(bar, annotate_offset, annotate_fontsize, annotate_alpha)

        # do not show ticks on xaxis
        self.set_ticks(x_tick_params=dict(length=0, **self._default_tick_params))
        return self

    def plot(self,
             *_data,
             color: Optional[str] = None,
             alpha: Optional[float] = None,
             fill_alpha: Optional[float] = None,
             label: Optional[str] = None,
             linestyle: Optional[str] = None,
             linewidth: Optional[float] = None,
             **kwargs) -> Figure:
        if len(_data) == 1:
            y = _to_numpy(_data[0])
            x = np.arange(y.shape[-1])
        elif len(_data) == 2:
            x, y = _data
            x, y = _to_numpy(x), _to_numpy(y)
            assert x.shape == y.shape
        else:
            raise ValueError('Too many data to unpack. `_data` expects `y` or `x, y`')

        std = None
        if y.ndim == 2:
            std = y.std(axis=0)
            y = y.mean(axis=0)

        self.ax.plot(x, y, color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha, label=label, **kwargs)
        if std is not None:
            self.ax.fill_between(x, y - std, y + std, facecolor=color, alpha=fill_alpha)
        return self

    def scatter(self,
                x: _DataType,
                y: _DataType,
                size: Optional[_DataType] = None,
                color: Optional[_StrSeq] = None,
                alpha: Optional[float] = None,
                label: Optional[str] = None,
                **kwargs) -> Figure:
        x = _to_numpy(x)
        y = _to_numpy(y)
        assert x.shape == y.shape
        size = _to_numpy(size)
        if size is not None:
            assert x.shape == size.shape
        self.ax.scatter(x, y, s=size, c=color, alpha=alpha, label=label, **kwargs)
        return self

    def set_labels(self,
                   xlabel: Optional[str] = None,
                   ylabel: Optional[str] = None,
                   fontsize: Optional[int] = None) -> Figure:
        """ Set labels ::

        >>> fig.set_labels('Epochs', 'Accuracy', fontsize=18)
        """

        if xlabel is not None:
            self.ax.set_xlabel(xlabel, fontsize=fontsize)
        if ylabel is not None:
            self.ax.set_xlabel(ylabel, fontsize=fontsize)
        return self

    def set_ticks(self,
                  xticks: Optional[List[float, str]] = None,
                  yticks: Optional[List[float, str]] = None,
                  fontsize: Optional[int] = None,
                  x_tick_params: Optional[dict] = None,
                  y_tick_params: Optional[dict] = None) -> Figure:
        if xticks is not None:
            if isinstance(xticks[0], Number):
                self.ax.set_xticks(xticks)
            self.ax.set_xticklabels(xticks, fontdict=dict(fontsize=fontsize))

        if yticks is not None:
            if isinstance(yticks[0], Number):
                self.ax.set_yticks(yticks)
            self.ax.set_yticklabels(yticks, fontdict=dict(fontsize=fontsize))

        if (x_tick_params is not None) and (y_tick_params is not None) and (x_tick_params == y_tick_params):
            self.ax.tick_params('both', **x_tick_params)

        else:
            if x_tick_params is not None:
                self.ax.tick_params('x', **x_tick_params)
            if y_tick_params is not None:
                self.ax.tick_params('y', **y_tick_params)

        return self

    def set_limits(self,
                   xlim: Optional[Tuple[float, float]] = None,
                   ylim: Optional[Tuple[float, float]] = None) -> Figure:
        """ Set xlim and ylim ::

        >>> fig.set_limits((0, 1))
        """

        if xlim is not None:
            self.ax.set_xlim(*xlim)
        if ylim is not None:
            self.ax.set_ylim(*ylim)
        return self

    def add_grid(self) -> Figure:
        self.ax.grid()
        return self

    def legend(self,
               loc: Optional[int] = None,
               fontsize: Optional[int] = None,
               title: Optional[str] = None,
               title_fontsize: Optional[int] = None) -> Figure:
        self.ax.legend(loc=loc, fontsize=fontsize, title=title, title_fontsize=title_fontsize)
        return self
