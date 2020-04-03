from __future__ import annotations

import warnings
from numbers import Number
from typing import Optional, Tuple, List, Union

import matplotlib.pyplot as plt
import numpy as np

__all__ = ['Figure']

_DataType = Union[np.ndarray, List, Number]


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
             dpi: Optional[int] = None,
             no_tight_layout: bool = False) -> None:
        """ Save figure to `path` with given `dpi`
        """

        if not no_tight_layout:
            self.tight_layout()
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

    def tight_layout(self) -> Figure:
        self.fig.tight_layout()
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

    def _bar(self,
             _data,
             labels,
             colors):
        # for self.bar and self.barh
        _data = [_to_numpy(d) for d in _data]
        num_types = len(_data)
        indices = np.arange(_to_numpy(_data[0]).shape[-1])

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
        return _data, indices, labels, colors, num_types

    def _annotate_bar(self,
                      bar: list,
                      offset: float,
                      fontsize: int,
                      weight: str,
                      alpha: float,
                      format: str,
                      horizontal: bool) -> None:
        # annotate self.bar and self.barh
        if weight is None:
            weight = 'bold'
        shared_kwargs = dict(weight=weight,
                             fontsize=fontsize,
                             alpha=alpha)
        for b in bar:
            height = b.get_height()
            width = b.get_width()
            if horizontal:
                w = self.ax.get_xlim()[1]
                _c = (w - width) / w < 0.1
                _offset = -offset if _c else offset
                _align = 'right' if _c else 'left'
                self.ax.annotate(f"{width:.{format}}",
                                 xy=(width + _offset, b.get_y() + height / 2),
                                 va='center',
                                 ha=_align,
                                 **shared_kwargs)
            else:
                h = self.ax.get_ylim()[1]
                _c = (h - height) / h < 0.1
                _offset = -offset if _c else offset
                _align = 'top' if _c else 'bottom'
                self.ax.annotate(f"{height:.{format}}",
                                 xy=(b.get_x() + width / 2, height + _offset),
                                 va=_align,
                                 ha='center',
                                 **shared_kwargs)

    def bar(self,
            *_data,
            width: Optional[float] = None,
            colors: Optional[List[str]] = None,
            labels: Optional[List[str]] = None,
            tick_labels: Optional[List[str]] = None,
            alpha: Optional[float] = None,
            add_annotate: bool = False,
            annotate_offset: float = 0.1,
            annotate_fontsize: Optional[int] = None,
            annotate_weight: Optional[str] = None,
            annotate_alpha: Optional[float] = None,
            annotate_format: str = '2f',
            err_alpha: Optional[float] = None) -> Figure:
        """ Vertical bars
        """

        _data, indices, labels, colors, num_types = self._bar(_data, labels, colors)
        width = 1 / (1 + num_types) if width is None else width
        bars = []
        for i, d in enumerate(_data):
            std = None
            if d.ndim == 2:
                std = d.std(axis=0)
                d = d.mean(axis=0)
            bar = self.ax.bar(indices + i * width, d, align='edge', yerr=std,
                              width=width, alpha=alpha, color=colors[i],
                              label=labels[i],
                              error_kw=dict(alpha=err_alpha))
            bars.append(bar)
        if add_annotate:
            for bar in bars:
                self._annotate_bar(bar, annotate_offset, annotate_fontsize, annotate_weight,
                                   annotate_alpha, annotate_format, horizontal=False)

        # do not show ticks on xaxis
        self.set_ticks(x_tick_params=dict(length=0, **self._default_tick_params))
        if tick_labels is None:
            tick_labels = [str(i) for i in indices]
        else:
            assert len(tick_labels) == len(indices)

        tick_position = (num_types - 1) / (2 * num_types)
        self.set_ticks(xticks=(indices + tick_position), xtick_labels=tick_labels)
        return self

    def barh(self,
             *_data,
             height: Optional[float] = None,
             colors: Optional[List[str]] = None,
             labels: Optional[List[str]] = None,
             tick_labels: Optional[List[str]] = None,
             alpha: Optional[float] = None,
             add_annotate: bool = False,
             annotate_offset: float = 0.1,
             annotate_fontsize: Optional[int] = None,
             annotate_weight: Optional[str] = None,
             annotate_alpha: Optional[float] = None,
             annotate_format: str = '2f',
             err_alpha: Optional[float] = None) -> Figure:
        """ Horizontal bars

        """

        _data, indices, labels, colors, num_types = self._bar(_data, labels, colors)
        height = 1 / (1 + num_types) if height is None else height
        bars = []
        for i, d in enumerate(_data):
            std = None
            if d.ndim == 2:
                std = d.std(axis=0)
                d = d.mean(axis=0)
            bar = self.ax.barh(indices + i * height, d, align='edge', xerr=std,
                               height=height, alpha=alpha, color=colors[i],
                               label=labels[i],
                               error_kw=dict(alpha=err_alpha))
            bars.append(bar)
        if add_annotate:
            for bar in bars:
                self._annotate_bar(bar, annotate_offset, annotate_fontsize, annotate_weight,
                                   annotate_alpha, annotate_format, horizontal=True)

        # do not show ticks on yaxis
        self.set_ticks(y_tick_params=dict(length=0, **self._default_tick_params))
        if tick_labels is None:
            tick_labels = [str(i) for i in indices]
        else:
            assert len(tick_labels) == len(indices)

        tick_position = (num_types - 1) / (2 * num_types)
        self.set_ticks(yticks=(indices + tick_position), ytick_labels=tick_labels)
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
                color: Optional[List[str]] = None,
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
                  xticks: Optional[_DataType] = None,
                  yticks: Optional[_DataType] = None,
                  xtick_labels: Optional[List[str]] = None,
                  ytick_labels: Optional[List[str]] = None,
                  fontsize: Optional[int] = None,
                  x_tick_params: Optional[dict] = None,
                  y_tick_params: Optional[dict] = None) -> Figure:
        if xticks is not None:
            if isinstance(xticks[0], Number):
                self.ax.set_xticks(xticks)
                if xtick_labels is None:
                    xtick_labels = xticks
        if xtick_labels is not None:
            self.ax.set_xticklabels(xtick_labels, fontdict=dict(fontsize=fontsize))

        if yticks is not None:
            if isinstance(yticks[0], Number):
                self.ax.set_yticks(yticks)
                if ytick_labels is None:
                    ytick_labels = yticks
        if ytick_labels is not None:
            self.ax.set_yticklabels(ytick_labels, fontdict=dict(fontsize=fontsize))

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

    def add_grid(self,
                 which: str = 'major',
                 axis: str = 'both',
                 **grid_properties) -> Figure:
        self.ax.grid(which=which, axis=axis, **grid_properties)
        return self

    def legend(self,
               loc: Optional[int] = None,
               fontsize: Optional[int] = None,
               title: Optional[str] = None,
               title_fontsize: Optional[int] = None) -> Figure:
        self.ax.legend(loc=loc, fontsize=fontsize, title=title, title_fontsize=title_fontsize)
        return self
