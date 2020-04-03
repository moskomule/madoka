# madoka

![](https://github.com/moskomule/madoka/workflows/pytest/badge.svg)

`madoka` is an N(>>1)th wheel of `matplotlib` wrapper.

## Installation

```shell script
pip install -U git+https://github.com/moskomule/madoka
```

## Basic Usage

See [example.ipynb](example.ipynb).

## Why `madoka`?

`madoka` makes drawing figures much easier than the original `matplotlib`, but still keeps its flexibility.

```python
from madoka import Figure
fig = Figure((5, 5))
# easy!
fig.scatter([1, 2, 3], [2, 3, 4]).add_grid().set_labels('accuracy', 'complexity').save('figure.pdf')

# flexible!
fig.fig # matplotlib's Figure object 
fig.ax # matplotlib's Axes object
```

`madoka`'s functionality is based on what I've needed for papers. But, if you have any suggestions, don't hesitate to make issues or PRs!