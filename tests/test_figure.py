import pytest

from madoka import Figure


@pytest.mark.parametrize('boxes', [(1, 4), (4, 1), (2, 2)])
def test_next(boxes):
    # check if warns or not
    with pytest.warns(None) as warn_list:
        fig = Figure((5, 5), boxes=boxes)
        # fig here is already the first box
        for i in range(4):
            fig.next()

    assert len(warn_list) == 1


def test_plot():
    fig = Figure((5, 5))

    # expect plot
    fig.plot([1, 2])

    # expect fill plot
    fig.plot([[1.0, 2.0, 3.0],
              [1.1, 2.4, 3.6]])

    with pytest.raises(AssertionError):
        fig.plot([1, 2, 3], [1, 2])


def test_bar():
    fig = Figure((5, 5), boxes=(1, 2))
    fig.bar([5, 10, 15])
    fig.next()
    fig.bar([5, 10, 15],
            [1, 2, 4])


def test_scatter():
    fig = Figure((5, 5))
    fig.scatter([1, 2, 3], [2, 3, 4])
    fig.scatter([1, 2, 3], [2, 3, 4], [4, 4, 4])
    with pytest.raises(AssertionError):
        fig.scatter([1, 2, 3], [2, 3, 4], [4, 4])
