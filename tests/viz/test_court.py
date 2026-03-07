import matplotlib.pyplot as plt

from nbafigs.viz.court import _get_texture_path, draw_court, make_shot_fig


def test_draw_court_returns_list_of_patches():
    elements = draw_court()

    assert isinstance(elements, list)
    assert len(elements) == 13


def test_draw_court_custom_params():
    elements = draw_court(color='red', lw=3)

    assert len(elements) == 13


def test_get_texture_path_exists():
    path = _get_texture_path()

    assert path is not None


def test_make_shot_fig_returns_fig_and_ax():
    fig, ax = make_shot_fig(title='Test', title_size=22, context=None, context_size=14)

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


def test_make_shot_fig_with_context():
    fig, _ = make_shot_fig(
        title='Test',
        title_size=22,
        context='Some context\nwith newline',
        context_size=14,
    )

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_make_shot_fig_no_title():
    fig, _ = make_shot_fig(title=None, title_size=22, context=None, context_size=14)

    assert isinstance(fig, plt.Figure)
    plt.close(fig)
