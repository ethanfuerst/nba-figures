from importlib import resources

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Rectangle


def _get_texture_path():
    '''Resolve the bundled court texture image.'''
    return resources.files('nbafigs.data').joinpath('basketball-floor-texture.png')


def draw_court(color='black', lw=2):
    '''Return a list of matplotlib patches representing NBA court elements.'''
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)
    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color, fill=False)
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color, fill=False)
    top_free_throw = Arc(
        (0, 142.5), 120, 120, theta1=0, theta2=180, linewidth=lw, color=color, fill=False
    )
    bottom_free_throw = Arc(
        (0, 142.5), 120, 120, theta1=180, theta2=0, linewidth=lw, color=color, linestyle='dashed'
    )
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw, color=color)
    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw, color=color)
    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw, color=color)
    center_outer_arc = Arc(
        (0, 422.5), 120, 120, theta1=180, theta2=0, linewidth=lw, color=color
    )
    center_inner_arc = Arc(
        (0, 422.5), 40, 40, theta1=180, theta2=0, linewidth=lw, color=color
    )
    outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw, color=color, fill=False)

    court_elements = [
        hoop, backboard, outer_box, inner_box, top_free_throw,
        bottom_free_throw, restricted, corner_three_a,
        corner_three_b, three_arc, center_outer_arc,
        center_inner_arc, outer_lines,
    ]

    return court_elements


def make_shot_fig(title, title_size, context, context_size):
    '''Create the base court figure with texture background.

    Args:
        title: Title text for the figure (None to skip).
        title_size: Font size for the title.
        context: Context text below the figure (None to skip).
        context_size: Font size for the context text.

    Returns:
        Tuple of (fig, ax).
    '''
    background_color = '#d9d9d9'
    fig, ax = plt.subplots(facecolor=background_color, figsize=(10, 10))
    fig.patch.set_facecolor(background_color)
    ax.patch.set_facecolor(background_color)

    court_elements = draw_court()
    for element in court_elements:
        ax.add_patch(element)

    if title is not None:
        plt.title(title, pad=10, fontdict={'fontsize': title_size, 'fontweight': 'semibold'})

    img = plt.imread(str(_get_texture_path()))
    plt.imshow(img, zorder=0, extent=[-275, 275, -50, 425])

    plt.xlim(-250, 250)
    plt.ylim(422.5, -47.5)
    plt.axis(False)

    if context is not None:
        ax.text(
            0, 435 + (context_size * context.count('\n')),
            s=context, fontsize=context_size, ha='center',
        )

    return fig, ax
