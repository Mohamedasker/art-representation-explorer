"""
Scatter-plot visualiser for 2D and 3D projections.

Produces both:
  * A static **matplotlib** PNG (publication quality, no browser required).
  * An interactive **Plotly** HTML file (hover to see image path and label).

The ``colorvar`` argument is a dict produced by any of the
``src.coloring.*`` modules — it provides the per-sample values, colormap
name, axis label, and whether the variable is categorical.

Design goals:
  * Completely model-agnostic — any (N, 2) or (N, 3) projection works.
  * Any colorvar with values of length N works.
  * No hard-coded dataset specifics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

# ── Helpers ───────────────────────────────────────────────────────────────────

def _assert_compatible(coords: np.ndarray, colorvar: dict) -> None:
    n = len(coords)
    if len(colorvar["values"]) != n:
        raise ValueError(
            f"colorvar has {len(colorvar['values'])} entries but coords has {n} rows."
        )


def _normalise_coords(coords: np.ndarray) -> np.ndarray:
    """Scale coords to roughly [0, 1] for consistent axis ranges."""
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    rng = np.where(hi - lo > 1e-8, hi - lo, 1.0)
    return (coords - lo) / rng


def _categorical_colors(values: np.ndarray, cmap_name: str = "tab10"):
    """Map integer category indices to RGBA colours via a ListedColormap."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    unique = np.unique(values)
    n = len(unique)
    base_cmap = plt.get_cmap(cmap_name, n)
    color_map = {int(u): base_cmap(i) for i, u in enumerate(unique)}
    rgba = np.array([color_map[int(v)] for v in values])
    return rgba, base_cmap, unique


def _continuous_colors(values: np.ndarray, cmap_name: str = "viridis"):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    norm = mcolors.Normalize(vmin=np.nanmin(values), vmax=np.nanmax(values))
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(norm(values))
    return rgba, cmap, norm


# ── Matplotlib static plot ────────────────────────────────────────────────────

def plot_matplotlib(
    coords: np.ndarray,
    colorvar: dict,
    title: str = "",
    out_path: Optional[str | Path] = None,
    figsize: tuple[int, int] = (12, 10),
    point_size: float = 8.0,
    alpha: float = 0.7,
    show: bool = False,
    dpi: int = 150,
) -> Path | None:
    """
    Create a static matplotlib scatter plot.

    Parameters
    ----------
    coords:
        Array (N, 2) or (N, 3) of projection coordinates.
    colorvar:
        Dict from any ``src.coloring.*`` function.
    title:
        Plot title.
    out_path:
        Save path for the PNG file.  If ``None``, the plot is not saved.
    figsize / point_size / alpha / dpi:
        Standard matplotlib parameters.
    show:
        Call ``plt.show()`` (useful in notebooks).

    Returns
    -------
    Path | None
        Path of the saved PNG, or ``None`` if not saved.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    _assert_compatible(coords, colorvar)
    is_3d = coords.shape[1] == 3
    is_cat = colorvar.get("is_categorical", False)
    values = colorvar["values"]
    names = colorvar.get("names", values.astype(str))
    cmap_name = colorvar.get("colormap", "tab10" if is_cat else "viridis")
    cb_label = colorvar.get("label", "")

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d") if is_3d else fig.add_subplot(111)

    if is_cat:
        rgba, cmap, unique = _categorical_colors(values, cmap_name)
        # plot each category separately to get a proper legend
        unique_names = [names[values == u][0] if np.any(values == u) else str(u) for u in unique]
        for u, uname in zip(unique, unique_names):
            mask = values == u
            c = rgba[mask]
            xy = coords[mask]
            scatter_kwargs = dict(s=point_size, alpha=alpha, label=uname, c=c)
            if is_3d:
                ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], **scatter_kwargs)
            else:
                ax.scatter(xy[:, 0], xy[:, 1], **scatter_kwargs)
        ax.legend(
            loc="upper right",
            fontsize=8,
            markerscale=2,
            framealpha=0.8,
            ncol=max(1, len(unique) // 12),
        )
    else:
        rgba, cmap, norm = _continuous_colors(values, cmap_name)
        scatter_kwargs = dict(s=point_size, alpha=alpha, c=values,
                              cmap=cmap, norm=norm)
        sc = ax.scatter(*([coords[:, i] for i in range(coords.shape[1])]),
                        **scatter_kwargs)
        plt.colorbar(sc, ax=ax, label=cb_label, shrink=0.7)

    ax.set_title(title or cb_label, fontsize=14)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    if is_3d:
        ax.set_zticks([])

    plt.tight_layout()

    saved: Path | None = None
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix not in (".png", ".pdf", ".svg"):
            out_path = out_path.with_suffix(".png")
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        saved = out_path
        print(f"  Saved  {out_path}")

    if show:
        plt.show()
    plt.close(fig)
    return saved


# ── Plotly interactive plot ───────────────────────────────────────────────────

def plot_plotly(
    coords: np.ndarray,
    colorvar: dict,
    paths: Optional[np.ndarray] = None,
    title: str = "",
    out_path: Optional[str | Path] = None,
    point_size: int = 4,
    opacity: float = 0.8,
    width: int = 1000,
    height: int = 800,
) -> Path | None:
    """
    Create an interactive Plotly scatter plot saved as a self-contained HTML.

    Hovering over a point shows its colour-variable value, class name, and
    optionally its image path.

    Parameters
    ----------
    coords:
        Array (N, 2) or (N, 3) of projection coordinates.
    colorvar:
        Dict from any ``src.coloring.*`` function.
    paths:
        Optional string array (N,) of image paths shown on hover.
    title:
        Plot title.
    out_path:
        Save path for the HTML file.
    point_size / opacity / width / height:
        Plotly trace parameters.

    Returns
    -------
    Path | None
    """
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        import pandas as pd
    except ImportError:
        raise ImportError(
            "plotly and pandas are required for interactive plots.\n"
            "Install with:  pip install plotly pandas"
        )

    _assert_compatible(coords, colorvar)
    is_3d = coords.shape[1] == 3
    is_cat = colorvar.get("is_categorical", False)
    values = colorvar["values"]
    names = colorvar.get("names", np.array(values, dtype=str))
    cb_label = colorvar.get("label", "value")
    cmap_name = colorvar.get("colormap", "tab10" if is_cat else "viridis")

    # Build dataframe
    df_dict: dict = {
        "x": coords[:, 0],
        "y": coords[:, 1],
        "color_value": values,
        "color_name": names,
    }
    if is_3d:
        df_dict["z"] = coords[:, 2]
    if paths is not None:
        df_dict["path"] = [str(p).split("/")[-1] for p in paths]

    df = pd.DataFrame(df_dict)

    hover_data = {"color_name": True, "color_value": ":.3f" if not is_cat else True}
    if "path" in df:
        hover_data["path"] = True

    # Convert colormap
    plotly_cmap = _mpl_to_plotly_cmap(cmap_name, is_cat, values)

    scatter_fn = px.scatter_3d if is_3d else px.scatter
    fig = scatter_fn(
        df,
        x="x",
        y="y",
        z="z" if is_3d else None,
        color="color_name" if is_cat else "color_value",
        color_discrete_sequence=plotly_cmap if is_cat else None,
        color_continuous_scale=plotly_cmap if not is_cat else None,
        hover_data=hover_data,
        title=title or cb_label,
        labels={"color_value": cb_label, "color_name": cb_label},
        opacity=opacity,
    )

    fig.update_traces(marker=dict(size=point_size))
    fig.update_layout(
        width=width,
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial, sans-serif", size=12),
        legend=dict(
            title=cb_label,
            itemsizing="constant",
            tracegroupgap=2,
        ),
    )
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

    saved: Path | None = None
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix != ".html":
            out_path = out_path.with_suffix(".html")
        fig.write_html(str(out_path), include_plotlyjs="cdn")
        saved = out_path
        print(f"  Saved  {out_path}")

    return saved


def _mpl_to_plotly_cmap(mpl_name: str, is_cat: bool, values: np.ndarray):
    """
    Best-effort conversion from matplotlib colormap name to Plotly equivalent.
    For categorical, returns a list of hex colour strings.
    For continuous, returns a Plotly built-in name when possible.
    """
    continuous_map = {
        "viridis": "Viridis",
        "plasma": "Plasma",
        "magma": "Magma",
        "inferno": "Inferno",
        "cividis": "Cividis",
        "turbo": "Turbo",
        "rainbow": "Rainbow",
        "hsv": "HSV",
    }
    if not is_cat:
        return continuous_map.get(mpl_name, "Viridis")

    # categorical — sample the mpl colormap
    import matplotlib.pyplot as plt
    unique = np.unique(values)
    cmap = plt.get_cmap(mpl_name, len(unique))
    return [
        "#{:02x}{:02x}{:02x}".format(
            int(cmap(i)[0] * 255),
            int(cmap(i)[1] * 255),
            int(cmap(i)[2] * 255),
        )
        for i in range(len(unique))
    ]


# ── Combined convenience function ─────────────────────────────────────────────

def visualize(
    coords: np.ndarray,
    colorvar: dict,
    paths: Optional[np.ndarray] = None,
    title: str = "",
    out_dir: str | Path = "visualizations",
    stem: str = "scatter",
    static: bool = True,
    interactive: bool = True,
    **kwargs,
) -> dict[str, Path | None]:
    """
    Produce both a static PNG and an interactive HTML scatter plot.

    Parameters
    ----------
    coords:
        Projection coordinates (N, 2) or (N, 3).
    colorvar:
        Color variable dict.
    paths:
        Image paths for hover (Plotly only).
    title:
        Plot title.
    out_dir:
        Output directory.
    stem:
        File stem used for both PNG and HTML filenames.
    static / interactive:
        Whether to produce the respective format.

    Returns
    -------
    dict with keys ``"png"`` and ``"html"`` mapping to saved paths.
    """
    out_dir = Path(out_dir)
    results: dict[str, Path | None] = {"png": None, "html": None}

    if static:
        results["png"] = plot_matplotlib(
            coords,
            colorvar,
            title=title,
            out_path=out_dir / f"{stem}.png",
            **{k: v for k, v in kwargs.items()
               if k in ("figsize", "point_size", "alpha", "show", "dpi")},
        )

    if interactive:
        results["html"] = plot_plotly(
            coords,
            colorvar,
            paths=paths,
            title=title,
            out_path=out_dir / f"{stem}.html",
            **{k: v for k, v in kwargs.items()
               if k in ("point_size", "opacity", "width", "height")},
        )

    return results
