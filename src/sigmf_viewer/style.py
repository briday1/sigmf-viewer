"""Compact Plotly styling local to the standalone viewer."""

from __future__ import annotations

from typing import Any

TEAL = "#087e8b"


def heatmap_grid_color(theme: str) -> str:
    return "rgba(169,189,194,0.13)" if theme == "dark" else "rgba(96,113,125,0.12)"


def style_figure(figure: Any, theme: str, title: str) -> Any:
    dark = theme == "dark"
    grid = "#36515b" if dark else "#dce5e8"
    figure.update_layout(
        template="plotly_dark" if dark else "simple_white",
        paper_bgcolor="#10252d" if dark else "white",
        plot_bgcolor="#10252d" if dark else "white",
        title={
            "text": title,
            "x": 0.01,
            "y": 0.98,
            "xanchor": "left",
            "yanchor": "top",
            "font": {"size": 15},
        },
        margin={"l": 76, "r": 32, "t": 66, "b": 58},
        legend={
            "orientation": "h",
            "x": 0.99,
            "y": 0.98,
            "xanchor": "right",
            "yanchor": "top",
        },
    )
    figure.update_xaxes(
        showgrid=True,
        gridcolor=grid,
        gridwidth=0.5,
        showline=True,
        mirror=True,
        linecolor=grid,
        zeroline=False,
        automargin=False,
    )
    figure.update_yaxes(
        showgrid=True,
        gridcolor=grid,
        gridwidth=0.5,
        showline=True,
        mirror=True,
        linecolor=grid,
        zeroline=False,
        automargin=False,
    )
    return figure


__all__ = ["TEAL", "heatmap_grid_color", "style_figure"]
