from typing import Optional

from IPython.display import HTML, display


def reveal(svg: Optional[str]) -> None:
    """
    Display an SVG string in a Jupyter notebook output cell.
    """
    if not svg:
        return

    # Trim leading/trailing whitespace that models sometimes add
    cleaned = svg.strip()
    display(HTML(cleaned))

