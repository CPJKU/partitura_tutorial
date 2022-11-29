import ipywidgets as wg
import os
import glob

from typing import Iterable, Union

from IPython.display import Image, display

from partitura.utils.misc import PathLike

import pdf2image

from PIL.Image import Image as PILImage


def slideshow(image_list: Iterable[Union[PathLike, PILImage]]) -> None:
    """
    An interactive widget to display a slideshow in a Jupyter notebook

    Parameters
    ----------
    image_list : Iterable[PathLike]
        List of images to show in the slideshow.
    """

    if isinstance(image_list[0], str):

        def show_image(slider_val: int) -> Image:
            return Image(image_list[slider_val])

    elif isinstance(image_list[0], PILImage):

        def show_image(slider_val: int) -> None:
            display(image_list[slider_val])

    wg.interact(
        show_image,
        slider_val=wg.IntSlider(
            min=0,
            max=len(image_list) - 1,
            step=1,
        ),
    )


def slideshow_from_pdf(pdf_path: PathLike) -> None:
    images = pdf2image.convert_from_path(pdf_path)
    slideshow(images)


def slideshow_from_dir(image_dir: PathLike) -> None:

    if not os.path.isdir(image_dir):
        raise ValueError("`image_dir` must be a directory")


def show_slideshow_or_gif(
    interactive: bool) -> None:
    from urllib.request import urlopen
    pdf_path = "https://raw.githubusercontent.com/CPJKU/partitura_tutorial/main/notebooks/02_alignment/figures/dtw_example.pdf",
    gif_path = "https://raw.githubusercontent.com/CPJKU/partitura_tutorial/main/notebooks/02_alignment/figures/dtw_example.gif"
    gif_data = urlopen(gif_path)
    if interactive:
        slideshow_from_pdf(pdf_path)
    else:
        display(Image(data=gif_data.read(), format="png"))
