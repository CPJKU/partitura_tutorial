import ipywidgets as wg
import os
import glob
import io

from typing import Iterable, Union

from IPython.display import Image, display

from partitura.utils.misc import PathLike

from PIL.Image import Image as PILImage
import zipfile
import requests
from urllib.request import urlopen

try:
    import google.colab

    IN_COLAB = True
except ImportError:
    IN_COLAB = False


if IN_COLAB:
    r = requests.get(
        "https://raw.githubusercontent.com/CPJKU/partitura_tutorial/"
        "main/notebooks/02_alignment/figures/dtw_example_png.zip",
        stream=True,
    )
    archive = zipfile.ZipFile(io.BytesIO(r.content), "r")

else:

    archive = zipfile.ZipFile(os.path.join("figures", "dtw_example_png.zip"), "r")

PNG_DTW_EXAMPLE = [
    io.BytesIO(archive.read(f"dtw_example_{i:02d}.png")) for i in range(30)
]


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

    else:

        def show_image(slider_val: int) -> None:
            return Image(image_list[slider_val].getvalue())

    wg.interact(
        show_image,
        slider_val=wg.IntSlider(
            min=0,
            max=len(image_list) - 1,
            step=1,
        ),
    )


# def slideshow_from_pdf(pdf_path: PathLike) -> None:
#     images = pdf2image.convert_from_path(pdf_path)
#     slideshow(images)


# def show_slideshow_or_gif(interactive: bool) -> None:
#     from urllib.request import urlopen

#     pdf_path = (
#         "https://raw.githubusercontent.com/CPJKU/partitura_tutorial/main/notebooks/02_alignment/figures/dtw_example.pdf",
#     )
#     gif_path = "https://raw.githubusercontent.com/CPJKU/partitura_tutorial/main/notebooks/02_alignment/figures/dtw_example.gif"
#     gif_data = urlopen(gif_path)
#     if interactive:
#         slideshow_from_pdf(pdf_path)
#     else:
#         display(Image(data=gif_data.read(), format="png"))


def dtw_example(interactive: bool) -> None:

    gif_path = "https://raw.githubusercontent.com/CPJKU/partitura_tutorial/main/notebooks/02_alignment/figures/dtw_example.gif"
    gif_data = urlopen(gif_path)
    if interactive:
        slideshow(PNG_DTW_EXAMPLE)
    else:
        display(Image(data=gif_data.read(), format="png"))
