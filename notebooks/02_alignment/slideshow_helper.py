import ipywidgets as wg
import os
import glob

from typing import Iterable, Union

from IPython.display import Image, display

from partitura.utils.misc import PathLike

from PIL.Image import Image as PILImage

from urllib.request import urlopen

try:
    import google.colab

    IN_COLAB = True
except:
    IN_COLAB = False


if IN_COLAB:
    PNG_DTW_EXAMPLE = [
        urlopen(
            "https://raw.githubusercontent.com/CPJKU/partitura_tutorial/"
            "main/notebooks/02_alignment/figures/dtw_example_png/"
            "dtw_example_{i:02d}.png"
        )
        for i in range(30)
    ]

else:
    PNG_DTW_EXAMPLE = glob.glob(os.path.join("figures", "dtw_example_png", "*.png"))
    PNG_DTW_EXAMPLE.sort()


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
