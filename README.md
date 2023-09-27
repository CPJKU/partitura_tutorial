# Partitura Tutorials

Welcome to a quick introduction to symbolic music processing with partitura!

This repo was originally developed for a tutorial session at ISMIR 22 and is now maintained as hands-on documentation for the [partitura library](https://github.com/CPJKU/partitura).
Partitura is a library for symbolic music processing, that is, it reads, writes, and manipulates musical scores in a variety of formats and MIDI files and aims at:
- file I/O for all things symbolic music; scores, performances, and alignments of the two.
- feature extraction: creating note arrays, piano rolls, and custom features.
- minimal reference implementations of automatic music analysis tools.

### What's in the tutorials?

In the directory `notebooks` there are four tutorials:
- 01_introduction: how to read and write scores and performances in partitura, how to manipulate musical material, and how to extract features like note arrays and piano rolls.
- 02_alignment: how to read, process, and create symbolic (note-to-note) music alignments with partitura and Daynamic Time Warping (fastdtw, Vienna4x22 dataset)
- 03_mlflow: automatic pitch spelling as an example of a machine learning pipeline for automatic music analysis (pytorch, lightning, LSTM, ASAP dataset)
- 04_generation: a small drum beat generator as an example of a machine learning pipeline for automatic music generation (pytorch, Transformer Encoder, Groove Midi Dataset)

### How to use the tutorials?

The tutorials consist of jupyter notebooks for you to run **locally** or on **google colab** (by clicking the link at the top of each notebook).
To run the notebooks locally, be sure to install the dependencies (or create an environment) based on `environment.yml`.
We aim to keep this tutorials updated with the current **version** of partitura, for prior versions of partitura, check the versions on this github repository.

### More information?

The partitura **documentation** and API reference is on [readthedocs](https://partitura.readthedocs.io/en/latest/).
Do you have **questions?** You're welcome on the [partitura discussions](https://github.com/CPJKU/partitura/discussions).
Encountered a **bug?** You're welcome to raise an [issue](https://github.com/CPJKU/partitura/issues).



### Partitura Installation

The easiest way to install the partitura package is via ``pip`` from the `PyPI (Python
Package Index) <https://pypi.python.org/pypi>`_::

```shell
  pip install partitura
```

This will install the latest release of the package and will install all
dependencies automatically.



**To install latest stable version:**

```shell
pip install git+https://github.com/CPJKU/partitura.git@develop
```



### Partitura QuickStart

The following code loads the contents of an example MusicXML file included in
the package:

```python
import partitura
my_xml_file = partitura.EXAMPLE_MUSICXML
score = partitura.load_musicxml(my_xml_file)
```

For **MusicXML**, **Kern**, or **MEI** files do:

```python
import partitura
score = partitura.load_score(my_file)
```

### Citation

If you find Partitura useful, we would appreciate it if you could cite us!

```
@inproceedings{partitura_mec,
  title={{Partitura: A Python Package for Symbolic Music Processing}},
  author={Cancino-Chac\'{o}n, Carlos Eduardo and Peter, Silvan David and Karystinaios, Emmanouil and Foscarin, Francesco and Grachten, Maarten and Widmer, Gerhard},
  booktitle={{Proceedings of the Music Encoding Conference (MEC2022)}},
  address={Halifax, Canada},
  year={2022}
}
```

