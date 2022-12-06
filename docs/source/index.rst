.. Partitura Tutorial documentation master file, created by
   sphinx-quickstart on Thu Nov 10 11:47:15 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: ./assets/partitura_logo_black.png
   :width: 600
   :align: center

Welcome to the Partitura Tutorial ISMIR 2022!
==============================================

Symbolic music formats (e.g., MIDI, MusicXML/MEI) can provide a variety of high-level musical information like note pitch and duration, key/time signature, beat/downbeat position, etc. Such data can be used as both input/training data and as ground truth for MIR systems. `Here <https://github.com/CPJKU/partitura_tutorial/blob/main/Partitura%20tutorial%20introduction.pdf>`_ you can find an introduction to symbolic music, with a presentation of the typical symbolic data types.

This tutorial aims to provide an introduction to symbolic music processing for a broad MIR audience, with a particular focus on showing how to extract relevant MIR features from symbolic musical formats in a fast, intuitive, and scalable way. We do this with the aid of the Python package Partitura. To target different kinds of symbolic data, we use an extended version of the ASAP Dataset, a multi-modal dataset that contains MusicXML scores, MIDI performances, audio performances, and score-to-performance alignments.

This tutorial is an ensemble of notebooks that will guide you
through the basics of the Partitura library.
The tutorial is divided into four parts:

1. An introduction to the Partitura library with all basic I/O functionality and
  the basic data structures.
2. A tutorial on how to use the Partitura library to perform automatic alignment between performances and their respecitve scores.
3. How to implement a Pitch spelling model using Partitura.
4. How to create a Transformer Based Beat Generator using Partitura.

The motivation behind this tutorial is to promote research on symbolic music processing in the MIR community. Therefore, we target a broad audience of researchers without requiring prior knowledge of this particular area. For the hands-on parts of the tutorial, we presuppose some practical experience with the Python language, but we will provide well-documented step-by-step access to the code in the form of Google Colab notebooks, which will be made publicly available after the tutorial. Furthermore, some familiarity with the basic concepts of statistics and machine learning is useful.


This work receives funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme, grant agreement No 101019375 (Whither Music?).


About the Authors
==============================================

`Carlos Cancino-Chacón <http://www.carloscancinochacon.com/>`_ is an Assistant Professor at the
Institute of Computational Perception,
Johannes Kepler University, Linz, Austria,
and a Guest Researcher at the RITMO Centre for
Interdisciplinary Studies in Rhythm, Time and Motion,
University of Oslo, Norway. His research focuses on
studying expressive music performance, music cognition,
and music theory with machine learning methods.
He received a doctoral degree in Computer Science at the
Institute of Computational Perception of the Johannes Kepler
University Linz, a M.Sc. degree in Electrical Engineering and
Audio Engineering from the Graz University of Technology,
a degree in Physics from the National Autonomous University
of Mexico, and a degree in Piano Performance from the National
Conservatory of Music of Mexico.

`Francesco Foscarin <https://www.jku.at/en/institute-of-computational-perception/about-us/people/franceso-foscarin/>`_ is a postdoctoral researcher at the Institute of Computational Perception, Johannes Kepler University, Linz, Austria. He completed his Ph.D. at CNAM Paris on music transcription, with a focus on the production of musical scores, and holds classical and jazz piano degrees from the Conservatory of Vicenza. His research interests include post-hoc explainability techniques for DL models, grammar-based parsing of hierarchical chord structures, piano comping generation for jazz music, and voice separation in symbolic music.

`Emmanouil Karystinaios <https://emmanouil-karystinaios.github.io/>`_ is a Ph.D. student at the Institute of Computational Perception, Johannes Kepler University, Linz, Austria. His research topics encompass graph neural networks, music structure segmentation, and automated music analysis. He holds an M.Sc. degree in Mathematical Logic from Paris Diderot University, an M.A. in Composition from Paris Vincennes University, and an integrated M.A. in Musicology from the Aristotle University of Thessaloniki.

`Silvan David Peter <https://www.jku.at/en/institute-of-computational-perception/about-us/people/silvan-david-peter/>`_ is a University Assistant at the Institute of Computational Perception, Johannes Kepler University, Linz, Austria. His research interests are the evaluation of and interaction with computational models of musical skills. He holds an M.Sc. degree in Mathematics from the Humboldt University of Berlin.


To reference this tutorial
==========================

.. code-block:: BibTeX

   @book{partitura_tutorial:book,
       Author = {Carlos Cancino-Chac{\'o}n and Francesco Foscarin and Emmanouil Karystinaios and Silvan David Peter},
       Month = Dec.,
       Publisher = {https://cpjku.github.io/partitura_tutorial},
       Title = {An Introduction to Symbolic Music Processing in Python with Partitura},
       Year = 2022,
       Url = {https://cpjku.github.io/partitura_tutorial}
   }


.. toctree::
   :glob:
   :titlesonly:
   :numbered:
   :hidden:

   ./index
   notebooks/01_introduction/Partitura_tutorial.ipynb
   notebooks/02_alignment/Symbolic_Music_Alignment.ipynb
   notebooks/03_mlflow/pitch_spelling.ipynb
   notebooks/04_generation/Drum_Generation_Transformer.ipynb

