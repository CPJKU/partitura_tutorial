# Partitura Tutorial
A quick introduction to symbolic music processing with partitura:

## Installation

The easiest way to install the package is via ``pip`` from the `PyPI (Python
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



## QuickStart

The following code loads the contents of an example MusicXML file included in
the package:

```python
import partitura
my_xml_file = partitura.EXAMPLE_MUSICXML
part = partitura.load_musicxml(my_xml_file)
```

### Import other formats

For **MusicXML** files do:

```python
import partitura
my_xml_file = partitura.EXAMPLE_MUSICXML
part = partitura.load_musicxml(my_xml_file)
```



For **Kern** files do:

```python
import partitura
my_kern_file = partitura.EXAMPLE_KERN
part = partitura.load_kern(my_kern_file)
```

For **MEI** files do:

```python
import partitura
my_mei_file = partitura.EXAMPLE_MEI
part = partitura.load_mei(my_mei_file)
```

