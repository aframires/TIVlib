# TIV.lib
A python library for the content-based tonal description of musical audio signals, which implements the Tonal Interval Vector space.
Its main novelty relies on the DFT-based perceptually-inspired Tonal Interval Vector space, from which multiple instantaneous and global representations, descriptors and metrics are computed---e.g., harmonic changes, dissonance, diatonicity, and musical key.

Installation and Usage
-------
To use this application, the user should first clone this repository to the working python directory.
Then, in case `numpy` is not installed, the user should run `pip install numpy`.
The library can be imported with `import TIVlib as tiv`.

Example Code
-------
A detailed example on how to use the TIV.lib is available in [TIVlib_example.ipynb](TIVlib_example.ipynb) and in a [Google Colab notebook](https://colab.research.google.com/drive/1QCoOI5Ix9_cekpMDcI7liVIJM--Kg6px) .

We provide here a simple example on how to run some of the feature extractors available in TIV.lib:

```python
import TIVlib as tiv
metal = "./audio_files/metal.wav"
metal_hpcp = file_to_hpcp(metal)
metal_tiv = tiv.TIV.from_pcp(metal_hpcp)
print("tiv.mag: " + str(tiv.mag.mags(glass_tiv)))
print("tiv.diatonicity: " + str(tiv.mag.diatonicity(glass_tiv)))
print("tiv.chromacity: " + str(tiv.mag.chromaticity(glass_tiv)))
print("tiv.dissonance: " + str(tiv.dissonance(glass_tiv)))

```

References
-------
For more information on the TIV, it's characteristics and it's applications, we redirect the readers to the [TIV website](https://sites.google.com/site/tonalintervalspace/home) . 


Authors
-------
António Ramires
aframires@gmail.com

Gilberto Bernardes
gba@feup.pt
