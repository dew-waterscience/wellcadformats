# wellcadformats

Read in export file formats produced by [Advanced Logic Technology (ALT)](http://www.alt.lu/default.htm)'s [WellCAD](http://www.alt.lu/software.htm) software, and provides a few utility functions for the imaging/waveform data formats.

ASCII export formats supported so far:

- .waw - well logs with a single floating point number per depth
- .waf - full-waveform sonic logs - vector array of floating point numbers per depth
- .wai - image log - vector array of integers per depth

The authors of this package have no affiliation with ALT. You may be interested in their official Python package [pywellcad](https://pypi.org/project/pywellcad/) which caters for direct automation with the WellCAD application, assuming you have the appropriate license. This package (wellcadformats) is solely about interfacing with the *file formats* which can be used to import and export data into and out of WellCAD, and can be used independently from the WellCAD software itself.
