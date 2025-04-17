# Embryo VNC align

A script for aligning an embryo VNC into the right orientation

## Installation

optional: use envrironment.yml to create an envrironment to install the data into

`pip install git+https://github.com/pnewstein/embryo-vnc-align`

## Usage


`align [OPTIONS] FILE`

Rotate an embryo into the right orientation cropping to the limits of the VNC.
This takes a .czi or ome.tiff file and reads a particular scene. It then
displays a particular channel of that image, and asks the user to click on 4
points. The deep anterior end of the VNC. The far left and far right sides in
the middle, and the deep posterior end of the VNC. The line between the
posterior and anterior points is taken as the Y axis. and the image is then
spun around the Y axis until the left and right points share the same Z. All
chanells are rotated in this manner and cropped according to the position of
the points, and the image is saved as an ome.tif

### Example
  align -s 0 -c 0 test.czi

### Options
```
  -s, --scene INTEGER             scene to read. Default is 0

  -c, --channel INTEGER           channel to visualize. Default is 0. Ignored
                                  if --dont-take-coords
  -g, --gamma FLOAT               gamma correction for visualization. Default
                                  is 0.2 Ignored if --dont-take-coords
  --take-coords / --dont-take-coords
                                  whether to open a gui to select the
                                  appropriate coordinates. orelse use previous
                                  coordinates
  --proc-image / --dont-proc-image
                                  whether to processes the image, or else just
                                  run the GUI saving the coords.
  -p, --pixel-buffer-factor FLOAT
                                  Scales the buffer around the landmarks that
                                  are still included in in the image. Larger
                                  numbers lead to larger images. Default is 1.
                                  Ignored if --dont-proc-image
  -v, --verbose                   Adds helpful messages. Use -vv for even more
                                  verbosity
  --help                          Show this message and exit.

```

