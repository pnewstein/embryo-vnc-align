"""
Uses subprocesses run in order to align
"""

from subprocess import run
from pathlib import Path
from typing import Protocol, Generator, TypeAlias
import logging
from dataclasses import dataclass
import xml.etree.ElementTree as ET
import socket

from aicspylibczi import CziFile
import nrrd
import numpy as np
from scipy.spatial.transform import Rotation
from PyQt5.QtWidgets import (
    QMainWindow,
    QSlider,
    QVBoxLayout,
    QWidget,
    QApplication,
    QLabel,
)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backend_bases import MouseEvent
from pyometiff import OMETIFFWriter
import tifffile
import click

Coords: TypeAlias = tuple[float, float, float]

logger = logging.getLogger(__name__)


@dataclass
class CoordsSet:
    """
    3 coordinates to define the orientation of the VNC
    """

    anterior: Coords
    "based in pixels"
    left: Coords
    "based in pixels"
    right: Coords
    "based in pixels"
    posterior: Coords
    "based in pixels"
    scale: np.ndarray

    def to_cmtk(self):
        """
        returns a string that can be written to disk
        """
        lines = [f"{z} {y} {x} {name}" for name, (z, y, x) in self.to_dict().items()]
        return "\n".join(lines)

    def to_array(self) -> np.ndarray:
        return np.stack([list(v) for v in self.to_dict().values()])

    def to_dict(self) -> dict[str, Coords]:
        unscaled_dict = {
            "anterior": self.anterior,
            "left": self.left,
            "right": self.right,
            "posterior": self.posterior,
        }
        return {
            k: tuple((np.array(list(v)) * self.scale).tolist())
            for k, v in unscaled_dict.items()
        }

    @classmethod
    def from_um_coords(cls, coords: np.ndarray, scale: np.ndarray):
        pixel_coords = [tuple(c / scale) for c in coords]
        return cls(*pixel_coords, scale=scale)


class CoordsCallback(Protocol):
    def __call__(self, coords_set: CoordsSet): ...


class ImageSlicer(QMainWindow):
    def __init__(
        self,
        volume: np.ndarray,
        app: QApplication,
        click_generator: Generator[str, Coords, None] | None,
    ):
        """
        a click_generator yeilds titles when a click occurs
        gamma is the gamma correction
        scale is the fraction scale to use
        """
        super().__init__()
        # set volume dynamic range
        self.volume = (volume * (254 / np.max(volume))).astype(np.uint8)
        self.app = app
        self.current_slice = 0
        self.click_generator = click_generator
        # Create main widget and layout
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)
        self.message = QLabel("test")
        if click_generator is not None:
            self.layout.addWidget(self.message)
        # Create matplotlib canvas
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        # Create slicer
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(volume.shape[0] - 1)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.update_slice)
        self.layout.addWidget(self.slider)
        # Connect click callback
        if self.click_generator is not None:
            message = next(self.click_generator)
            self.message.setText(message)
            self.figure.canvas.mpl_connect("button_press_event", self.on_click)  # type: ignore

        # Initialize plot
        self.update_slice(0)

    def on_click(self, event: MouseEvent):
        assert self.click_generator is not None
        if event.inaxes is not None:
            y_coord = event.ydata
            assert y_coord is not None
            x_coord = event.xdata
            assert x_coord is not None
            try:
                self.message.setText(
                    self.click_generator.send((self.current_slice, y_coord, x_coord))
                )
            except StopIteration:
                self.close()

    def update_slice(self, slice_idx: int):
        self.current_slice = slice_idx
        self.ax.clear()
        self.ax.imshow(
            self.volume[slice_idx, :, :],
            cmap="gray",
            aspect="equal",
            vmin=0,
            vmax=255,
        )
        self.ax.set_title(f"Slice {slice_idx}")
        self.ax.axis("off")
        self.canvas.draw()

    def quit(self):
        # causes segfault
        self.app.quit()


def coords_generator(callback: CoordsCallback, scale: np.ndarray):
    anterior = yield "Click on most deep anterior part of the VNC"
    assert anterior is not None
    logger.info(",".join(str(c) for c in anterior))
    left = yield "Click on a side of the VNC"
    assert left is not None
    logger.info(",".join(str(c) for c in left))
    right = yield "Click on the other side of the VNC "
    assert right is not None
    logger.info(",".join(str(c) for c in right))
    posterior = yield "Click on most deep posterior part of the VNC"
    assert posterior is not None
    logger.info(",".join(str(c) for c in posterior))
    callback(
        CoordsSet(
            posterior=posterior, left=left, right=right, anterior=anterior, scale=scale
        )
    )


class LazyImageChannels(Protocol):
    in_path: Path
    scale: np.ndarray
    channel_range: range
    unique_path: Path

    # def __init__(self, in_path, other_image_specs: dict | None = None): ...

    def get_channel_data(self, chan: int) -> np.ndarray: ...

    def get_channel_metadata(self, chan: int) -> dict[str, str]: ...


class LazyTiffChannels:
    """
    Lazily read channels from a tiff file
    with caching
    """

    def __init__(self, in_path: Path, scene=0):
        self.in_path = in_path
        tif = tifffile.TiffFile(in_path)
        self.series = tif.series[scene]
        shape = self.series.shape
        assert tif.ome_metadata is not None
        mdata = ET.fromstring(tif.ome_metadata)
        image = mdata[scene]
        namespace = {"ome": next(iter(mdata.attrib.values())).split()[0]}
        pixels = image.find("ome:Pixels", namespaces=namespace)
        assert pixels is not None
        pixel_md = pixels.attrib
        # get xyz scale
        self.scale = np.array(
            [
                pixel_md["PhysicalSizeZ"],
                pixel_md["PhysicalSizeY"],
                pixel_md["PhysicalSizeX"],
            ]
        ).astype(float)
        self.channel_range = range(shape[0])
        self._channel_dict: dict[int, np.ndarray | None] = {
            c: None for c in self.channel_range
        }
        self.unique_path = Path(f"{in_path.stem}")
        self.channel_xml_mdatata = pixels.findall("ome:Channel", namespaces=namespace)
        tot = len(self.series)
        n_slices = len(self.channel_range)
        assert tot % n_slices == 0
        step = tot // n_slices
        self.channel_slices = [slice(i, i + step) for i in range(0, tot, step)]

    def get_channel_data(self, chan: int) -> np.ndarray:
        if chan not in self._channel_dict:
            raise ValueError(f"No channel {chan}")
        chan_or_none = self._channel_dict[chan]
        if chan_or_none is not None:
            return chan_or_none
        logger.debug("Loading chan %d", chan)
        image = self.series.asarray(key=self.channel_slices[chan])
        self._channel_dict[chan] = image
        return image

    def get_channel_metadata(self, chan: int) -> dict[str, str]:
        if chan not in self._channel_dict:
            raise ValueError(f"No channel {chan}")
        chan_md = self.channel_xml_mdatata[chan].attrib
        return chan_md


class LazyCziChannels:
    """
    Lazily read channels from a czi file
    with Caching
    """

    def __init__(self, in_path: Path, other_image_specs: dict | None = None):
        self.in_path = in_path
        if other_image_specs is None:
            other_image_specs = {}
        self.other_image_specs = other_image_specs
        self.czi = CziFile(in_path)
        metadata = self.czi.meta
        assert metadata is not None
        distances = metadata.find("./Metadata/Scaling/Items")
        assert distances is not None
        # get xyz scale
        self.spacings_dict: dict[str, float] = {}
        for distance in distances:
            value = distance.find("Value")
            assert value is not None
            assert value.text is not None
            self.spacings_dict[next(iter(distance.items()))[-1]] = (
                float(value.text) * 10e5
            )
        self.scale = np.array(
            [self.spacings_dict["Z"], self.spacings_dict["Y"], self.spacings_dict["X"]]
        )
        assert self.czi.shape_is_consistent
        (dimensions,) = self.czi.get_dims_shape()
        assert all(i[0] == 0 for i in dimensions.values())
        self.channel_range = range(*dimensions["C"])
        self._channel_dict: dict[int, np.ndarray | None] = {
            c: None for c in self.channel_range
        }
        image_specs = "".join(f"{k}{v}" for k, v in self.other_image_specs.items())
        self.unique_path = Path(f"{in_path.stem}-{image_specs}")
        channels_element = self.czi.meta.find(
            "./Metadata/Information/Image/Dimensions/Channels"
        )
        assert channels_element is not None
        self.channels_element = channels_element

    def get_channel_data(self, chan: int) -> np.ndarray:
        if chan not in self._channel_dict:
            raise ValueError(f"No channel {chan}")
        chan_or_none = self._channel_dict[chan]
        if chan_or_none is not None:
            return chan_or_none
        logger.debug("Loading chan %d", chan)
        self.other_image_specs.update({"C": chan})
        image, dims = self.czi.read_image(**self.other_image_specs)
        assert "".join(dim for dim, count in dims if count != 1) == "ZYX"
        self._channel_dict[chan] = image
        return image

    def get_channel_metadata(self, chan: int) -> dict[str, str]:
        if chan not in self._channel_dict:
            raise ValueError(f"No channel {chan}")
        full_name = self.channels_element[chan].attrib["Name"]
        fluor = self.channels_element[chan].findtext("./Fluor")
        assert fluor is not None
        name = full_name.split("-")[0]
        return {"Name": name, "Fluor": fluor}


def save_files(lic: LazyImageChannels) -> list[Path]:
    header = {
        "space": "RAS",
        "sample units": ("micron", "micron", "micron"),
        "space directions": np.diag(lic.scale),
        "labels": ["Z", "Y", "X"],
    }
    out_files: list[Path] = []
    for channel in lic.channel_range:
        image = lic.get_channel_data(channel)
        file_path = lic.unique_path.with_suffix(f".chan{channel+1}.nrrd")
        logger.debug("writting chan %d", channel)
        nrrd.write(
            str(file_path),
            data=image.squeeze(),
            header=header,
            compression_level=1,
        )
        out_files.append(Path(file_path))
    return out_files


def compute_alignment_rotation(
    top: np.ndarray, bottom: np.ndarray, left: np.ndarray, right: np.ndarray
) -> Rotation:
    """
    Compute a rotation such that:
      - After rotation (with coordinates expressed in the [Z, Y, X] order),
        the "top" and "bottom" points differ only in their Y coordinate.
      - The "left" and "right" points share the same Z coordinate,
        which is less than that for top and bottom.

    Parameters
    ----------
    top, bottom, left, right : np.ndarray
        3-element arrays with coordinates in order [Z, Y, X].

    Returns
    -------
    Rotation
        A scipy.spatial.transform.Rotation object representing the rotation.

    Raises
    ------
    ValueError
        If the left-right points are degenerate with respect to the top-bottom line.

    Notes
    -----
    This function builds a new orthonormal basis as follows:

      - new Y axis = normalize(top - bottom)
      - new Z axis is chosen so that (right - left) has zero projection on it:
          d_proj = (right - left) minus its component along new Y,
          then new Z = normalize(new Y x d_proj).
      - new X axis = normalize(new Z x new Y)
      - If necessary, new Z (and new X) are flipped so that
          new_Z · top > new_Z · left.

    In the new coordinate system the rotated coordinates v' of any point v satisfy:
      v' = [ v'_Z, v'_Y, v'_X ] = [ dot(v, new_Z), dot(v, new_Y), dot(v, new_X) ]
    Thus, top and bottom will share the same new Z and new X, and left/right
    will share the same new Z.
    """
    # Step 1: New Y axis: along top-to-bottom direction.
    y_axis = top - bottom
    norm_y = np.linalg.norm(y_axis)
    if norm_y < 1e-8:
        raise ValueError("Top and bottom points must be distinct.")
    y_axis = y_axis / norm_y
    # Step 2: For left-right difference.
    d = right - left
    # Remove component along y_axis:
    d_proj = d - np.dot(d, y_axis) * y_axis
    norm_d_proj = np.linalg.norm(d_proj)
    if norm_d_proj < 1e-8:
        raise ValueError(
            "Left and right points are degenerate (or colinear with top-bottom)."
        )
    d_proj = d_proj / norm_d_proj
    # Step 3: New Z axis: perpendicular to both y_axis and d_proj.
    new_z = np.cross(y_axis, d_proj)
    norm_z = np.linalg.norm(new_z)
    if norm_z < 1e-8:
        raise ValueError("Failed to compute a valid new Z axis.")
    new_z = new_z / norm_z
    # Step 4: New X axis to complete the right-handed system.
    new_x = np.cross(new_z, y_axis)
    new_x = new_x / np.linalg.norm(new_x)
    # Step 5: Adjust sign: we want new_Z(top) > new_Z(left)
    if np.dot(new_z, top) < np.dot(new_z, left):
        new_z = -new_z
        new_x = -new_x
    # Form a rotation matrix whose columns are the new basis vectors:
    # new_Z, new_Y, new_X corresponding to coordinates [Z, Y, X].
    R_new = np.column_stack((new_z, y_axis, new_x))
    # For a point v, its new coordinates are: v' = R_new^T v.
    # Thus, we return the rotation corresponding to the transpose of R_new.
    return Rotation.from_matrix(R_new.T)


def rotate_image(lcc: LazyImageChannels):
    # read the landmarks file
    src_landmarks = lcc.unique_path.with_suffix(".landmarks")
    lm_text = src_landmarks.read_text()
    coords_dict: dict[str, np.ndarray] = {}
    for line in lm_text.splitlines():
        list_str = line.split()
        coords_dict[list_str[-1]] = np.array(list_str[:-1]).astype(float)
    # determine xformed cords by rotating then adding buffers
    lower_range_divisor = np.array((0.3, 10, 10))
    upper_range_divisor = 10
    top = np.array(coords_dict["anterior"])
    left = np.array(coords_dict["left"])
    right = np.array(coords_dict["right"])
    bottom = np.array(coords_dict["posterior"])
    target_landmarks = lcc.unique_path.with_suffix(".target.landmarks")
    xform = lcc.unique_path.with_suffix(".xform")
    out_scale = np.array([0.19] * 3)
    rot = compute_alignment_rotation(top, bottom, left, right)
    rotated_coords = rot.apply(np.stack((top, left, right, bottom)))
    rc_min = rotated_coords.min(axis=0)
    rc_max = rotated_coords.max(axis=0)
    rc_range = rc_max - rc_min
    assert min(rc_range) > 0, "bad coordiate system"
    offset = rc_min - (rc_range / lower_range_divisor)
    npix = (
        rc_range + (rc_range / upper_range_divisor) + (rc_range / lower_range_divisor)
    ) / out_scale
    xformed_coords = rotated_coords - offset
    out_coords = CoordsSet.from_um_coords(xformed_coords, out_scale)
    target_landmarks.write_text(out_coords.to_cmtk())
    # find cmtk affine matrix
    args = (
        "fit_affine_xform_landmarks",
        "--rigid",
        target_landmarks,
        src_landmarks,
        xform,
    )
    run(args, check=True)
    # determine which data to sample
    target_grid = (
        ",".join((f"{p:d}" for p in np.ceil(npix).astype(int)))
        + ":"
        + ",".join([f"{s:.4f}" for s in out_scale])
    )
    unrotated_nrrds = save_files(lcc)
    rotated_nrrds = [p.with_suffix(".reformat.nrrd") for p in unrotated_nrrds]
    for unrotated_file, rotated_file in zip(unrotated_nrrds, rotated_nrrds):
        args = (
            "reformatx",
            "-o",
            rotated_file,
            "--cubic",
            "--target-grid",
            target_grid,
            "--floating",
            unrotated_file,
            xform,
        )
        args_str = " ".join(str(a) for a in args)
        logger.info(args_str)
        logger.debug("Aligning %s", str(unrotated_file))
        run(args, check=True)
    # combine into tif
    channel_md: dict[str, dict] = {}
    channel_data: list[np.ndarray] = []
    md: dict | None = None
    for i, rotated_file in enumerate(rotated_nrrds):
        image, md = nrrd.read(str(rotated_file))
        channel_data.append(image)
        channel_dict = lcc.get_channel_metadata(i)
        channel_md[channel_dict["Name"]] = channel_dict
    assert md is not None
    scale = np.diag(md["space directions"])
    metadata_dict = {
        "PhysicalSizeX": scale[2],
        "PhysicalSizeXUnit": "µm",
        "PhysicalSizeY": scale[1],
        "PhysicalSizeYUnit": "µm",
        "PhysicalSizeZ": scale[0],
        "PhysicalSizeZUnit": "µm",
        "Channels": channel_md,
    }
    array = np.stack(channel_data)
    writer = OMETIFFWriter(
        fpath=lcc.unique_path.with_suffix(".ome.tiff"),
        dimension_order="CZYX",
        array=array,
        metadata=metadata_dict,
    )
    logger.debug("Composing OME tiff")
    writer.write()
    # Clear garbage
    to_clear = unrotated_nrrds + rotated_nrrds + [target_landmarks, xform]
    for file in to_clear:
        file.unlink()


def main(
    in_path: Path,
    scene=0,
    channel=0,
    other_image_specs: dict[str, int] | None = None,
    take_coords=True,
    proc_image=True,
):
    """
    opens the file in a viewer, prompts landmarks then writes those landmarks
    to disk as a txt file
    """
    if other_image_specs is None:
        other_image_specs = {}
    other_image_specs.update(S=scene)
    # perhaps update file_name
    if socket.gethostname() == "UO-2008493" and len(in_path.parts) > 2 and in_path.parts[2] == "DoeLab65TB":
        in_path = Path("/mnt/z") / in_path.relative_to(Path(*in_path.parts[:3]))
    if in_path.suffix == ".czi":
        lcc = LazyCziChannels(in_path, other_image_specs)
    elif tuple(in_path.suffixes) in ((".ome", ".tiff"), (".ome", ".tiff)")):
        lcc = LazyTiffChannels(in_path, scene)
    else:
        raise NotImplementedError("Only .ome.tif and czi are supported")
    out_path = lcc.unique_path.with_suffix(".landmarks")

    if not take_coords:
        rotate_image(lcc)
        return

    image = lcc.get_channel_data(channel)

    slicer: ImageSlicer | None = None

    def callback(coords_set: CoordsSet):
        out_path.write_text(coords_set.to_cmtk())
        nonlocal slicer
        assert slicer is not None
        slicer.close()
        if proc_image:
            rotate_image(lcc)

    app = QApplication([])
    slicer = ImageSlicer(
        image.squeeze(),
        app,
        coords_generator(
            callback,
            np.array([0.19] * 3),
            # np.array([spacings_dict["Z"], spacings_dict["Y"], spacings_dict["X"]]),
        ),
    )
    slicer.resize(800, 600)
    slicer.setWindowTitle(str(in_path))
    slicer.show()
    slicer.app.exec_()


@click.command(name="embryo-vnc-align")
@click.argument("file", type=click.Path())
@click.option("-s", "--scene", type=int, default=0, help="scene to read. Default is 0")
@click.option(
    "-c",
    "--channel",
    type=int,
    default=0,
    help="channel to visualize. Default is 0. Ignored if --dont-take-coords",
)
@click.option(
    "--take-coords/--dont-take-coords",
    type=bool,
    default=True,
    help="whether to open a gui to select the appropriate coordinates. orelse use previous coordinates",
)
@click.option(
    "--proc-image/--dont-proc-image",
    type=bool,
    default=True,
    help="whether to processes the image, or else just run the GUI saving the coords.",
)
def cli(file: str, scene: int, channel: int, take_coords: bool, proc_image: bool):
    """
    Rotate an embryo into the right orientation cropping to the limits of the
    VNC. This takes a .czi or ome.tiff file and reads a particular scene. It
    then displays a particular channel of that image, and asks the user to
    click on 4 points. The deep anterior end of the VNC. The far left and far
    right sides in the middle, and the deep posterior end of the VNC. The line
    between the posterior and anterior points is taken as the Y axis. and the
    image is then spun around the Y axis until the left and right points share
    the same Z. All chanells are rotated in this manner and cropped according
    to the position of the points, and the image is saved as an ome.tif

    \b
    embryo-vnc-align -s 0 -c 0 test.czi
    """
    main(
        Path(file),
        scene=scene,
        channel=channel,
        take_coords=take_coords,
        proc_image=proc_image,
    )


cli()
