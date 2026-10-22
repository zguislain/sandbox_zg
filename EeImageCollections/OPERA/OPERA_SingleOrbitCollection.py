from __future__ import annotations
from typing import TYPE_CHECKING, Self
from EeImageCollections.MultiOrbitCollection import MultiOrbitCollection
import ee
from EeImageCollections.OPERA import OPERA_ImageUtils as ImageUtils
import pandas as pd
import FeatureManipulation.FeatureCollectionUtils as FeatureCollectionUtils
from typing import Literal
from utils.Batch import BatchID
from EeImageCollections.SingleOrbitCollection import SingleOrbitCollection


if TYPE_CHECKING:
    from OPERA_MultiOrbitCollection import OPERA_MultiOrbitCollection


class OPERA_SingleOrbitCollection(SingleOrbitCollection):
    """
    A mutable collection containing OPERA RTC-S1 images (bursts) all of the same relative orbit (TRACK_NUMBER).
    """

    STATIC_COLLECTION_ID: str = "OPERA/RTC/L2_V1/S1_STATIC"
    """
    The OPERA collection with the static radar geometry layers (incidence angle, local incidence angle, ...).
    """

    _reprojected : bool = False
    """
    Expresses whether all images in this collection are already reprojected to a common crs or not.
    """

    _reducer_groups : dict[str, tuple[set[str], ee.Reducer]] = {
            "mean_count_stdev": (
                set(),
                ee.Reducer.mean()
                .combine(ee.Reducer.stdDev(), sharedInputs=True)
                .combine(ee.Reducer.count(), sharedInputs=True)),
            "mean_stdev": (
                set(),
                ee.Reducer.mean()
                .combine(ee.Reducer.stdDev(), sharedInputs=True) ),
            "mean": (
                set(),
                ee.Reducer.mean() ),
        }
    """
    Guides which reducers should be applied to different variables.
    Is a dictionary of
        (1) the reducer name
        (2) a list of
            (1) the set of band_names that should be reduced using this reducer
            (2) the reducer instance that should be used.
    """

    _batch_size : int = 500
    """
    The batch size into which features will be split for processing. Default batch size is 500.
    """

    _units : Literal["db", "linear"]
    """
    The  units in which the 'VV' and 'VH' bands are expressed.
    """

    def __init__(self,
                start_date: str,
                end_date: str,
                relative_orbit_dict: dict[int, Literal["ASCENDING", "DESCENDING"]],
                collection: ee.ImageCollection,
                collection_name: Literal["OPERA/RTC/L2_V1/S1"],
                footprint_size: int,
                mother_collection_id: str,
                units: Literal["db", "linear"],
                ):
        """ PRIVATE CONSTRUCTOR """
        super().__init__(
            start_date,
            end_date,
            relative_orbit_dict,
            collection,
            collection_name,
            footprint_size,
            mother_collection_id
        )

        # For OPERA the units can not be derived from the collection name (always linear natively):
        # they depend on whether the mother collection converted the bands to dB.
        self._units = units

    @classmethod
    def of(cls, multi_orbit_collection: "OPERA_MultiOrbitCollection", relative_orbit: int) -> Self:
        filtered = (
            multi_orbit_collection._collection
            .filter(ee.Filter.eq("TRACK_NUMBER", relative_orbit))
        )
        name = multi_orbit_collection.get_collection_name()
        assert name == "OPERA/RTC/L2_V1/S1"

        return cls(
            multi_orbit_collection.get_time_period()[0],
            multi_orbit_collection.get_time_period()[1],
            {relative_orbit: multi_orbit_collection.get_relative_orbit_info()[relative_orbit]},   # only this orbit
            filtered,
            name,
            multi_orbit_collection.get_footprint_size(),
            multi_orbit_collection.get_collection_id(),
            multi_orbit_collection.get_units(),
        )

    def get_units(self) -> Literal["db", "linear"]:
        """
        Returns the units in which the 'VV' and 'VH' band are expressed.
        """
        return self._units

    def add_db_cross_polarization(self) -> None:
        """
        Adds the cross polarization in dB units. The resulting band will be called 'CP'.
        Note: you cannot add both linear and dB cross polarisation bands.
        """
        if 'CP' in self.get_bands():
            return
        if self._units == 'db':
            self._collection = self._collection.map(ImageUtils.add_db_cross_polarization_from_db)
        else:
            self._collection = self._collection.map(ImageUtils.add_db_cross_polarization_from_linear)

    def add_linear_cross_polarization(self) -> None:
        """
        Adds the cross polarization in linear units. The resulting band will be called 'CP'.
        Note: you cannot add both linear and dB cross polarisation bands.
        """
        if 'CP' in self.get_bands():
            return
        if self._units == 'db':
            self._collection = self._collection.map(ImageUtils.add_linear_cross_polarization_from_db)
        else:
            self._collection = self._collection.map(ImageUtils.add_linear_cross_polarization_from_linear)

    def filter_border_noise(self) -> None:
        """
        Masks low-backscatter edge pixels.
        All pixels with 'VV' value <= -30 [dB] or 0.001 [-] are masked.

        Unlike S1 GRD, OPERA RTC-S1 bursts are not affected by GRD-style border noise and carry no
        'angle' band, so the incidence angle (28°-47°) filter of the GRD version is not applied.
        Layover/shadow/invalid pixels are handled by `mask_layover_shadow` in OPERA_MultiOrbitCollection.of().
        """
        threshold = -30 if self._units == 'db' else 0.001

        def _mask(image: ee.Image) -> ee.Image:
            image = ee.Image(image)
            return image.updateMask(image.select('VV').gt(threshold)).copyProperties(image, image.propertyNames())

        self._collection = self._collection.map(_mask)

    def add_azimuth_and_local_incidence_angle(self) -> None:
        """
        Adds two bands:
            (1) the 'AZI' or azimuth angle band containing the azimuth angle (in ° off North)
            (2) the 'LIA' or projected local incidence angle band, containting the local incidence
                angle relative to the azimuth angle and ground terrain.

        OPERA images have no 'angle' band. It is taken from the OPERA static layers
        ('incidence_angle', ellipsoid incidence angle) of the same track, added as band 'angle'
        (as in S1 GRD), after which the same computation as for S1 GRD is applied.
        """
        pass_type = self.get_pass_type()
        track = self.get_relative_orbit()

        # One static image per burst; all static layers of this track are mosaicked into a single
        # image. Overlapping bursts of the same track have virtually identical geometry.
        angle = (
            ee.ImageCollection(self.STATIC_COLLECTION_ID)
            .filter(ee.Filter.eq("TRACK_NUMBER", track))
            .select("incidence_angle")
            .mosaic()
            .rename("angle")
        )

        def _add(image: ee.Image) -> ee.Image:
            image = ee.Image(image)
            image = ee.Image(image.addBands(angle))
            return ImageUtils.add_lia_and_azimuth(image, pass_type)

        self._collection = self._collection.map(_add)

    def get_name(self) -> str:
        """
        Returns the name of this single orbit collection. This name is unique for each collection of
        a certain satellite mission, relative orbit, pass type and year.
        """
        return f"OPERA_RO_{self.get_relative_orbit()}_Pass_{self.get_pass_type()[:3]}_Year_{self.get_start_year()}"