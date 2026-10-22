from __future__ import annotations
from typing import TYPE_CHECKING, Literal
import ee
from EeImageCollections.MultiOrbitCollection import MultiOrbitCollection

if TYPE_CHECKING:
    # TODO: replace by an OPERA-specific single orbit collection (see split_into_relative_orbit_collections)
    from EeImageCollections.OPERA.OPERA_SingleOrbitCollection import OPERA_SingleOrbitCollection


class OPERA_MultiOrbitCollection(MultiOrbitCollection):

    _units: Literal["db", "linear"]
    """
    Expressing the units in which bands 'VV' and 'VH' are expressed
    """

    FOOTPRINT_SIZE: int = 1_700_000
    """
    The approximate footprint size of an OPERA RTC-S1 image. Unlike S1 GRD (a full IW scene),
    each OPERA RTC-S1 image is a single burst (~20 km azimuth x ~85 km range, i.e. ~1/25 of a GRD scene).
    NOTE: this is scaled from the previous GRD value (45_000_000), assuming the same unit
    as used by MultiOrbitCollection. Adjust if that class uses a different convention.
    """

    COLLECTION_ID: str = "OPERA/RTC/L2_V1/S1"
    ORBIT_NUMBER_PROPERTY: str = "TRACK_NUMBER"
    ORBIT_PASS_PROPERTY: str = "ORBIT_PASS_DIRECTION"

    def __init__(self,
                 start_date: str,
                 end_date: str,
                 relative_orbit_info: dict[int, Literal["ASCENDING", "DESCENDING"]],
                 collection: ee.ImageCollection,
                 collection_id: str,
                 units: Literal["db", "linear"],
                 ):
        """ PRIVATE CONSTRUCTOR """

        super().__init__(start_date, end_date, relative_orbit_info, collection, collection_id, self.FOOTPRINT_SIZE)
        self._units = units

    @staticmethod
    def of(roi: ee.Geometry,
           start_date: str,
           end_date: str,
           units: Literal["db", "linear"] = "db",
           mask_layover_shadow: bool = False,
           ) -> OPERA_MultiOrbitCollection:
        """
        Creates an OPERA_MultiOrbitCollection instance representing OPERA RTC-S1 images (burst-based,
        radiometrically terrain corrected gamma0, 30 m) over the given region of interest ('roi')
        from 'start_date' to 'end_date'.

        Only images that contain both the 'VV' and 'VH' polarisation are retained. The resulting images
        have exactly two bands: 'VV' and 'VH'.

        'units': the OPERA dataset is natively in linear power scale. With 'db' (default) the
        bands are converted to decibels (10*log10(x)); with 'linear' they are left untouched.
        'mask_layover_shadow': if True, pixels affected by layover, shadow or invalid pixels
        (OPERA 'mask' band != 0) are masked out.
        """
        roi = MultiOrbitCollection.validate_roi(roi)
        start_date, end_date = MultiOrbitCollection.validate_dates(start_date, end_date)

        if units not in ("db", "linear"):
            raise ValueError("`units` must be either 'db' or 'linear'.")

        collection_id = OPERA_MultiOrbitCollection.COLLECTION_ID
        collection = (
            ee.ImageCollection(collection_id)
            .filterBounds(roi)
            .filterDate(start_date, end_date)
            # OPERA images can be single-pol (e.g. HH/HV): keep only dual-pol VV+VH bursts
            .filter(ee.Filter.listContains("POLARIZATIONS", "VV"))
            .filter(ee.Filter.listContains("POLARIZATIONS", "VH"))
        )

        # Retrieved before the band processing below, to keep the (client-side) metadata query cheap.
        relative_orbit_info = OPERA_MultiOrbitCollection._retrieve_relative_orbit_info(collection)

        def _prepare(img: ee.Image) -> ee.Image:
            img = ee.Image(img)
            pol = img.select(["VV", "VH"])
            if mask_layover_shadow:
                pol = pol.updateMask(img.select("mask").eq(0))
            if units == "db":
                # non-positive values have no dB representation -> mask them instead of producing -inf/NaN
                pol = pol.updateMask(pol.gt(0)).log10().multiply(10)
            return pol.copyProperties(img, img.propertyNames())

        collection = collection.map(_prepare)

        return OPERA_MultiOrbitCollection(start_date,
                                          end_date,
                                          relative_orbit_info,
                                          collection,
                                          collection_id,
                                          units)

    def get_units(self) -> Literal["db", "linear"]:
        """
        Returns the units in which the 'VV' and 'VH' band in this collection are represented.
        """
        return self._units

    @staticmethod
    def _retrieve_relative_orbit_info(collection: ee.ImageCollection) -> dict[int, Literal["ASCENDING", "DESCENDING"]]:
        """
        Fetches all relative orbit numbers (keys) and their pass directions (values).
        In OPERA, the relative orbit number is stored in 'TRACK_NUMBER'.
        """
        distinct = collection.distinct(OPERA_MultiOrbitCollection.ORBIT_NUMBER_PROPERTY)
        numbers = distinct.aggregate_array(OPERA_MultiOrbitCollection.ORBIT_NUMBER_PROPERTY).getInfo()
        passes = distinct.aggregate_array(OPERA_MultiOrbitCollection.ORBIT_PASS_PROPERTY).getInfo()
        assert numbers is not None
        assert passes is not None
        # TRACK_NUMBER is stored as a DOUBLE -> make sure the keys are real ints
        return {int(n): p for n, p in zip(numbers, passes)}

    def split_into_relative_orbit_collections(self) -> dict[int, OPERA_SingleOrbitCollection]:
        """
        Returns a dictionary with the relative orbit numbers as keys and the associated
        image collections as values.

        NOTE: S1GRD_SingleOrbitCollection filters on the GRD property 'relativeOrbitNumber_start'
        (assumed). For OPERA it must filter on 'TRACK_NUMBER' instead, so this should
        point to an OPERA-specific single orbit collection once that exists.
        """
        from EeImageCollections.OPERA.OPERA_SingleOrbitCollection import OPERA_SingleOrbitCollection
        return {
            ro: OPERA_SingleOrbitCollection.of(self, ro)
            for ro in self.get_relative_orbits()
        }