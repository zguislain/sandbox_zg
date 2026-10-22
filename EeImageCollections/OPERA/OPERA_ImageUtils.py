import ee
import math
from typing import Literal

#TODO this still has to get updated to the OPERA DATASET

def filter_border_noise_linear(image: ee.Image) -> ee.Image:
    """
    Masks Sentinel‑1 border noise.
    All pixels with 'VV' value <= 0.001 [-] are masked.
    All angles >= 47° or <= 28° are masked.
    """
    image= ee.Image(image)
    angle_mask = image.select('angle').gt(28).And(image.select('angle').lt(47))
    vv_mask = image.select('VV').gt(0.001)

    return image.updateMask(angle_mask).updateMask(vv_mask)

def filter_border_noise_db(image: ee.Image) -> ee.Image:
    """
    Masks Sentinel‑1 border noise.
    All pixels with 'VV' value <= -30 dB are masked.
    All angles >= 47° or <= 28° are masked.
    """
    image= ee.Image(image)
    angle_mask = image.select('angle').gt(28).And(image.select('angle').lt(47))
    vv_mask = image.select('VV').gt(-30)

    return image.updateMask(angle_mask).updateMask(vv_mask)

def remove_band(image: ee.Image, band_name: str) -> ee.Image:
       """
       Returns a new ee.Image with the given band name removed 
       and all other band names included
       """
       image = ee.Image(image)
       remaining_bands = image.bandNames().remove(band_name)
       return image.select(remaining_bands)

def _compute_db(image: ee.Image, band_name) -> ee.Image:
       return image.expression(
                '10 * log10(V)', {
                'V': image.select(band_name)
        }).rename(f"{band_name}_db")

def add_db_band(image: ee.Image, band_name: str) -> ee.Image:
    return image.addBands(_compute_db(image, band_name).copyProperties(image, image.propertyNames()))

def _compute_linear_cross_polarization_from_linear(image: ee.Image) -> ee.Image:
       """
       Creates a new Image with the values cross polarization
       based on the input image.
       :param image: ee.Image with linear sigma0 bands called 'VV' and 'VH'
       :result: ee.Image with 1 band called 'CP' with the linear cross polarization values
       """
       return image.expression(
                'VH/VV', {
                'VH': image.select(['VH']),
                'VV': image.select(['VV'])
        }).rename("CP")

def add_linear_cross_polarization_from_linear(image) -> ee.Image:
        image = ee.Image(image)
        return image.addBands(_compute_linear_cross_polarization_from_linear(image).copyProperties(image, image.propertyNames()))

def add_linear_cross_polarization_from_db(image) -> ee.Image:
        image = ee.Image(image)
        cp_db = _compute_db_cross_polarization_from_db(image)
        cp_linear = _convert_db_band_to_linear(cp_db)
        return image.addBands(cp_linear.copyProperties(image, image.propertyNames()))

def _convert_db_band_to_linear(image: ee.Image) -> ee.Image:
    """
    :param image: an ee.Image instance with one band in decibel scale.
    """
    return ee.Image(10).pow(image.divide(ee.Image(10))).rename(image.bandNames()).copyProperties(image, image.propertyNames())

def _convert_linear_band_to_db(image: ee.Image) -> ee.Image:
    """
    Convert a linear-scale Sentinel‑1 band to decibels.
    :param image: ee.Image with one linear-scale band.
    :return: ee.Image with the same band name, converted to dB.
    """
    return image.log10().multiply(10).rename(image.bandNames())


def _compute_db_cross_polarization_from_db(image: ee.Image, VV_band_name = 'VV', VH_band_name = 'VH') -> ee.Image:
       """
       Creates a new Image with the values cross polarization [dB]
       based on the input image.
       :param image: ee.Image with decibel sigma0 bands called 'VV' and 'VH'
       :result: ee.Image with 1 band called 'CP' with the cross polarization values in dB
       """
       return image.expression(
                'VH - VV', {
                'VH': image.select(['VH']),
                'VV': image.select(['VV'])
        }).rename("CP")
        
def add_db_cross_polarization_from_db(image, VV_band_name = 'VV', VH_band_name = 'VH') -> ee.Image:
        image = ee.Image(image)
        return image.addBands(_compute_db_cross_polarization_from_db(image).copyProperties(image, image.propertyNames()))

def add_db_cross_polarization_from_linear(image, VV_band_name = 'VV', VH_band_name = 'VH') -> ee.Image:
       image = ee.Image(image)
       cp_linear = _compute_linear_cross_polarization_from_linear(image)
       cp_db = _convert_linear_band_to_db(cp_linear)
       return image.addBands(cp_db)

def _get_srtm() -> ee.Image:
    """
    Returns a one band image containing values from the NASA SRTM Digital Elevation 30m
    """
    dataset = ee.Image('USGS/SRTMGL1_003')
    return dataset.select('elevation')

def _get_azimuth_relative_to_image_grid(angle: ee.Image) -> ee.Number:
      """
      Takes an ee.Image with one band containing the indidence angle values
      (relative to earth elipse).
      Returns the direection relative to the top of the image that the image
      was taken from.
      """
      return ee.Number(
                ee.Terrain.aspect(angle) \
                .reduceRegion(ee.Reducer.mean(), angle.geometry(), 100) \
                .get('aspect')
                )

def _get_image_corners(image) -> ee.Dictionary:
    """
    Returns the four extreme corner points of a given image's geometry
    as (lon, lat) pairs: the most northern, eastern, southern, and
    western points.
    """
    # Ring repeats its first point at the end to close the polygon - drop it
    # so it can't be mistaken for a distinct corner when sorting.
    ring = ee.List(image.geometry().coordinates().get(0))
    coords = ring.slice(0, ring.length().subtract(1))
    coords = ee.Array(coords).transpose()

    all_longitudes = ee.List(coords.toList().get(0))
    all_latitudes = ee.List(coords.toList().get(1))
    paired = all_longitudes.zip(all_latitudes)  # [[lon, lat], ...]

    # Sort once by each axis; the extremes sit at the ends of each sort.
    by_longitude = paired.sort(all_longitudes)
    by_latitude = paired.sort(all_latitudes)

    west = by_longitude.get(0)
    east = by_longitude.get(-1)
    south = by_latitude.get(0)
    north = by_latitude.get(-1)

    return ee.Dictionary({
        'north': north,
        'east': east,
        'south': south,
        'west': west,
    })

def _get_image_tilt(image: ee.Image, pass_type: Literal["ASCENDING", "DESCENDING"]) -> ee.Number:
        """
        Computes the tilt of the image relative to North, using the two most
        eastern corner points. The tilt is the angle between the line joining
        those two points and the north-south line running through whichever of
        the two is more southern.

        Sign convention:
                Positive -> the more southern of the two points is also the more
                        eastern one (image tilts clockwise).
                Negative -> the more southern of the two points is the more
                        western one (image tilts counterclockwise).
        """
        corners = _get_image_corners(image)

        # Approximates the image tilt as the tilt if the western border
        # of the image.
        # When DESCENDING this is the border bounded by the most
        # Nothern and Western corner of the image.
        if pass_type == "DESCENDING":
                upper = ee.List(corners.get('north'))
                lower = ee.List(corners.get('west'))
        # When ASCENDING this is the border bounded by the most
        # Wester and Souther corner of the image.
        else:
                upper = ee.List(corners.get('west'))
                lower = ee.List(corners.get('south'))

        delta_lon = ee.Number(upper.get(0)).subtract(ee.Number(lower.get(0)))
        delta_lat = ee.Number(upper.get(1)).subtract(ee.Number(lower.get(1)))

        # The tilt is then approximated as the angle this border makes with
        # the North in the most southern (lower) point of the two. The angle is
        # defined clockwise, with positive values for DESCENDING and negative Values
        # for ASCENDING.
        tilt_rad = delta_lat.atan2(delta_lon)  # ee.Number(a).atan2(b) == atan2(a, b)
        tilt_deg = tilt_rad.multiply(180 / math.pi)

        # If ASCENDING is does not have a negative result or DESCENDING does
        # does not have a positive result the assumptions of this method are violated
        #local_tilt = tilt_deg.getInfo()
        #if local_tilt == None:
        #       raise RuntimeError("Got an unexpected none output from gee.")
        #if pass_type == "ASCENDING" and local_tilt > 0:
        #       raise RuntimeError("Assumptions of this method were violated.")
        #elif pass_type == "DESCENDING" and local_tilt < 0:
        #       raise RuntimeError("Assumptions of this method were violated.")

        return tilt_deg

def _compute_azimuth(image: ee.Image, pass_type: Literal["ASCENDING", "DESCENDING"]) -> ee.Number:
      """
      Computes the azimuth angle [°] relative to north for an image.
      NOTE: the underlying function rely on assumptions that are valid for
      images of the COPERNICUS/S1_GRD and COPERNICUS/S1GRD_FLOAT collections.
      Transferability to other collections is not guarantied.
      """
      image_grid_azimuth = _get_azimuth_relative_to_image_grid(image.select("angle"))
      image_tilt =  _get_image_tilt(image, pass_type)
      return image_grid_azimuth.add(image_tilt)


def _compute_lia_and_azimuth(image: ee.Image, pass_type: Literal["DESCENDING", "ASCENDING"]) -> ee.Image:
        """
        returns an image with two bands:
                'LAI': with the projected local incidence angle per pixel
                'AZI': with the average azimuthal angle (constant over all pixels)
        NOTE: The internal logic should be verified again.
        NOTE: I made significant changes to the calculation procedure. I think now it is correct.
        However, the result IS different from previous versions
        """
        srtm = _get_srtm()
        srtm_slope = ee.Terrain.slope(srtm)
        srtm_aspect = ee.Terrain.aspect(srtm)

        azimuth = _compute_azimuth(image, pass_type)

        projected_slope = srtm_slope \
                                .multiply(
                                ee.Image.constant(azimuth)
                                        .subtract(srtm_aspect)
                                        .multiply(math.pi/180)\
                                        .cos())
        projected_lia = image.select("angle").subtract(projected_slope).abs()
        result = projected_lia.addBands(azimuth).clip(projected_lia.geometry()) # add azimuth angle to the lia image... this new band is called constant
        result = result.select(['angle','constant']).rename(['LIA','AZI']) # REnaming the bands in order to merge it together with the VH and VV datasets (they need to have the same coloumn names in order to)
        return result
    
def _compute_lia_and_azimuth_old(image: ee.Image, pass_type: Literal["ASCENDING", "DESCENDING"]) -> ee.Image:
        """
        returns an image with two bands:
                'LAI': with the projected local incidence angle per pixel
                'AZI': with the average azimuthal angle (constant over all pixels)
        NOTE: The internal logic should be verified again.
        """
        #TODO what is the meaning of the values?
        azimuth_vals = {
               "ASCENDING"      : 270,
               "DESCENDING"     : 360
        }
        rotation_vals = {
               "ASCENDING"      : 180,
               "DESCENDING"     : 180    
        }

        S1angle = image.select('angle')
        #We can use the gradient of the "angle" band of the S1 image to derive the S1 azimuth angle.
        S1_azimuth = ee.Terrain.aspect(S1angle) \
                                .reduceRegion(ee.Reducer.mean(), S1angle.geometry(), 100) \
                                .get('aspect')
        # Attention, this is not actually azimuth, but the look direction across range, which is NOT
        # yet corrected for the angle with which s1_inc is rotated relative to North!!!
        # Calculate True azimuth direction for the near range image edge
        def getCorners(f) -> ee.Element:
                # Get the coords as a transposed array
                coords = ee.Array(f.geometry().coordinates().get(0)).transpose()
                crdLons = ee.List(coords.toList().get(0))
                crdLats = ee.List(coords.toList().get(1))
                minLon = crdLons.sort().get(0)
                maxLon = crdLons.sort().get(-1)
                minLat = crdLats.sort().get(0)
                maxLat = crdLats.sort().get(-1)
                azimuth = ee.Number(crdLons.get(crdLats.indexOf(minLat))).subtract(minLon).atan2(ee.Number(crdLats.get(crdLons.indexOf(minLon))).subtract(minLat)) \
                        .multiply(180.0/math.pi).add(azimuth_vals[pass_type]) 
                return ee.Feature(ee.Geometry.LineString([
                                        crdLons.get(crdLats.indexOf(minLat)),
                                        minLat, minLon, crdLats.get(crdLons.indexOf(minLon))
                                        ]), 
                                  { 'azimuth': azimuth}).copyProperties(f)
        
        azimuthEdge = getCorners(image)
        
        TrueAzimuth = azimuthEdge.get('azimuth')   # This should be some degree off the North direction, due to Earth rotation
        rotationFromNorth_or_South = ee.Number(TrueAzimuth).subtract(rotation_vals[pass_type]) # Use subtract(180.0) for DESCENDING and subtract(360.0) for ASCENDING image.
        
        # Correct the across-range-look direction
        S1_azimuth = ee.Number(S1_azimuth).add(rotationFromNorth_or_South)  
        # Here we derive the terrain slope and aspect
        
        srtm = _get_srtm()
        srtm_slope = ee.Terrain.slope(srtm).select('slope')
        srtm_aspect = ee.Terrain.aspect(srtm).select('aspect')

        # And finally the local incidence angle
        #TODO why use TrueAzimuth in the calculations here, but add S1_azimuth?
        slope_projected2 = srtm_slope.multiply(ee.Image.constant(TrueAzimuth).subtract(90.0).subtract(srtm_aspect).multiply(math.pi/180).cos())
        lia2 = S1angle.subtract(ee.Image.constant(90).subtract(ee.Image.constant(90).subtract(slope_projected2))).abs()
        angles = lia2.addBands(S1_azimuth).clip(lia2.geometry()) # add azimuth angle to the lia image... this new band is called constant
        angles = angles.select(['angle','constant']).rename(['LIA','AZI']) # REnaming the bands in order to merge it together with the VH and VV datasets (they need to have the same coloumn names in order to)
        return angles

def add_lia_and_azimuth(image: ee.Image, pass_type : Literal["ASCENDING", "DESCENDING"], method:Literal['new', 'old']='new') -> ee.Image:
        """
        """
        if method == 'old':
                return image.addBands(_compute_lia_and_azimuth_old(image, pass_type).copyProperties(image, image.propertyNames()))
        return image.addBands(_compute_lia_and_azimuth(image, pass_type).copyProperties(image, image.propertyNames()))
