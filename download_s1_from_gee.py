import ee
from typing import List, Dict
import datetime

# Initialize Earth Engine
ee.Initialize()

def create_sar_composite(year: int, bands: List[str]) -> None:
    """
    Create and export SAR composites from Sentinel-1 imagery.
    
    Args:
        year: The year for which to create the composite
        bands: List of bands to process (e.g., ['VV', 'VH'])
    """
    # Define the Sentinel-1 collection
    s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
          .filter(ee.Filter.date(f'{year}-01-01', f'{year+1}-01-01'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
          .select(bands))

    # Separate ascending and descending orbits
    s1_asc = s1.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
    s1_desc = s1.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))

    # Create yearly composites using median
    composite_asc = s1_asc.median()
    composite_desc = s1_desc.median()

    # Merge ascending and descending composites
    composite = ee.Image.cat([
        composite_asc.select(bands, [f'{b}_asc' for b in bands]),
        composite_desc.select(bands, [f'{b}_desc' for b in bands])
    ])

    return composite

def get_utm_epsg(name: str, combined: str) -> ee.Number:
    """
    Construct EPSG code for UTM zone.
    
    Args:
        name: Tile name
        combined: Combined string containing hemisphere information
    
    Returns:
        ee.Number: EPSG code
    """
    zone = ee.Number.parse(ee.String(name).slice(0, 2))
    hemisphere = ee.String(combined).slice(-1)
    return ee.Number(32600).add(zone).add(
        ee.Algorithms.If(hemisphere.equals('S'), 100, 0))

def export_tile(feature: ee.Feature, composite: ee.Image, year: int) -> None:
    """
    Export a single tile to Google Drive.
    
    Args:
        feature: Earth Engine feature representing the tile
        composite: The SAR composite image
        year: Year of the composite
    """
    tile_id = ee.String(feature.get('Name'))
    combined = ee.String(feature.get('Combined'))
    tile_geometry = feature.geometry()
    epsg = get_utm_epsg(tile_id, combined)
    
    # Get the values that need to be computed server-side
    tile_id_str = tile_id.getInfo()
    epsg_code = epsg.getInfo()
    
    task = ee.batch.Export.image.toDrive(
        image=composite.clip(tile_geometry),
        description=f'SAR_Composite_{year}_{tile_id_str}',
        folder='GEE_SAR_Composites',
        scale=10,
        crs=f'EPSG:{epsg_code}',
        region=tile_geometry,
        maxPixels=1e13
    )
    
    # Start the export task
    task.start()

def main():
    # Set parameters
    year = 2018
    bands = ['VV', 'VH']
    
    # Create the composite
    composite = create_sar_composite(year, bands)
    
    # Visualization parameters (if needed)
    vis_params = {
        'bands': ['VV_asc', 'VH_asc', 'VV_desc'],
        'min': -25,
        'max': 5
    }

    tiles = ["T30TXP"]
    
    # Get the table of tiles
    table = ee.FeatureCollection('projects/worldwidemap/assets/all_sentinel_tiles')  # Replace with your actual table asset ID
    
    # Filter the table for specified tiles
    filtered_table = table.filter(ee.Filter.inList('Name', tiles))
    
    # Create export tasks
    features = filtered_table.getInfo()['features']
    for feature in features:
        export_tile(ee.Feature(feature), composite, year)
    
    # Print information
    print(f'Year: {year}')
    print(f'Bands: {bands}')
    print(f'Number of tiles: {table.size().getInfo()}')

if __name__ == "__main__":
    main()