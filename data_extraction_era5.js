/**
 * ERA5-Land Monthly Multi-Band GeoTIFF Export
 * Project: Subsidence Analysis - Lake Urmia / Tabriz, Iran
 * Period: Oct 2014 - Dec 2024
 */

// 1. Define Study Area (Updated based on Lake Urmia/Tabriz HDF5 Metadata)
// Longitude: [44.591667 to 46.301667]
// Latitude:  [37.086667 to 38.350667]
var region = ee.Geometry.Rectangle([44.591667, 37.086667, 46.301667, 38.350667]);
Map.centerObject(region, 8);
Map.addLayer(region, {color: 'blue'}, 'Lake Urmia - Tabriz Study Area Outline');

// 2. Define the list of comprehensive bands
var bands = [
  // --- Hydrological Cycle (Groundwater Proxy) ---
  'total_precipitation_sum',       // [m] Accumulated liquid and frozen water
  'total_evaporation_sum',         // [m of water equivalent] Water leaving the system (Negative)
  'runoff_sum',                    // [m] Surface and sub-surface drainage
  'volumetric_soil_water_layer_1', // [m3/m3] Soil moisture (0-7cm)
  'volumetric_soil_water_layer_2', // [m3/m3] Soil moisture (7-28cm)
  'volumetric_soil_water_layer_3', // [m3/m3] Soil moisture (28-100cm)
  'volumetric_soil_water_layer_4', // [m3/m3] Soil moisture (100-289cm - Deepest)
  
  // --- Thermal & Energy (Land Surface Deformation) ---
  'temperature_2m',                // [K] Air temperature at 2m height
  'skin_temperature',              // [K] Temperature of the surface of the Earth
  'soil_temperature_level_1',      // [K] Soil temp (0-7cm)
  'soil_temperature_level_4',      // [K] Soil temp (100-289cm)
  'surface_net_solar_radiation_sum', // [J/m2] Shortwave radiation reaching the surface
  'surface_sensible_heat_flux_sum',  // [J/m2] Heat transfer between surface and atmosphere
  
  // --- Atmospheric & Vegetation (Secondary Drivers) ---
  'surface_pressure',              // [Pa] Weight of all the air in a column
  'u_component_of_wind_10m',       // [m/s] Eastward wind speed
  'v_component_of_wind_10m',       // [m/s] Northward wind speed
  'dewpoint_temperature_2m',       // [K] Measure of humidity at 2m
  'leaf_area_index_high_vegetation', // [m2/m2] Green leaf area for high veg
  'leaf_area_index_low_vegetation'   // [m2/m2] Green leaf area for low veg
];

// 3. Load and Filter Collection
// Aligned to start from Oct 2014 to match Lake Urmia InSAR start date
var era5 = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
    .filterBounds(region)
    .filterDate('2014-10-01', '2025-01-01') 
    .select(bands);

// 4. Create a Multi-Temporal Stack
var stackedImage = era5.iterate(function(img, stack) {
  var dateStr = ee.Image(img).date().format('YYYYMM');
  var bandList = ee.List(bands);
  var renamedBands = bandList.map(function(b) {
    return ee.String(b).cat('_').cat(dateStr);
  });
  var renamedImg = ee.Image(img).rename(renamedBands);
  return ee.Image(stack).addBands(renamedImg);
}, ee.Image().select()); // Initial empty image

// 5. Export to Google Drive as GeoTIFF
Export.image.toDrive({
  image: ee.Image(stackedImage),
  description: 'ERA5_Comprehensive_Stack_LakeUrmia_2014_2024',
  folder: 'GEE_Subsidence_Project',
  fileNamePrefix: 'ERA5_LakeUrmia_MultiTemporal',
  region: region,
  scale: 11132, 
  fileFormat: 'GeoTIFF',
  maxPixels: 1e13 
});

print("Updated for Lake Urmia / Tabriz region.");
print("Time Period: Oct 2014 to Dec 2024.");
print("Please go to the 'Tasks' tab and click 'Run'.");
