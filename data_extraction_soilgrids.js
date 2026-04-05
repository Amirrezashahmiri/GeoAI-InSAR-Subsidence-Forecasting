/**
 * SoilGrids 0–100 cm Multi-Band GeoTIFF Export for 8 Regions
 * Grid-aligned to ERA5-Land Monthly Aggregated product
 * One output TIFF per region
 * Only logically consistent features retained
 * Console prints value ranges for each feature
 */

// =====================================================
// 1. Define all study regions
// =====================================================
var regions = [
  {
    name: 'Isfahan',
    bounds: [51.269610, 31.896112, 52.324609, 32.516112]
  },
  {
    name: 'Jiroft',
    bounds: [56.061056, 28.177111, 56.794056, 28.528111]
  },
  {
    name: 'LakeUrmia_Tabriz',
    bounds: [44.591667, 37.086667, 46.301667, 38.350667]
  },
  {
    name: 'Marvdasht',
    bounds: [52.120221, 29.663446, 53.454220, 30.410445]
  },
  {
    name: 'Nishapur',
    bounds: [58.198555, 35.829001, 59.282555, 36.423001]
  },
  {
    name: 'Qazvin_Alborz_Tehran',
    bounds: [50.695333, 35.206000, 51.629333, 35.904000]
  },
  {
    name: 'Rafsanjan',
    bounds: [56.194276, 30.032502, 57.369275, 30.968501]
  },
  {
    name: 'Semnan',
    bounds: [53.005889, 35.303556, 53.649889, 35.679555]
  }
];

// =====================================================
// 2. Fixed ERA5 grid definition
// =====================================================
var targetCRS = 'EPSG:4326';
var targetTransform = [0.1, 0, -180.05, 0, -0.1, 90.05];
var targetScale = 11132;

// =====================================================
// 3. Soil depth configuration for 0–100 cm weighted mean
// =====================================================
var depthBands = ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm'];
var thicknesses = [5, 10, 15, 30, 40];
var totalThickness = 100;

// =====================================================
// 4. Variables to export
// Removed: cec, cfvo
// =====================================================
var variables = [
  'bdod',   // Bulk density
  'clay',   // Clay
  'phh2o',  // pH in H2O
  'sand',   // Sand
  'silt',   // Silt
  'soc'     // Soil organic carbon
];

// =====================================================
// 5. Final unit-aware band names
// =====================================================
var outputBandNames = {
  bdod: 'bdod_gcm3',
  clay: 'clay_pct',
  phh2o: 'phh2o_pH',
  sand: 'sand_pct',
  silt: 'silt_pct',
  soc: 'soc_dgkg'
};

// =====================================================
// 6. Function: compute weighted mean over 0–100 cm
// =====================================================
function getWeightedSoilGridsImage(variableName) {
  var asset = ee.Image("projects/soilgrids-isric/" + variableName + "_mean");
  var weightedSum = ee.Image(0);

  for (var i = 0; i < depthBands.length; i++) {
    var bandName = variableName + '_' + depthBands[i] + '_mean';
    var weightedBand = asset.select(bandName).multiply(thicknesses[i]);
    weightedSum = weightedSum.add(weightedBand);
  }

  return weightedSum.divide(totalThickness);
}

// =====================================================
// 7. Function: convert units and rename band
// =====================================================
function convertUnits(variableName, image) {
  if (variableName === 'bdod') {
    return image.divide(100).rename(outputBandNames[variableName]);
  }

  if (variableName === 'phh2o') {
    return image.divide(10).rename(outputBandNames[variableName]);
  }

  if (variableName === 'clay' || variableName === 'silt' || variableName === 'sand') {
    return image.divide(10).rename(outputBandNames[variableName]);
  }

  return image.rename(outputBandNames[variableName]);
}

// =====================================================
// 8. Function: print band statistics for QA/QC
// =====================================================
function printBandStats(image, regionGeom, regionName) {
  var stats = image.reduceRegion({
    reducer: ee.Reducer.min()
      .combine({
        reducer2: ee.Reducer.max(),
        sharedInputs: true
      })
      .combine({
        reducer2: ee.Reducer.mean(),
        sharedInputs: true
      }),
    geometry: regionGeom,
    scale: targetScale,
    maxPixels: 1e13,
    bestEffort: true
  });

  print('Statistics for region: ' + regionName, stats);
}

// =====================================================
// 9. Function: build aligned multiband image for one region
// =====================================================
function buildRegionSoilImage(regionGeom, regionName) {
  var images = variables.map(function(variableName) {
    var weightedImage = getWeightedSoilGridsImage(variableName);
    var convertedImage = convertUnits(variableName, weightedImage);

    var alignedImage = convertedImage
      .reduceResolution({
        reducer: ee.Reducer.mean(),
        maxPixels: 65535
      })
      .reproject({
        crs: targetCRS,
        crsTransform: targetTransform
      })
      .clip(regionGeom);

    return alignedImage;
  });

  var combinedImage = ee.Image.cat(images);

  print('Band names for ' + regionName + ':', combinedImage.bandNames());
  print('Projection for ' + regionName + ':', targetCRS, targetTransform);

  printBandStats(combinedImage, regionGeom, regionName);

  return combinedImage;
}

// =====================================================
// 10. Loop over all regions and export one TIFF per region
// =====================================================
for (var r = 0; r < regions.length; r++) {
  var regionInfo = regions[r];
  var regionGeom = ee.Geometry.Rectangle(regionInfo.bounds);

  Map.addLayer(regionGeom, {color: 'blue'}, regionInfo.name + '_Outline');

  var combinedImage = buildRegionSoilImage(regionGeom, regionInfo.name);

  Export.image.toDrive({
    image: combinedImage,
    description: 'SoilGrids_0_100cm_' + regionInfo.name + '_ERA5Grid_Filtered',
    folder: 'GEE_Subsidence_Project',
    fileNamePrefix: 'SoilGrids_' + regionInfo.name + '_0_100cm_ERA5Grid_Filtered',
    region: regionGeom,
    crs: targetCRS,
    crsTransform: targetTransform,
    fileFormat: 'GeoTIFF',
    maxPixels: 1e13
  });
}

print('All 8 region exports have been created.');
print('Depth: 0 to 100 cm (weighted mean).');
print('Grid alignment: matched to fixed ERA5-Land grid.');
print('Retained features: bdod, clay, phh2o, sand, silt, soc.');
print('Removed features: cec, cfvo.');
print('Each region will export one multi-band GeoTIFF.');
print('Band statistics (min, max, mean) are printed in the Console for QA/QC.');
print("Please go to the 'Tasks' tab and click 'Run' for all 8 tasks.");
