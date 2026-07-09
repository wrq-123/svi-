// ============================================================
// annual_mapping_rf_TEMPLATE.js
// GEE workflow template for annual shrubland–grassland mapping
// Consistent with the manuscript preprocessing description
// ============================================================

// ------------------------------
// 0. User configuration
// ------------------------------
var CONFIG = {
  // Replace with your own assets
  roi: ee.FeatureCollection('users/your_username/maqu_roi'),
  samples: ee.FeatureCollection('users/your_username/samples_for_gee'),

  years: [2018, 2019, 2020, 2021, 2022, 2023, 2024],

  cloudThreshold: 30,
  scale: 10,
  seed: 42,

  // Export settings
  exportFolder: 'GEE_Annual_Shrubland_Mapping',
  exportCRS: 'EPSG:4326'  // replace with your actual CRS if UTM was used
};

Map.centerObject(CONFIG.roi, 9);
Map.addLayer(CONFIG.roi, {color: 'red'}, 'ROI');


// ============================================================
// 1. Sentinel-2 preprocessing and vegetation indices
//    - COPERNICUS/S2_SR_HARMONIZED
//    - QA60 + SCL masking
//    - 20 m bands resampled to 10 m using bilinear interpolation
//    - monthly maximum-value composites
// ============================================================

function maskS2(image) {
  var qa = image.select('QA60');
  var scl = image.select('SCL');

  // QA60 cloud and cirrus bits
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  var qaMask = qa.bitwiseAnd(cloudBitMask).eq(0)
    .and(qa.bitwiseAnd(cirrusBitMask).eq(0));

  // Remove cloud shadow, cloud, cirrus, and snow/ice using SCL
  var sclMask = scl.neq(3)   // cloud shadow
    .and(scl.neq(8))         // medium probability cloud
    .and(scl.neq(9))         // high probability cloud
    .and(scl.neq(10))        // cirrus
    .and(scl.neq(11));       // snow/ice

  return image
    .updateMask(qaMask.and(sclMask))
    .divide(10000)
    .resample('bilinear')
    .toFloat()
    .copyProperties(image, ['system:time_start']);
}


function addS2Indices(image) {
  var b2  = image.select('B2');
  var b3  = image.select('B3');
  var b4  = image.select('B4');
  var b8  = image.select('B8');
  var b11 = image.select('B11');
  var b12 = image.select('B12');

  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('ndvi');
  var gndvi = image.normalizedDifference(['B8', 'B3']).rename('gndvi');

  // EVI2
  var evi2 = image.expression(
    '2.5 * (NIR - RED) / (NIR + 2.4 * RED + 1)',
    {NIR: b8, RED: b4}
  ).rename('evi2');

  // MSAVI
  var msavi = image.expression(
    '(2 * NIR + 1 - sqrt(pow(2 * NIR + 1, 2) - 8 * (NIR - RED))) / 2',
    {NIR: b8, RED: b4}
  ).rename('msavi');

  // NDMI
  var ndmi = image.normalizedDifference(['B8', 'B11']).rename('ndmi');

  // NDGI
  var ndgi = image.expression(
    '((0.62 * GREEN + 0.38 * NIR) - RED) / ((0.62 * GREEN + 0.38 * NIR) + RED)',
    {GREEN: b3, NIR: b8, RED: b4}
  ).rename('ndgi');

  // NDPI
  var ndpi = image.expression(
    '(NIR - (0.78 * RED + 0.22 * SWIR1)) / (NIR + (0.78 * RED + 0.22 * SWIR1))',
    {NIR: b8, RED: b4, SWIR1: b11}
  ).rename('ndpi');

  // NDSVI
  var ndsvi = image.normalizedDifference(['B11', 'B4']).rename('ndsvi');

  // NDTI
  var ndti = image.normalizedDifference(['B11', 'B12']).rename('ndti');

  // NDLI
  var ndli = image.expression(
    '(log(1 / NIR) - log(1 / SWIR1)) / (log(1 / NIR) + log(1 / SWIR1))',
    {NIR: b8, SWIR1: b11}
  ).rename('ndli');

  return image.addBands([
    ndvi, gndvi, evi2, msavi, ndmi, ndgi, ndpi, ndsvi, ndti, ndli
  ]);
}


function getS2Collection(startDate, endDate) {
  return ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(CONFIG.roi)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CONFIG.cloudThreshold))
    .map(maskS2)
    .map(addS2Indices);
}


// Monthly maximum-value composites
function makeMonthlyS2Composites(year) {
  var startDate = ee.Date.fromYMD(year - 1, 10, 1);
  var months = ee.List.sequence(0, 12);

  var col = getS2Collection(startDate, ee.Date.fromYMD(year, 10, 31).advance(1, 'day'));

  var monthly = months.map(function(m) {
    m = ee.Number(m);
    var start = startDate.advance(m, 'month');
    var end = start.advance(1, 'month');

    var img = col
      .filterDate(start, end)
      .select(['ndvi', 'gndvi', 'evi2', 'msavi', 'ndmi', 'ndgi', 'ndpi', 'ndsvi', 'ndti', 'ndli'])
      .max()
      .set('system:time_start', start.millis())
      .set('year', year)
      .set('month_index', m);

    return img;
  });

  return ee.ImageCollection.fromImages(monthly);
}


// Placeholder: replace with your verified linear gap-filling implementation
function linearGapFill(collection) {
  // In the manuscript, missing monthly observations were gap-filled by linear interpolation.
  // Insert the verified interpolation routine used in the final analysis here.
  return collection;
}


// Placeholder: replace with your verified Savitzky–Golay implementation
function savitzkyGolaySmooth(collection, windowLength, polyOrder) {
  // In the manuscript, temporal smoothing used Savitzky–Golay
  // with window length = 5 and polynomial order = 2.
  // Insert the verified SG smoothing routine used in the final analysis here.
  return collection;
}


// Annual statistics from monthly image collection
function annualStats(collection, bandName) {
  var x = collection.select(bandName);
  var mean = x.mean().rename(bandName + '_mean');
  var min  = x.min().rename(bandName + '_min');
  var max  = x.max().rename(bandName + '_max');
  var std  = x.reduce(ee.Reducer.stdDev()).rename(bandName + '_std');
  var amp  = max.subtract(min).rename(bandName + '_amp');

  return mean.addBands([min, max, std, amp]);
}


// ============================================================
// 2. Sentinel-1 preprocessing and SAR structural descriptors
//    - COPERNICUS/S1_GRD
//    - IW mode, VV/VH, descending passes
//    - dB retained for VV/VH summary statistics
//    - linear scale used for VH/VV, RVI, and RFDI
//    - monthly mean composites
// ============================================================

function refinedLee5x5(image) {
  // Replace with your verified 5 × 5 Refined Lee implementation.
  // This placeholder returns the input image unchanged.
  // Do not use this placeholder for generating final results.
  return image;
}


function addS1Metrics(image) {
  var vv_db = image.select('VV').rename('VV');
  var vh_db = image.select('VH').rename('VH');

  // Convert dB to linear scale for ratio/index-type metrics
  var vv = ee.Image(10).pow(vv_db.divide(10)).rename('VV_lin');
  var vh = ee.Image(10).pow(vh_db.divide(10)).rename('VH_lin');

  var vh_vv = vh.divide(vv).rename('VH_VV');
  var vv_minus_vh = vv_db.subtract(vh_db).rename('VV_minus_VH');

  var rvi = vh.multiply(4).divide(vv.add(vh)).rename('RVI');
  var rfdi = vv.subtract(vh).divide(vv.add(vh)).rename('RFDI');

  return image.addBands([vh_vv, vv_minus_vh, rvi, rfdi])
    .copyProperties(image, ['system:time_start']);
}


function getS1Collection(startDate, endDate) {
  return ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(CONFIG.roi)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .select(['VV', 'VH'])
    .map(refinedLee5x5)
    .map(addS1Metrics);
}


// Monthly mean composites
function makeMonthlyS1Composites(year) {
  var startDate = ee.Date.fromYMD(year - 1, 10, 1);
  var months = ee.List.sequence(0, 12);

  var col = getS1Collection(startDate, ee.Date.fromYMD(year, 10, 31).advance(1, 'day'));

  var monthly = months.map(function(m) {
    m = ee.Number(m);
    var start = startDate.advance(m, 'month');
    var end = start.advance(1, 'month');

    var img = col
      .filterDate(start, end)
      .mean()
      .set('system:time_start', start.millis())
      .set('year', year)
      .set('month_index', m);

    return img;
  });

  return ee.ImageCollection.fromImages(monthly);
}


// 3 × 3 spatial mean filter
function spatialMean3x3(image) {
  return image.focal_mean({
    radius: 1,
    kernelType: 'square',
    units: 'pixels'
  }).copyProperties(image, ['system:time_start', 'year', 'month_index']);
}


// ============================================================
// 3. SVI* construction
//    Replace SVI_COEFS_BY_YEAR with the coefficients derived
//    from sparse logistic regression for each year.
// ============================================================

var SVI_COEFS_BY_YEAR = {
  // Example structure only.
  // Replace with your actual coefficients.
  '2018': {
    'intercept': 0,
    'mean': 0,
    'std': 1
    // 'VV_mean': ..., 'VV_min': ..., etc.
  }
};


function computeSVIStar(featureImage, year) {
  var yearStr = String(year);
  var coefs = SVI_COEFS_BY_YEAR[yearStr];

  // Replace this block with the full coefficient expression used in your analysis.
  // Example:
  // var svi = ee.Image.constant(coefs.intercept)
  //   .add(featureImage.select('VV_mean').multiply(coefs.VV_mean))
  //   .add(featureImage.select('VH_mean').multiply(coefs.VH_mean))
  //   ...
  // var sviStar = svi.subtract(coefs.mean).divide(coefs.std).rename('SVI_star');

  var sviStar = ee.Image.constant(0).rename('SVI_star');
  return sviStar;
}


// ============================================================
// 4. Feature construction for one mapping year
// ============================================================

function computeAnnualFeatures(year) {
  // Sentinel-2 monthly composites
  var s2Monthly = makeMonthlyS2Composites(year);
  s2Monthly = linearGapFill(s2Monthly);
  s2Monthly = savitzkyGolaySmooth(s2Monthly, 5, 2);

  // Sentinel-1 monthly composites
  var s1Monthly = makeMonthlyS1Composites(year);
  s1Monthly = savitzkyGolaySmooth(s1Monthly, 5, 2);
  s1Monthly = s1Monthly.map(spatialMean3x3);

  // Optical annual statistics
  var opticalBands = ['ndvi', 'gndvi', 'evi2', 'msavi', 'ndmi', 'ndgi', 'ndpi', 'ndsvi', 'ndti', 'ndli'];
  var opticalStats = ee.Image.cat(opticalBands.map(function(b) {
    return annualStats(s2Monthly, b);
  }));

  // SAR annual statistics
  var sarBands = ['VV', 'VH', 'VH_VV', 'VV_minus_VH', 'RVI', 'RFDI'];
  var sarStats = ee.Image.cat(sarBands.map(function(b) {
    return annualStats(s1Monthly, b);
  }));

  var featureImage = opticalStats.addBands(sarStats).clip(CONFIG.roi);

  // SVI*
  var sviStar = computeSVIStar(featureImage, year);
  featureImage = featureImage.addBands(sviStar);

  // Terrain variables
  var elevation = ee.Image('USGS/SRTMGL1_003').select('elevation').clip(CONFIG.roi);
  var slope = ee.Terrain.slope(elevation).rename('slope');

  // TWI example; replace with your final TWI implementation if different
  var flowAccum = ee.Image('WWF/HydroSHEDS/15ACC').select('b1').clip(CONFIG.roi);
  var twi = flowAccum.add(1).log()
    .divide(slope.multiply(Math.PI / 180).tan().add(0.001))
    .rename('TWI');

  featureImage = featureImage.addBands([elevation, slope, twi]);

  return featureImage;
}


// ============================================================
// 5. Unified vegetation mask
//    Manuscript criterion:
//    NDVI > 0.15 or EVI2 > 0.10 or VH > −18 dB or RVI > 0.8
// ============================================================

function computeAnnualVegMask(featureImage) {
  return featureImage.select('ndvi_max').gt(0.15)
    .or(featureImage.select('evi2_max').gt(0.10))
    .or(featureImage.select('VH_mean').gt(-18))
    .or(featureImage.select('RVI_mean').gt(0.8))
    .rename('vegMask');
}


var unifiedVegMask = null;
CONFIG.years.forEach(function(year) {
  var feats = computeAnnualFeatures(year);
  var mask = computeAnnualVegMask(feats);
  unifiedVegMask = (unifiedVegMask === null) ? mask : unifiedVegMask.or(mask);
});
unifiedVegMask = unifiedVegMask.rename('vegMask').clip(CONFIG.roi);
Map.addLayer(unifiedVegMask.selfMask(), {palette: ['green']}, 'Unified vegetation mask');


// ============================================================
// 6. Annual RF classification
// ============================================================

// Reduced feature set used in the manuscript
var selectedBands = [
  'SVI_star',
  'ndli_std',
  'ndgi_min',
  'evi2_mean',
  'ndpi_min',
  'ndmi_mean',
  'ndpi_std',
  'ndgi_amp',
  'msavi_min',
  'ndmi_min',
  'slope',
  'TWI'
];


// Convert sample table to point features if needed
var samples = CONFIG.samples.map(function(f) {
  return ee.Feature(
    ee.Geometry.Point([ee.Number(f.get('lon')), ee.Number(f.get('lat'))]),
    {
      'label': f.get('label'),
      'split': f.get('split')
    }
  );
}).filterBounds(CONFIG.roi);


CONFIG.years.forEach(function(year) {
  var featureImage = computeAnnualFeatures(year).updateMask(unifiedVegMask);

  var trainingSamples = samples.filter(ee.Filter.eq('split', 'train'));
  var validationSamples = samples.filter(ee.Filter.eq('split', 'val'));

  var training = featureImage.select(selectedBands).sampleRegions({
    collection: trainingSamples,
    properties: ['label'],
    scale: CONFIG.scale,
    tileScale: 4
  }).filter(ee.Filter.notNull(selectedBands));

  var rf = ee.Classifier.smileRandomForest({
    numberOfTrees: 300,
    seed: CONFIG.seed
  }).train({
    features: training,
    classProperty: 'label',
    inputProperties: selectedBands
  });

  var classified = featureImage.select(selectedBands).classify(rf).rename('classification');

  // Validation
  var val = classified.sampleRegions({
    collection: validationSamples,
    properties: ['label'],
    scale: CONFIG.scale,
    tileScale: 4
  });

  var cm = val.errorMatrix('label', 'classification');
  print('Year:', year);
  print('Confusion matrix:', cm);
  print('OA:', cm.accuracy());
  print('Kappa:', cm.kappa());

  Map.addLayer(
    classified,
    {min: 0, max: 1, palette: ['#FFD700', '#228B22']},
    'Shrubland–grassland ' + year,
    year === 2024
  );

  Export.image.toDrive({
    image: classified.toByte(),
    description: 'Maqu_shrubland_grassland_' + year,
    folder: CONFIG.exportFolder,
    region: CONFIG.roi.geometry(),
    scale: CONFIG.scale,
    crs: CONFIG.exportCRS,
    maxPixels: 1e13
  });
});

print('GEE workflow template loaded. Replace placeholder assets/functions before final execution.');
