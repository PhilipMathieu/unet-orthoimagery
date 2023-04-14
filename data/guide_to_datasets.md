# Commands Used to Generate Image Chips

All datasets were generated using ArcGIS Pro 3.1 with the Image Analyst extension on Windows 11. Specifically, we used the [Export Training Data for Deep Learning](https://pro.arcgis.com/en/pro-app/latest/tool-reference/image-analyst/export-training-data-for-deep-learning.htm) tool.

## Input Datasets

- Maine Orthoimagery 2021 Layer (4-band R/G/B/NIR, 0.075m resolution)
- Maine DEM 2020 (1-band, 1m resolution)
- _Fallopia japonica_ polygon layer (created by researchers)

## Preprocessing
In order to align the Maine DEM layer with the orthoimagery, we used the `rasterio` python module to resample using cubic interpolation. In order to keep the image size reasonable, we converted data types from 32-bit float to 16-bit uint. This included stretching the dataset to maximize use of the bit depth. Recovering absolute elevation values is not possible at this time.

## Image Chip Generation
The datasets currently available consist of 128x128 pixel image chips. The image chips were generated with and without overlap (i.e. "nostride" or "overlap") and with or without chips containing no target features (i.e. "unbalanced" or "balanced").

## Folder Structure

```
data/
├── Image_Chips_[px]_[nostride|overlap]_[un|balanced]_dem/
|   ├── images/
|   |   ├── [image number].tif: 4-band raster data (16-bit float)
|   |   ├── [image number].tfw: info file with bounds of image
|   ├── images2/
|   |   ├── [image number].tif: 1-band elevation raster (16-bit float)
|   |   ├── [image number].tfw: info file with bounds of image
|   ├── labels/
|   |   ├── [image number].tif: 1-band class label raster (16-bit uint)
|   |   ├── [image number].tfw: info file with bounds of image
|   |   ├── [image number].tif.aux.xml: ArcGIS-specific raster information
|   ├── esri_accumulated_stats.json: information about the contents and parameters used to generate the model
|   ├── esri_model_definition.emd: ArcGIS-specific model information
|   ├── map.txt: one line per image chip, tab-separated, relative paths to image, dem, and label
|   ├── stats.txt: basic summary of image counts and feature counts
```

