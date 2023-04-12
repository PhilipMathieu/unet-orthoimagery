# Commands Used to Generate Image Chips

Commands were run in ArcGIS Pro 3.1 with the Image Analyst extension on Windows 11.

## Image_Chips_20230410
- 64x64 chips with 32 pixel spacing between rows and columns
- chips included even if they don't contain any of the target class.
- `with` is setting the environment to only consider extent of training data

```python
with arcpy.EnvManager(extent='399000 4824125 402000 4829000 PROJCS["NAD_1983_2011_UTM_Zone_19N",GEOGCS["GCS_NAD_1983_2011",DATUM["D_NAD_1983_2011",SPHEROID["GRS_1980",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["False_Easting",500000.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",-69.0],PARAMETER["Scale_Factor",0.9996],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]'):
    arcpy.ia.ExportTrainingDataForDeepLearning(
        in_raster="raster_merge",
        out_folder=r"C:\Users\phili\Documents\ArcGIS\Projects\CS5330_Final_Project\Image_Chips_64_overlap_balanced",
        in_class_data="Invasives",
        image_chip_format="TIFF",
        tile_size_x=64,
        tile_size_y=64,
        stride_x=96,
        stride_y=96,
        output_nofeature_tiles="ALL_TILES",
        metadata_format="Classified_Tiles",
        start_index=0,
        class_value_field="SpeciesN",
        buffer_radius=0,
        in_mask_polygons="Bounds",
        rotation_angle=0,
        reference_system="MAP_SPACE",
        processing_mode="PROCESS_AS_MOSAICKED_IMAGE",
        blacken_around_feature="NO_BLACKEN",
        crop_mode="FIXED_SIZE",
        in_raster2=None,
        in_instance_data=None,
        instance_class_value_field=None,
        min_polygon_overlap_ratio=0
    )
```

## Image_Chips_64_overlap_balanced
- 64x64 chips with -32 pixel spacing between rows and columns (i.e., 50% overlap)
- chips included only if they contain some of the target class.

```python
arcpy.ia.ExportTrainingDataForDeepLearning(
    in_raster="raster_merge",
    out_folder=r"C:\Users\phili\Documents\ArcGIS\Projects\CS5330_Final_Project\Image_Chips_64_overlap_balanced",
    in_class_data="Invasives",
    image_chip_format="TIFF",
    tile_size_x=64,
    tile_size_y=64,
    stride_x=32,
    stride_y=128,
    output_nofeature_tiles="ONLY_TILES_WITH_FEATURES",
    metadata_format="Classified_Tiles",
    start_index=0,
    class_value_field="SpeciesN",
    buffer_radius=0,
    in_mask_polygons="Bounds",
    rotation_angle=0,
    reference_system="MAP_SPACE",
    processing_mode="PROCESS_AS_MOSAICKED_IMAGE",
    blacken_around_feature="NO_BLACKEN",
    crop_mode="FIXED_SIZE",
    in_raster2=None,
    in_instance_data=None,
    instance_class_value_field=None,
    min_polygon_overlap_ratio=0
)
```

## Image_Chips_64_overlap_unbalanced
- 64x64 chips with -32 pixel spacing between rows and columns (i.e., 50% overlap)
- chips included even if they don't contain any of the target class.

```python
arcpy.ia.ExportTrainingDataForDeepLearning(
    in_raster="raster_merge",
    out_folder=r"C:\Users\phili\Documents\ArcGIS\Projects\CS5330_Final_Project\Image_Chips_64_overlap_unbalanced",
    in_class_data="Invasives",
    image_chip_format="TIFF",
    tile_size_x=64,
    tile_size_y=64,
    stride_x=32,
    stride_y=128,
    output_nofeature_tiles="ALL_TILES",
    metadata_format="Classified_Tiles",
    start_index=0,
    class_value_field="SpeciesN",
    buffer_radius=0,
    in_mask_polygons="Bounds",
    rotation_angle=0,
    reference_system="MAP_SPACE",
    processing_mode="PROCESS_AS_MOSAICKED_IMAGE",
    blacken_around_feature="NO_BLACKEN",
    crop_mode="FIXED_SIZE",
    in_raster2=None,
    in_instance_data=None,
    instance_class_value_field=None,
    min_polygon_overlap_ratio=0
)
```