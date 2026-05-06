
from osgeo import gdal
import numpy as np

src = '/home/liese2/SPRI_AI_project/Dataset/Merged_4Band/Huamuchil_original.tiff'
dst = '/home/liese2/SPRI_AI_project/Dataset/Merged_4Band/Huamuchil_4band.tiff'

ds = gdal.Open(src)
W, H = ds.RasterXSize, ds.RasterYSize

driver = gdal.GetDriverByName('GTiff')
out = driver.Create(dst, W, H, 4, gdal.GDT_UInt16)
out.SetGeoTransform(ds.GetGeoTransform())
out.SetProjection(ds.GetProjection())

for b in range(3):
    data = ds.GetRasterBand(b+1).ReadAsArray()
    out.GetRasterBand(b+1).WriteArray(data)

# Banda 4 = ceros (o copia de banda 1 si prefieres)
out.GetRasterBand(4).WriteArray(np.zeros((H, W), dtype=np.uint16))
out.FlushCache()
print(f"Guardado: {dst}  ({W}x{H}, 4 bandas)")
EOF