import data_utils as du

dataset = du.open_grib_file("data.grib")

du.export_images(dataset, "data/1sttest")
