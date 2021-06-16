## Fast QAOA simulations

### Directory structure

The general approach is to have the data separate from the presentation, à-la MVC pattern.

* `data/*.(nc|json|npy)` - data files
* `data/generators/` - scripts that generate the data 
* `plots/*.ipynb` - scripts that generate figures
* `plots/pdf/` - pdf output from figure generators


### Nice tools used

You'll need these to understand and run the code

* [QTensor](https://github.com/danlkv/qtensor) - tensor network simulator with focus on MaxCut
* [Cartesian Explorer](https://github.com/danlkv/cartesian-explorer/) - a handy tool to map multi-dimensional data
* [xarray](http://xarray.pydata.org/en/stable/) - used to store datasets with their coordinates in [netcdf format](http://xarray.pydata.org/en/stable/getting-started-guide/quick-overview.html?highlight=netcdf#read-write-netcdf-files)
