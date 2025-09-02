### Multi-E2E: An end-to-end urban land-use mapping framework integrating high-resolution remote sensing images and multi-source geographical data

------------

This is an official implementation of Multi-E2E framework.

![Multi-E2E_visual_encoder](/assets/Multi-E2E.jpg)

#### Requirements

- pytorch >= 2.4.1
- python >= 3.8
- dgl >= 1.1.2

#### Instruction

``main.py`` is the program entry used to train the Multi-E2E framework. You can configure your own multi-source land-use dataset and use the following script for experiment.

```python
python main.py
```

The ``data`` directory contains samples required to run the code. The information represented by each file is as follows: 

- ``example_DT_GraphData``：file to store POI graph of parcels in the following format

  ``````
  {Number of POI graphs}
  {Number of nodes} {Parcel gt} {Parcel id}
  {POI category} {Number of edges} {Index of the connected POIs}
  ...
  ``````

- ``example_DT_GraphIndexData``：file to store the ID of parcels containing POIs
- ``example_label_gt``：file recording ground truth of parcels
- ``graph_type_15``：file storing the correspondence between land-use categories and quantitative values

#### Citation

-------

If you use Multi-E2E in your research, please cite our paper.


