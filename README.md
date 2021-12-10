## about 

- this repo implements [OrcVIO](https://moshan.cf/orcvio_githubpage/) for KITTI dataset 

## dependencies 

- same dependencies as in [VO step](https://github.com/shanmo/kitti-vo-prediction)
- [pytorch models](https://github.com/moshanATucsd/orcvio_pytorch_models), put it in `orcvio-kitti-python/third_party`
- `conda install -c conda-forge filterpy`
- `conda install -c pytorch pytorch`, test with `'1.4.0'`
- `conda install -c anaconda pandas` 

## demo 

- demo is for `odometry 06` 
- for first time run, set `load_detection_flag = False` since there are no saved detections 
- `python main.py`

## references 

- https://github.com/uoip/stereo_ptam