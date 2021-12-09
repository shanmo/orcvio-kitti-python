## about 

- this repo contains a visual odometry to provide prediction step for KITTI dataset, similar to [libviso](https://github.com/seanbow/object_pose_detection/tree/master/viso_pose)

## dependencies 

- OpenCV
- [g2opy](https://github.com/uoip/g2opy)
    - need to follow [this](https://github.com/uoip/g2opy/issues/38#issuecomment-595065792)
- [pangolin](https://github.com/uoip/pangolin)
    - `conda install -c anaconda pyopengl`
    - `conda install pybind11` 
    - need to use [this setup.py](https://github.com/shanmo/kitti-vo-prediction/issues/1)

## demo 

- folder structure in `KITTI odometry 04` should be 
```
calib.txt  image_0  image_1  image_2  image_3  times.txt
```
- run `python sptam.py` 
> Visualization for odometry 04 
![demo](assets/demo_04.gif)
- [demo for odometry 06](https://youtu.be/RsZGw0zbKSg)

## references 

- https://github.com/uoip/stereo_ptam