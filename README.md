# MedicalDataAugmentationTool
This tool allows on-the-fly augmentation and training for networks in the medical imaging domain. It uses [SimpleITK](http://www.simpleitk.org/) to load and augment input data, and [Tensorflow](https://www.tensorflow.org/) to define and train networks.
Some example applications are under `bin/experiments`. I will add more examples the near future.
As this framework is mainly used for research, some files are not well documented. However, I'm working on improving this.
If you have problems or find any bugs, don't hesitate to send me a message.

Andres' comments:


1. Create the .nii.gz files and put them into the TODO folder. I utilised itk-SNAP software to manually transform from dicom to nii.gz extension.
2. Remember to change the names into something as ct_test_0000_image.nii.gzz in the folder TODO. So the script reorient.py can read them. I STILL NEED TO MODIFY THIS!
3. Execute reorient.py script to create the .mha files that are used for the training
4. Remember to modify the the ct_seg_center_rai.csv inside mmwhs_dataset/setup. Not sure how to generate the coordinates automatically.
5. Execute main.py script


June 23 2019

1. To test images utilize the jupyter script test_CTA_MRI

August 5, 2019

Tensorflow: 

To install tensorflow version 1.13.1, I got problems for the version of Nvidia drivers. So, I used CUDA Toolkit 10.0 Archive (https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=CentOS&target_version=7&target_type=rpmlocal).

I install the tenxorflow preented in this links: https://github.com/tensorflow/tensorflow/issues/26182 -  https://github.com/mind/wheels/releases

But before, I did this:

pip uninstall tensorflow protobuf --yes
find $CONDA_PREFIX -name "tensorflow" | xargs -Ipkg rm -rfv pkg
pip install --ignore-installed --upgrade https://github.com/mind/wheels/releases/download/tf1.13-gpu-cuda10-tensorrt/tensorflow-1.13.1-cp36-cp36m-linux_x86_64.whl --no-cache-dir

Solved!!!


## Citation
If you use this code for your research, please cite any of our papers.

[Integrating spatial configuration into heatmap regression based CNNs for landmark localization](https://doi.org/10.1016/j.media.2019.03.007)
```
@article{Payer2019,
  title   = {Integrating spatial configuration into heatmap regression based {CNNs} for landmark localization},
  author  = {Payer, Christian and {\v{S}}tern, Darko and Bischof, Horst and Urschler, Martin},
  journal = {Medical Image Analysis},
  volume  = {54},
  year    = {2019},
  month   = {may},
  pages   = {207--219},
  doi     = {10.1016/j.media.2019.03.007},
}
```

[Instance Segmentation and Tracking with Cosine Embeddings and Recurrent Hourglass Networks](https://doi.org/10.1007/978-3-030-00934-2_1):

```
@inproceedings{Payer2018b,
  title     = {Instance Segmentation and Tracking with Cosine Embeddings and Recurrent Hourglass Networks},
  author    = {Payer, Christian and {\v{S}}tern, Darko and Neff, Thomas and Bischof, Horst and Urschler, Martin},
  booktitle = {Medical Image Computing and Computer-Assisted Intervention - {MICCAI} 2018},
  doi       = {10.1007/978-3-030-00934-2_1},
  pages     = {3--11},
  year      = {2018},
}
```

[Multi-label Whole Heart Segmentation Using CNNs and Anatomical Label Configurations](https://doi.org/10.1007/978-3-319-75541-0_20):

```
@inproceedings{Payer2018a,
  title     = {Multi-label Whole Heart Segmentation Using {CNNs} and Anatomical Label Configurations},
  author    = {Payer, Christian and {\v{S}}tern, Darko and Bischof, Horst and Urschler, Martin},
  booktitle = {Statistical Atlases and Computational Models of the Heart. ACDC and MMWHS Challenges. STACOM 2017},
  doi       = {10.1007/978-3-319-75541-0_20},
  pages     = {190--198},
  year      = {2018},
}
```

[Regressing Heatmaps for Multiple Landmark Localization Using CNNs](https://doi.org/10.1007/978-3-319-75541-0_20):

```
@inproceedings{Payer2016,
  title     = {Regressing Heatmaps for Multiple Landmark Localization Using {CNNs}},
  author    = {Payer, Christian and {\v{S}}tern, Darko and Bischof, Horst and Urschler, Martin},
  booktitle = {Medical Image Computing and Computer-Assisted Intervention - {MICCAI} 2016},
  doi       = {10.1007/978-3-319-46723-8_27},
  pages     = {230--238},
  year      = {2016},
}
```
