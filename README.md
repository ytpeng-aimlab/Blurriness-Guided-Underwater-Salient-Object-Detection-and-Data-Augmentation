# FocusAugment

# U-SOD

Underwater Salient Object Detection (U-SOD) dataset, containing 1,111 underwater images with various contents, backgrounds, and watercolors. 

# Introduction
As known, training data is crucial for deep-learning-based methods to succeed. Although many SOD datasets are available for terrestrial scenes, there is little data for underwater SOD, thus lacking well-acknowledged datasets for underwater SOD. As a result, we have collected 1,011 underwater images from the images or videos of \cite{HURL}, and \cite{BVision}, and National Geographic footage~\cite{NationalGeographic}, labeling their saliency maps. The collected images have different contents, watercolors, visibility degrees, and scales of objects to cover a wide range of underwater scenes. We label these images using an annotation tool for image segmentation implemented by Amaury Br{\'e}h{\'e}ret. The figure below shows some examples for the dataset, where the first row demonstrates images with different object sizes, categories, and watercolors, and the second row shows the corresponding saliency maps we annotated.


![dataset-min](https://user-images.githubusercontent.com/56446649/158001100-1c404834-3a14-4999-9911-6e9ff4305ed6.png)

@online{HURL,
  title = {Hawaii Udersea Research Laboratory},
  url = {http://www.soest.hawaii.edu/HURL/galleries.php},
  note = {Accessed on Jun. 2019.},
  urldate = {01.06.2019}
}

@online{BVision,
  title = {Bubble Vision},
  url = {https://www.bubblevision.com/},
  note = {Accessed on Jun. 2019.},
  urldate = {01.06.2019}
}


@MISC{Breheret:2017,
author = {Amaury Br{\'e}h{\'e}ret},
title = {Pixel Annotation Tool},
howpublished = "\url{https://github.com/abreheret/PixelAnnotationTool}",
year = {2017}
}

@online{NationalGeographic,
title = {National Geographic},
  url = {https://nationalgeographic.com/},
  note = {Accessed on Jun. 2019.},
  urldate = {01.06.2019}
}
