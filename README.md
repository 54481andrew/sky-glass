# Background

This notebook describes the ongoing development of a convolutional neural network (CNN) that identifies village-features in satellite imagery. Though this notebook is intended as a stand-alone example of machine vision, this work is a part of a larger project with the goal of forecasting the risk of disease spillover from wildlife into humans. 

As a postdoc at the University of Idaho, I develop computational pipelines that forecast the spread of viruses from wildlife into humans. Oftentimes, these zoonotic viruses occur in areas with limited health infrastructure. As a result, the extent of the risk of these viruses is often underestimated or unknown. My current work focuses on Lassa virus, an arenavirus that circulates within rodent populations in West Africa and transmits to humans that come into contact with rodent waste. Building better forecasts requires environmental features that describe the abundance of rodents that host the virus. Past field surveys have indicated that rodent populations are more prevalent in areas with houses near agricultural cultivations, for example, and less prevalent in forested areas of the village. Consequently, CNN's that extract features like houses, cultivations, forest, etc, from imagery, could provide a feature-set that allows for finescale risk prediction. 

I'll stress this a few times: **this work is my own**. I was responsible for collecting the satellite imagery from a Google API (using QGIS), creating shapefiles that described building perimeter and type, creating annotated versions of images, and of course, designing and fitting the CNN. Obviously, I learned quite a bit from online sources and peer-reviewed articles. These are cited throughout the walkthrough. If the reader is interested in using this repository's datasets, I kindly ask that they give the appropritate credit. 

&nbsp;

# Convolutional neural network design and performance

As a first step towards this goal, I have worked on a CNN that identifies buildings in satellite imagery. This CNN is based on the [U-Net image segmentation design](https://link.springer.com/chapter/10.1007%2F978-3-319-24574-4_28), using an [EfficientNet](https://arxiv.org/abs/1905.11946) as an encoder and a simple decoder. Specifically, this CNN will classify buildings as traditional hut (circular thatch structure), modern building (rectangular aluminum roof), and background. The figures below show an example of the CNN's ability to segment the different building types.  

<img align="center" src="Figures/CNN_Segmentation_Performance_2.png" alt="CNN Image Segmentation" width="700"/> &nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;



By segmenting the images in this way, I can then use openCV tools to identify groups of pixels as unique buildings. The image below shows the output of a function I wrote that groups building pixels together using contours, then overlays the contours onto the original image.  \

<img align="center" src="Figures/CNN_Contour_Performance.png" alt="Identify unique buildings with openCV" width="700"/> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

Associating pixel groups with individual pixels, in turn, allows the CNN pipeline to "count" the number of different building types in an image. The image below shows the CNN pipeline's ability to count modern buildings (left) and traditional thatch huts (right) in test images that were omitted from the training process. Generally, the CNN is able to accurately assess the number of building types in an image -- however, it is also clear that the CNN pipeline underestimates areas with high building density.   
\
&nbsp;





<img align="center" src="Figures/CNN_Building_Counts.png" alt="CNN building counts vs true building counts" width="700"/> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

# Jupyter notebooks

&nbsp;

This repository contains the jupyter notebooks, CNN model architectures, and training results of the model-building process. Unfortunately, the full dataset of satellite imagery is too large to upload to Github. As a compromise, I've included two (of 29) large rasters of data that will allow a smaller version of the image dataset to be generated, and a simpler version of the CNN model to be fit. To be clear, the images that are shown above are all from the standard, large-dataset version of the model. 

The original Jupyter notebook is titled "Building_Segmentation.ipynb", and the version of the notebook that uses the truncated dataset is titled "Small_Building_Segmentation.ipynb". Because of the reduced amount of training data, the performance of the small version of the model is significantly worse. Even so, hopefully this helps users of this repository understand how the notebook functions. 

\
&nbsp;
\
&nbsp;
\
&nbsp;
\
&nbsp;


