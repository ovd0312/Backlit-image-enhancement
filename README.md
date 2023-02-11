# Backlit-image-enhancement
This repository contains the code for backlit image enhancement, based on the research paper - M. Akai, Y. Ueda, T. Koga and N. Suetake, "A Single Backlit Image Enhancement Method For Improvement Of Visibility Of Dark Part".

Backlit images are images where the light source is behind the subject of the photograph. This results in an overly bright background and a dark subject, which may be undesirable at times. This solution attempts to enhance the image by using the following algorithm.

## The Algorithm 
<ul>
  <li>Generate an enhanced image by applying the histogram equalization and gamma correction on the image and merging the results.</li>
  <li>Generate a weight map by applying Otsu threshold on the image and smoothing it using guided filter</li>
  <li>Merge the input image and enhanced image using alpha blending, according to the weight map generated.</li>
</ul>

I have also included a change of using k-means thresholding instead of Otsu, which provides significantly better results. You can choose what technique to use in main.py.

## Requirements
Make sure `opencv2` is installed for python.

## How to run
<ul>
  <li>Navigate to the directory of the repo on command line.</li>
   <li>Ensure that the input path for images is as you need in main.py.</li>
  <li>Run 
    
    python main.py
  </li>
 
</ul>
