# CMAP_Drug_safety_Machine_learning
 predict DILI scores with machine learning from expression data.

# 1. Installation
In order to use this code you will have to setup an appropriate python environment. Within this repository there are two anaconda environment files '*.yml', which can be used to easily install all the required packages.
   # 1.1 Installation on windows 10 64-bit
      - Make sure you have Anaconda installed (anaconda.com)
      - open Anaconda prompt
      - navigate to the directory containing this repository including the *.yml files with the cd command.
      - run the following command to install the python environment with all packages:
            conda env create -f DILI_prediction_jip.yml
      - If you get an error, try the approach in step 1.2 for non-windows 10 platforms (even if you have windows 10)
      - If everything went well, run the following command to activate the new python environment:
           conda activate keras-cuda2
      - Now you're all set, you can type 'spyder' in the anaconda command prompt to start using the code.
  
   # 1.2 Installation on Linux/Mac or if step 1.1 failed
      - Make sure you have Anaconda installed (anaconda.com)
      - open Anaconda prompt
      - navigate to the directory containing this repository including the *.yml files with the cd command.
      - run the following command to install the python environment with all necessary packages:
           conda env create -f DILI_prediction_jip_multi_platform.yml
      - If everything went well, run the following command to activate the new python environment:
           conda activate keras-cuda2
      - Now you're all set, you can type 'spyder' in the anaconda command prompt to start using the code.
      
 
