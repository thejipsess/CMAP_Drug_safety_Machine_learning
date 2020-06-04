# CMAP_Drug_safety_Machine_learning
 predict DILI scores with machine learning from expression data.

# 1. Installation
In order to use this code you will have to setup an appropriate python environment. Within this repository there are two anaconda environment files '*.yml', which can be used to easily install all the required packages. If you don't want to use anaconda but rather install the packages through a requirements.txt, see step 1.3.
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

   # 1.2 Installation on Linux/Mac or if step 1.1 failed
      - Make sure you have Python 3 installed (this code has been tested only with python 3.6, 3.7 and 3.8)
      - navigate to the directory containing this repository including the requirements.txt file with the cd command. 
      - run the following command:
          pip3 install -r requirements.txt
          
 
# 2. The Code
The code contains one main file: ‘Assignment_jip.py’. From this file, all the other scripts/functions are called. To get an idea of what this file does, have a look at the 'schematic_overview.svg' file in the repository
