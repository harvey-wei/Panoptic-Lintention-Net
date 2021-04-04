# Here is a full script for setting up mmdetection with conda and link the dataset path (supposing that your COCO dataset path is $COCO_ROOT)
# CAVEAT: For the conda activate to work properly, you should type source sample_conda.sh instead of ./sample_conda.sh or bash ./sample_conda.sh
# source executes the bash script in the current shell instead of in the sub-shell like bash or ./xx.sh.
# Start configuring by run ||source xxx_setup.sh|| in the shell !!!
conda create -n panoptic_pyconv python=3.7 -y
conda activate panoptic_pyconv

conda install -c conda-forge opencv -y
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.0 -c pytorch -y

 git clone https://github.com/Wei-Mao/panoptic_pyconv_net.git
 python -m pip install -e detectron2
# python -m pip install -e .



