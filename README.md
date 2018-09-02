# tiny-yolo-objection-detection
This repository was forked from https://github.com/nilboy/tensorflow-yolo and trained on own data.
## dependence
 - tensorflow >=1.0
 - numpy
 - OpenCV >=3.0

## my contributions
 - added data augmentation in the training process (yolo/dataset/text_dataset.py: line 161 )
 - modified the demo.py for multi-object detection with non-maximum-suppression (demo_image.py, demo_video.py)
 - added the evaluation code to compute the precision and recall (demo_dir_for_pr.py, evaluation.py). 
 - added the code prepared for mAP computation (for mAP evaluation, please refer to https://github.com/Cartucho/mAP )
 - trained on own data with the pretrained tiny yolo model:  https://drive.google.com/file/d/0B-yiAeTLLamRekxqVE01Yi1RRlk/view?usp=sharing
 
 ## train
1. **download the dataset** \
   the url of the dataset is in ./data/dataset_url.txt

2. **prepare the data** \
   generate the annotation (a text file) of the images for training, the code is in ./tools/preprocess_stick_cup_pen.py
3. **train the network with .cfg file**
   ```shell
   python tools/train.py -c conf/train.cfg
   ```
## evaluation
 - demo_image.py: test an image
 - demo_video.py: test a video
 - demo_dir_for_pr.py: test the images in specified directory (this code will generate a text file for precision and recall computation)
 - demo_dir_for_mAP.py: test the images in specified directory (this code will generate a text file for mAP computation)
 - evaluation.py: compute the precisiona and recall
 - mAP computation: refer to https://github.com/Cartucho/mAP 

   
    
   
 
 
