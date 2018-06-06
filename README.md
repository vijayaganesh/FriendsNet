# FriendsNet
A Faster RCNN neural network implementing transfer learning to classify and localize characters of the TV Series Friends

In order to run the code, face scrub dataset must be downloaded from http://vintage.winklerbros.net/facescrub.html

A form must be filled and the URL to download the data will be emailed in few days.

Once the dataset is downloaded, use the following script to extract the images and faces.

https://github.com/faceteam/facescrub

Download the content of the submission to Project_3

Create a new folder Dataset inside Project_3

The desired directory Structure is:

Project_3\Dataset\Face Scrub\Actors\faces
Project_3\Dataset\Face Scrub\Actors\images
Project_3\Dataset\Face Scrub\Actress\faces
Project_3\Dataset\Face Scrub\Actress\images

## This repo has the trained model attached to it, just to check the model directly run the test script

1. Face Faster-RCNN

python keras-frcnn\test_frcnn.py --path test --config_filename config_face.pickle

2. Character Faster-RCNN with transfer learning

python keras-frcnn\test_transfer_frcnn.py --path test

The images to be tested should be placed in the Project_3\test directory
Output can be seen in the Project_3\Output directory


## Run the following only if you need to do the training process again
## WARNING: Training process takes more than 20 hours, this repo has the final model attached

Run the jupyter notebook script(Resnet50 base network)
Project_3\ArrayPickleDump.ipynb
Project_3\Friendsnet Resnet50.ipynb

Run the jupyter notebook script (Pascal VOC Creation)
Project_3\BBoxToPascalVoc.ipynb
Project_3\PascalVocToDataParser.ipynb

Run the Faster RCNN Training Script:

python keras-frcnn\train_frcnn.py --path final_data.p --parser simple --input_weight_path model_frcnn.hdf5 --config_filename config_face.pickle

Run the transfer RCNN Training Script:

python keras-frcnn\train_transfer_frcnn.py --path final_data.p --parser simple --input_weight_path model_frcnn.hdf5 --output_weight_path transfer_model.hdf5


## Once the training is done, use the test commands given above
