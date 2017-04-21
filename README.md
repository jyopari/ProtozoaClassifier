# Protozoa-Classifier
This uses the Inception V3 model to classfiy Protozoas. Running on a flask server, it uses tensorflow as its machine learning library. 

* Web.py uses the model files, which can be obtained from running this function, change the file path to the one of your training folder.  
  * bazel-bin/tensorflow/examples/image_retraining/retrain --image_dir /Users/jyopari/Desktop/Protozoa
  * This will create 2 files, the tf_files/output_graph.pb and a text file containing the labels to tf_files/output_labels.txt. Web.py uses these 2 files to classify the image. 
