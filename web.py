import os
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
import time
import tensorflow as tf

#UPLOAD_FOLDER = '/root/images/'
UPLOAD_FOLDER = 'static/'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timename = time.time()
            timename = str(timename)+'.jpg'
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], timename))
            image_path = 'static/'+timename
            image_data = tf.gfile.FastGFile(image_path, 'rb').read()
            label_lines = [line.rstrip() for line
                in tf.gfile.GFile("/root/test/flask-hello-world/retrained_labels.txt")]
            with tf.gfile.FastGFile("/root/test/flask-hello-world/retrained_graph.pb", 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(graph_def, name='')

            with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
              softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

              predictions = sess.run(softmax_tensor, \
                       {'DecodeJpeg/contents:0': image_data})
              top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    # Sort to show labels of first prediction in order of confidence
              top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
              l = []
              for node_id in top_k:
                  human_string = label_lines[node_id]
                  score = predictions[0][node_id]
                  #score = format(score, '.4f')
                  #print((human_string, score))
                  l.append([human_string+'&nbsp; (',str(score*100)+'%)'])
              return '<title> Top 3 predictions: </title><h3> Top 3 Predictions from Tensorflow</h3> <hr><br> <br> <img src=\'/static/' + timename+ '\'/><h4>' +  l[0][0]+ l[0][1] + ' <br> ' + l[1][0]+l[1$
              #return "your top 3 predictions are    " +l[0][0] + l[0][1] + '   ' +l[1][0]+l[1][1] + '   ' l[2][0]+l[2][1]

    return '''
    <!doctype html>
    <title>Protozoa Classifier</title>
    <h3>Protozoa Classifier using Machine Learning Models from Tensorflow</h3>
    <p> By <a href="https://twitter.com/scienceinnyc"> Evie Alexander </a>, biology teacher in NYC and <a href="https://www.twitter.com/jyo_pari"> Jyo Pari </a>, student in Acton Boxborough Regional High$
<br> 
<img src='/static/1490219700.01.jpg' /><br>
    <p> <a href="http://104.131.102.162:8000/static/output.html">Demo: classification on protozoa shown above</a> </p>

    <p> Input images in .jpg format only, and try to have only one protozoa in the image for best results, know that this isn't perfect and might give you the wrong answer but by adding in images you are$
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
<br>
Here are the possible classifications, it will get bigger over time </p>
    <ul>
  <li>aeolosome</li>
  <li>ankistrodesmus</li>
  <li>cladocera</li>
  <li>coleps</li>
  <li>copepod</li>
  <li>cosmarium</li>
  <li>cyclidum</li>
  <li>daphnia</li>
  <li>dileptus</li>
  <li>euplotes</li>
  <li>frontonia</li>
  <li>ostracod</li>
  <li>peranema</li>
  <li>rotifers</li>
  <li>spirostomum</li>
  <li>staurastrum</li>
  <li>stylonychia</li>
  <li>synedra</li>
</ul>
    '''
