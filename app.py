from flask import Flask, request,redirect, url_for, send_from_directory, render_template, flash, send_file
import pickle 
import matplotlib.pyplot as plt
import os
import tensorflow
import sys
import PIL
import numpy as np
from PIL import Image
sys.path.append("..")
from src.standardizer import Standardizer
from src.imageData_generator import ImageGenerator
from src.predict_images import LineScrubber
from werkzeug import secure_filename
this_file = os.path.realpath(__file__)
SCRIPT_DIRECTORY = os.path.split(this_file)[0]
ROOT_DIRECTORY = os.path.split(SCRIPT_DIRECTORY)[0]
DATA_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'data') 
CLASSIFICATION_DIRECTORY = os.path.join(DATA_DIRECTORY, 'classification/merged')
RESULTS_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'results') 
MODELS_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'models/models')
UPLOAD_FOLDER = os.path.join(SCRIPT_DIRECTORY, 'uploads')
CLEANED_UPLOADS = os.path.join(SCRIPT_DIRECTORY, 'cleaned_uploads')
SCRUBBED_UPLOADS = os.path.join(SCRIPT_DIRECTORY, 'scrubbed_uploads')

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


app = Flask(__name__)
app.secret_key = "lineremover"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class fileguy(object):

    def __init__(self, filename):
        self.filename = filename

globalfile = fileguy(' ')
# load the pickled model
model_path = 'models/models/random_forest.sav'

def resize(filename, width=400):
    mywidth = width

    img = Image.open(os.path.join(UPLOAD_FOLDER, filename))
#     wpercent = (mywidth/float(img.size[0]))
#     hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((mywidth,int(mywidth*1.26)), PIL.Image.ANTIALIAS)
    img.save(os.path.join(CLEANED_UPLOADS, filename))


# allowed files
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    for the_file in os.listdir(SCRUBBED_UPLOADS):
        file_path = os.path.join(SCRUBBED_UPLOADS, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)
    for the_file in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)
    for the_file in os.listdir(CLEANED_UPLOADS):
        file_path = os.path.join(CLEANED_UPLOADS, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)
    if request.method == 'POST':
        try:
            # check if the post request has the file part
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)
            file = request.files['file']
            # if user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                return redirect(url_for('uploaded_file',
                                        filename=filename))
        except:
            flash('Error')
            return redirect(request.url)
    return render_template('index.html')


@app.route('/show/<filename>', methods = ["GET", "POST"])
def uploaded_file(filename):
    print(filename)
    
    globalfile.filename = filename 
    # standardized = Standardizer(os.path.join(UPLOAD_FOLDER, filename))
    resize(filename)
    return render_template('template.html', filename=filename)

@app.route('/cleaned_uploads/<filename>')
def send_file1(filename):
    return send_from_directory(CLEANED_UPLOADS, filename)
 
@app.route('/scrubbed_uploads/<filename>')
def get_prediction(filename):
    return send_from_directory(SCRUBBED_UPLOADS, filename)

@app.route('/download', methods = ["GET", "POST"])
def downloadFile ():
    #For windows you need to use drive name [ex: F:/Example.pdf]
    filename = globalfile.filename
    print(filename)
    return send_file(os.path.join(SCRUBBED_UPLOADS, filename), as_attachment=True)
    
    # return '''
    #    <form action="/predict" method='POST'>
    #       Article 1:<br>
    #       <input type="text" name="article1"> 
    #       <br>
    #       Article 2:<br>
    #       <input type="text" name="article2"> 
    #       <br><br>
    #       <input type="submit" value="Submit for class prediction">
    #     </form>
    #     '''

@app.route('/predict', methods = ["GET", "POST"])
def predict():
    filename = globalfile.filename
    print(filename)
    image = os.path.join(CLEANED_UPLOADS, filename)
    standardizer_subset = Standardizer(image)
    gray_image = standardizer_subset.greyscale_image
    flat_arr = np.array(gray_image).flatten()
    bin_image = standardizer_subset.binarized_image
    whitespace = np.mean(np.sort(flat_arr)[::-1][100:150])
    generator = ImageGenerator(bin_image, gray_image, filename)
    generator.pad(15, whitespace)
    gray = generator.gray_padded_image
    binar = generator.bin_padded_image
    scrubber = LineScrubber(binar, gray, 0.55, whitespace, model_path, os.path.join(SCRUBBED_UPLOADS, filename))
    # request the text from the form 
    # X = [request.form['article1'], request.form['article2']]
    # y_pred = model.predict(X)
    # page = 'Here is the prediction.<br><br>'

    # go_to_home_html = '''
    #     <form action="/" >
    #         <input type="submit" value = "Upload another picture"/>
    #     </form>
    # '''
    return render_template('predict.html', filename=filename)
   

@app.route('/about', methods = ["GET", "POST"])
def about():
    return render_template("about.html")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
