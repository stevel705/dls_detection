import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory, make_response, jsonify
from werkzeug.utils import secure_filename
import urllib.request
import urllib3
import cv2
from image_processing import ImageProcessing

ip = ImageProcessing() 

UPLOAD_FOLDER = './src/uploads'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    r = open(path,'rb').read()
    image_with_boxes = ip.object_detection(r)
    retval, buffer = cv2.imencode('.jpg', image_with_boxes)
    response = make_response(buffer.tobytes())
    response.headers['Content-Type'] = 'image/jpeg'
    return response
    # return send_from_directory(app.config['UPLOAD_FOLDER'],
    #        
    #                     filename)

@app.route('/detect', methods=['GET', 'POST'])
def detect_human():
    if request.method == 'GET':
        if request.args.get('url'):
            with urllib.request.urlopen(request.args.get('url')) as url:
                image_with_boxes = ip.object_detection(url.read())
                retval, buffer = cv2.imencode('.jpg', image_with_boxes)
                response = make_response(buffer.tobytes())
                response.headers['Content-Type'] = 'image/jpeg'
                return response
                
                # return jsonify({"status": "ok", "result": ip.object_detection(url.read())}), 200
        else:
            return jsonify({"status": "bad request", "message": "Parameter url is not present"}), 400


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
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
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

if __name__ == "__main__":
    app.run(debug=True)