from flask import Flask, request, jsonify, make_response, redirect, render_template, flash, url_for
from werkzeug.utils import secure_filename
import urllib.request
import os
import cv2
from image_processing import ImageProcessing

app = Flask(__name__)
ip = ImageProcessing() 

UPLOAD_FOLDER = './src/uploads'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.errorhandler(400)
def bad_request(e):
    # return also the code error
    return jsonify({"status": "not ok", "message": "this server could not understand your request"}), 400

@app.errorhandler(404)
def not_found(e):
    # return also the code error
    return jsonify({"status": "not found", "message": "route not found"}), 404

@app.errorhandler(500)
def not_found(e):
    # return also the code error
    return jsonify({"status": "internal error", "message": "internal error occurred in server"}), 500

@app.route('/detect', methods=['GET', 'POST'])
def detect_human_faces():
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
    elif request.method == 'POST':
        if request.files.get("image"):
            image_with_boxes = ip.object_detection(request.files["image"].read())
            retval, buffer = cv2.imencode('.jpg', image_with_boxes)
            response = make_response(buffer.tobytes())
            response.headers['Content-Type'] = 'image/jpeg'
            return response
            #return jsonify({"status": "ok", "result": ip.object_detection(request.files["image"].read())}), 200
        else:
            return jsonify({"status": "bad request", "message": "Parameter image is not present"}), 400
    else:
        return jsonify({"status": "failure", "message": "Method not supported for API"}), 405

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    r = open(path,'rb').read()
    image_with_boxes = ip.object_detection(r)
    retval, buffer = cv2.imencode('.jpg', image_with_boxes)
    response = make_response(buffer.tobytes())
    response.headers['Content-Type'] = 'image/jpeg'
    return response

@app.route('/upload', methods=['GET', 'POST'])
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


@app.route('/', methods=["GET"])
def info_view():
    # List of routes for this API:
    output = {
        'info': 'GET /',
        'detect faces via POST': 'POST /detect',
        'detect faces via GET': 'GET /detect',
    }
    return jsonify(output), 200

if __name__ == "__main__":
    app.run()


