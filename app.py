from flask import Flask, request, jsonify, make_response, redirect, render_template
import urllib.request
# import urllib3.request
import cv2
from image_processing import ImageProcessing

app = Flask(__name__)
ip = ImageProcessing() 

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
        # return render_template("index.html")
        if request.files.get("image"):
            return jsonify({"status": "ok", "result": ip.object_detection(request.files["image"].read())}), 200
        else:
            return jsonify({"status": "bad request", "message": "Parameter image is not present"}), 400
    else:
        return jsonify({"status": "failure", "message": "PUT method not supported for API"}), 405

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
    app.run(debug=True)


