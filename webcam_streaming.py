#!/usr/bin/env python
from flask import Flask, render_template, Response
from camera import Camera
import cv2

from image_processing import ImageProcessing
ip = ImageProcessing() 

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('webcam.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        image_with_boxes = ip.object_detection(frame)
        #print(type(image_with_boxes))
        retval, buffer = cv2.imencode('.jpg', image_with_boxes)
        frame = buffer.tobytes()
        #response = make_response(buffer.tobytes())
        #response.headers['Content-Type'] = 'image/jpeg'
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)