from camera_stream import CameraStream
from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

def gen(camera):
    while(True):
        frame = camera.read()
        ret, jpeg = cv2.imencode('.jpg',frame)
        if jpeg is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        else:
            print("Frame is none")

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video_feed')
def video_feed():
    return Response(gen(CameraStream().start()),
        mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, threaded=True)