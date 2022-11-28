from camera_stream import CameraStream
from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)
camera = CameraStream()

def gen(camera):
    camera.start()
    while(True):
        ret, frame = camera.read()
        if not ret:
            print("Can't receive frame...")
            break
        ret, jpeg = cv2.imencode('.jpg',frame)
        if not ret:
            print("Bad imencode operation...")
            break
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video_feed')
def video_feed():
    camera.stop()
    return Response(gen(camera),
        mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, threaded=True)