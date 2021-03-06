# (c) 2020 Emir Erbasan (humanova)
# Apache2 License, see LICENSE for more details

from flask import Flask, request, send_file
from cropper import Cropper

cropper_util = Cropper()

app = Flask("cropper-service", template_folder="src/templates")

@app.route('/cropping-api/crop', methods=['POST'])
def crop_image():
    settings = request.json["settings"]
    try:
        image_data = request.json["img"]
        res_img = cropper_util.process(image_data=image_data, settings=settings)
        if res_img:
            return send_file(res_img, mimetype='image/png')
        else:
            return "bad request/internal error", 400
    except Exception as e:
        print(e)
        return "bad request/internal error", 400

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')