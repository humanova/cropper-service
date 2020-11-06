from flask import Flask, request, send_file, jsonify
from cropper import Cropper

cropper_util = Cropper()

app = Flask("cropper-service", template_folder="src/templates")

@app.route('/cropping-api/crop', methods=['POST'])
def crop_image():
    settings = request.json["settings"]
    is_url = settings['is_url']
    try:
        image_data = request.json["img"]
        res_img = cropper_util.process(image_data=image_data, settings=settings)
        if res_img:
            return send_file(res_img, mimetype='image/png')
        else:
            raise InternalError('bad request/internal error', status_code=400)
    except Exception as e:
        print(e)
        raise InternalError('bad request/internal error', status_code=400)

@app.errorhandler(InternalError)
def handle_internal_error(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')