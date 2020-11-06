from PIL import Image
import io
import base64

def encode_base64_image(data):
    return base64.b64encode(data)

def decode_base64_image(encoded_data):
    i_data = base64.b64decode(encoded_data)
    return Image.open(io.BytesIO(i_data))
