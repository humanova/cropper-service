import io
import utils
from models.postprocessing import RemovingTooTransparentBordersHardAndBlurringHardBorders
from models.preprocessing import BoundingBoxDetectionFastRcnn
from models.u2net import U2NET

class Cropper:
    def __init__(self):
        self.u2net = U2NET("u2net")
        self.u2netp = U2NET("u2netp")
        self.preprocessing = BoundingBoxDetectionFastRcnn()
        self.postprocessing = RemovingTooTransparentBordersHardAndBlurringHardBorders()
        pass

    def process(self, image_data, settings):
        try:
            image = image_data if settings['is_url'] else self.decode_encoded_image(image_data)
            prep = self.preprocessing if settings['preprocessing'] else None
            postp = self.postprocessing if settings['postprocessing'] else None
            main_method = (self.u2net if settings['model'] == 'u2net' else self.u2netp) if settings['model'] else self.u2net

            img = main_method.process_image(data=image, preprocessing=prep, postprocessing=postp)
            img_obj = io.BytesIO(); img.save(img_obj, "PNG"); img_obj.seek(0)
            return img_obj
        except Exception as e:
            print(e)
            return None

    def decode_encoded_image(self, encoded_image):
        return utils.decode_base64_image(encoded_image)

    def encode_image(self, image):
        return utils.encode_base64_image(image)