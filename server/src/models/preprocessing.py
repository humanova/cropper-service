"""
Name: Pre-processing class file
Description: This file contains pre-processing classes.
Version: [release][3.2]
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Modified by humanova (humanova)[https://github.com/humanova].
Changes:
1. Removed method_detect function.
2. Moved __init__ imports to the top of the file.
2. Removed BoundingBoxDetectionWithMaskMaskRcnn class.
3. Removed MaskRcnn class.
-- humanova
License: Apache License 2.0
License:
   Copyright 2020 OPHoperHPO
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
from PIL import Image
from gluoncv import model_zoo, data
from mxnet import nd

import time
import numpy as np

import logging
logger = logging.getLogger(__name__)

class FastRcnn:
    """
    Fast Rcnn Neural Network to detect objects in the photo.
    """

    def __init__(self):
        self.model_zoo = model_zoo
        self.data = data
        self.nd = nd
        logger.debug("Loading Fast RCNN neural network")
        self.__net__ = self.model_zoo.get_model('faster_rcnn_resnet50_v1b_voc',
                                                pretrained=True)  # Download the pre-trained model, if one is missing.
        # noinspection PyUnresolvedReferences
        self.class_names = self.__net__.classes

    def __load_image__(self, data_input):
        """
        Loads an image file for other processing
        :param data_input: Path to image file or PIL image
        :return: image
        """
        if isinstance(data_input, str):
            try:
                data_input = Image.open(data_input)
                # Fix https://github.com/OPHoperHPO/image-background-remove-tool/issues/19
                data_input = data_input.convert("RGB")
                image = np.array(data_input)  # Convert PIL image to numpy arr
            except IOError:
                logger.error('Cannot retrieve image. Please check file: ' + data_input)
                return False, False
        else:
            # Fix https://github.com/OPHoperHPO/image-background-remove-tool/issues/19
            data_input = data_input.convert("RGB")
            image = np.array(data_input)  # Convert PIL image to numpy arr
        x, resized_image = self.data.transforms.presets.rcnn.transform_test(self.nd.array(image))
        return x, image, resized_image

    def process_image(self, image):
        """
        Detects objects in the photo and returns their names, borders.
        :param image: Path to image or PIL image.
        :return: original pil image, resized pil image, dict(ids, scores, bboxes)
        """
        start_time = time.time()  # Time counter
        x, image, resized_image = self.__load_image__(image)
        ids, scores, bboxes = [xx[0].asnumpy() for xx in self.__net__(x)]
        logger.debug("Finished! Time spent: {}".format(time.time() - start_time))
        return image, resized_image, {"ids": ids, "scores": scores, "bboxes": bboxes}


class BoundingBoxDetectionFastRcnn:
    """
    Class for the image preprocessing method.
    This image pre-processing technique uses two neural networks ($used_model and Fast RCNN)
    to first detect the boundaries of objects in a photograph,
    cut them out, sequentially remove the background from each object in turn
    and subsequently collect the entire image from separate parts
    """

    def __init__(self):
        self.__fast_rcnn__ = FastRcnn()
        self.model = None
        self.prep_image = None
        self.orig_image = None

    @staticmethod
    def trans_paste(bg_img, fg_img, box=(0, 0)):
        """
        Inserts an image into another image while maintaining transparency.
        :param bg_img: Background pil image
        :param fg_img: Foreground pil image
        :param box: Bounding box
        :return: Pil Image
        """
        fg_img_trans = Image.new("RGBA", bg_img.size)
        fg_img_trans.paste(fg_img, box, mask=fg_img)
        new_img = Image.alpha_composite(bg_img, fg_img_trans)
        return new_img

    @staticmethod
    def __orig_object_border__(border, orig_image, resized_image, indent=16):
        """
        Rescales the bounding box of an object
        :param indent: The boundary of the object will expand by this value.
        :param border: array consisting of the coordinates of the boundaries of the object
        :param orig_image: original pil image
        :param resized_image: resized image ndarray
        :return: tuple consisting of the coordinates of the boundaries of the object
        """
        x_factor = resized_image.shape[1] / orig_image.size[0]
        y_factor = resized_image.shape[0] / orig_image.size[1]
        xmin, ymin, xmax, ymax = [int(x) for x in border]
        if ymin < 0:
            ymin = 0
        if ymax > resized_image.shape[0]:
            ymax = resized_image.shape[0]
        if xmax > resized_image.shape[1]:
            xmax = resized_image.shape[1]
        if xmin < 0:
            xmin = 0
        if x_factor == 0:
            x_factor = 1
        if y_factor == 0:
            y_factor = 1
        border = (int(xmin / x_factor) - indent,
                  int(ymin / y_factor) - indent, int(xmax / x_factor) + indent, int(ymax / y_factor) + indent)
        return border

    def run(self, model, prep_image, orig_image):
        """
        Runs an image preprocessing algorithm to improve background removal quality.
        :param model: The class of the neural network used to remove the background.
        :param prep_image: Prepared for the neural network image
        :param orig_image: Source image
        :returns: Image without background
        """
        _, resized_image, results = self.__fast_rcnn__.process_image(orig_image)

        classes = self.__fast_rcnn__.class_names
        bboxes = results['bboxes']
        ids = results['ids']
        scores = results['scores']

        object_num = len(bboxes)  # We get the number of all objects in the photo

        if object_num < 1:  # If there are no objects, or they are not found,
            # we try to remove the background using standard tools
            return model.__get_output__(prep_image, orig_image)
        else:
            # Check that all arrays match each other in size
            if ids is not None and not len(bboxes) == len(ids):
                return model.__get_output__(prep_image,
                                            orig_image)  # we try to remove the background using standard tools
            if scores is not None and not len(bboxes) == len(scores):
                return model.__get_output__(prep_image, orig_image)
                # we try to remove the background using standard tools
        objects = []
        for i, bbox in enumerate(bboxes):
            if scores is not None and scores.flat[i] < 0.5:
                continue
            if ids is not None and ids.flat[i] < 0:
                continue
            object_cls_id = int(ids.flat[i]) if ids is not None else -1
            if classes is not None and object_cls_id < len(classes):
                object_label = classes[object_cls_id]
            else:
                object_label = str(object_cls_id) if object_cls_id >= 0 else ''
            object_border = self.__orig_object_border__(bbox, orig_image, resized_image)
            objects.append([object_label, object_border])
        if objects:
            if len(objects) == 1:
                return model.__get_output__(prep_image, orig_image)
                # we try to remove the background using standard tools
            else:
                obj_images = []
                for obj in objects:
                    border = obj[1]
                    obj_crop = orig_image.crop(border)
                    # TODO: make a special algorithm to improve the removal of background from images with people.
                    if obj[0] == "person":
                        obj_img = model.process_image(obj_crop)
                    else:
                        obj_img = model.process_image(obj_crop)
                    obj_images.append([obj_img, obj])
                image = Image.new("RGBA", orig_image.size)
                for obj in obj_images:
                    image = self.trans_paste(image, obj[0], obj[1][1])
                return image
        else:
            return model.__get_output__(prep_image, orig_image)