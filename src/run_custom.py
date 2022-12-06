from lib.detectors.detector_factory import detector_factory
from lib.opts import opts

import io
import numpy as np
import PIL
import requests
import torch
import openpifpaf

flip_idx = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
            [11, 12], [13, 14], [15, 16], [17, 20], [18, 21], [19, 22], [24, 25]]
face_flip_idx = [[0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11], [6, 10], [7, 9], [17, 26],
                 [18, 25], [19, 24], [20, 23], [21, 22], [31, 35], [32, 34], [36, 45], [37, 44],
                 [38, 43], [39, 42], [40, 47], [41, 46], [48, 54], [49, 53], [50, 52], [55, 59],
                 [56, 58], [60, 64], [61, 63], [65, 67]]

hand_flip_idx = [[0, 21], [1, 22], [2, 23], [3, 24], [4, 25], [5, 26], [6, 27], [7, 28],
                 [8, 29], [9, 30], [10, 31], [11, 32], [12, 33], [13, 34], [14, 35], [15, 36],
                 [16, 37], [17, 38], [18, 39], [19, 40], [20, 41]]

foot_flip_idx = [[0, 3], [1, 4], [2, 5]]


def run(opt):
    Detector = detector_factory[opt.task]
    obj = type('', (object,), {
        'default_resolution': [512, 512], 'num_classes': 1,
        'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
        'dataset': 'coco_hp', 'num_joints': 17,
        'flip_idx': flip_idx,
        'face_flip_idx': face_flip_idx,
        'hand_flip_idx': hand_flip_idx,
        'foot_flip_idx': foot_flip_idx})()
    opt = opts().update_dataset_info_and_set_heads(opt, obj)
    detector = Detector(opt)
    ret = detector.run('/home/ferhat/Development/data/images/2/001469.jpg')
    results = {}
    results[0] = ret['results']
    print(results[0])

def pifpaf():
    #image_response = requests.get('https://akhaliq-keypoint-communities.hf.space/file/soccer.jpeg')
    #pil_im = PIL.Image.open(io.BytesIO(image_response.content)).convert('RGB')

    pil_im = PIL.Image.open('/home/ferhat/Development/data/images/4/004531.jpg').convert('RGB')
    im = np.asarray(pil_im)

    # with openpifpaf.show.image_canvas(im) as ax:
    #     pass

    predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k16-wholebody', json_data=True)
    predictions, gt_anns, image_meta = predictor.pil_image(pil_im)
    #
    # annotation_painter = openpifpaf.show.AnnotationPainter()
    # with openpifpaf.show.image_canvas(im) as ax:
    #     annotation_painter.annotations(ax, predictions)

    print(predictions)
    print(len(predictions[0]['keypoints']))
    # ann_coco = openpifpaf.Annotation.from_cif_meta(
    #     openpifpaf.plugins.coco.CocoKp().head_metas[0])
    # ann_wholebody = openpifpaf.Annotation.from_cif_meta(
    #     openpifpaf.plugins.wholebody.Wholebody().head_metas[0])
    #
    # # visualize the annotation
    # openpifpaf.show.KeypointPainter.show_joint_scales = False
    # openpifpaf.show.KeypointPainter.line_width = 3
    # keypoint_painter = openpifpaf.show.KeypointPainter()
    # with openpifpaf.show.Canvas.annotation(ann_wholebody, ncols=2) as (ax1, ax2):
    #     keypoint_painter.annotation(ax1, ann_coco)
    #     keypoint_painter.annotation(ax2, ann_wholebody)


if __name__ == '__main__':
    opt = opts().parse()
    run(opt)
    #pifpaf()
