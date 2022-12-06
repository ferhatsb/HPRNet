from lib.detectors.detector_factory import detector_factory
from lib.opts import opts


def run(opt):
    Detector = detector_factory['multi_pose']
    obj = type('', (object,), {
        'default_resolution': [512, 512], 'num_classes': 1,
        'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
        'dataset': 'coco_hp', 'num_joints': 17,
        'flip_idx': [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
                     [11, 12], [13, 14], [15, 16]]})()
    opt = opts().update_dataset_info_and_set_heads(opt, obj)
    detector = Detector(opt)
    ret = detector.run('/Users/ferhat/Development/game/data_home/images/2/001469.jpg')
    results = {}
    results[0] = ret['results']
    print(results[0])


if __name__ == '__main__':
    opt = opts().parse()
    run(opt)
