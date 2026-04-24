from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS

@DATASETS.register_module()
class LIACiInsSegDataset(CocoDataset):
    METAINFO = {
        'classes': ['sea_chest_grating', 'paint_peel', 'over_board_valve', 'defect',
                    'corrosion', 'propeller', 'anode',
                    'bilge_keel', 'marine_growth', 'ship_hull'],
        'palette': [(220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
                    (0, 60, 100), (0, 80, 100), (0, 0, 230),
                    (119, 11, 32), (0, 255, 0), (0, 0, 255)]
    }