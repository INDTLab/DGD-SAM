from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

@DATASETS.register_module()
class NWPUInsSegDataset(CocoDataset):
    METAINFO = {
        'classes': ['airplane', 'ship', 'storage_tank', 'baseball_diamond',
                    'tennis_court', 'basketball_court', 'ground_track_field',
                    'harbor', 'bridge', 'vehicle'],
        'palette': [(220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
                    (0, 60, 100), (0, 80, 100), (0, 0, 230),
                    (119, 11, 32), (0, 255, 0), (0, 0, 255)]
    }


@DATASETS.register_module()
class WHUInsSegDataset(CocoDataset):
    METAINFO = {
        'classes': ['building'],
        'palette': [(0, 255, 0)]
    }


@DATASETS.register_module()
class SSDDInsSegDataset(CocoDataset):
    METAINFO = {
        'classes': ['ship'],
        'palette': [(0, 0, 255)]
    }

# @DATASETS.register_module()
# class USISInsSegDataset(CocoDataset):
#     METAINFO = {
#         'classes': ['wrecks/ruins', 'fish', 'reefs', 'aquatic plants',
#                     'human divers', 'robots', 'sea-floor',
#                     'harbor', 'bridge', 'vehicle'],
#         'palette': [(220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
#                     (0, 60, 100), (0, 80, 100), (0, 0, 230)]
#     }

@DATASETS.register_module()
class USISInsSegDataset(CocoDataset):
    """Dataset for Cityscapes."""

    METAINFO = {
        'classes': ['wrecks/ruins', 'fish', 'reefs', 'aquatic plants',
                    'human divers', 'robots', 'sea-floor'],
        'palette': [(220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
                    (0, 60, 100), (0, 80, 100), (0, 0, 230)]
    }

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

@DATASETS.register_module()
class UIISInsSegDataset(CocoDataset):
    """Dataset for Cityscapes."""

    METAINFO = {
        'classes': ['fish', 'reefs', 'aquatic plants', 'wrecks/ruins',
                    'human divers', 'robots', 'sea-floor'],
        'palette': [(0, 0, 142), (255, 0, 0), (0, 0, 70), (220, 20, 60),
                    (0, 60, 100), (0, 80, 100), (0, 0, 230)]
    }

# @DATASETS.register_module()
# class KITTI_Ins_Dataset(CocoDataset):
#     METAINFO = {
#         'classes': [
#             'road', 'sidewalk', 'building', 'wall', 'fence',
#             'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain',
#             'sky', 'person', 'rider', 'car', 'truck',
#             'bus', 'train', 'motorcycle', 'bicycle'
#         ],
#         'palette': [
#             (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
#             (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
#             (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
#             (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)
#         ]
#     }

@DATASETS.register_module()
class KITTI_Ins_Dataset(CocoDataset):
    METAINFO = {
        'classes': [
            'person', 'rider', 'car', 'truck', 'bus',
            'train', 'motorcycle', 'bicycle'
        ],
        'palette': [
            (220, 20, 60),   # person
            (255, 0, 0),     # rider
            (0, 0, 142),     # car
            (0, 0, 70),      # truck
            (0, 60, 100),    # bus
            (0, 80, 100),    # train
            (0, 0, 230),     # motorcycle
            (119, 11, 32)    # bicycle
        ]
    }

@DATASETS.register_module()
class SIS10K_Ins_Dataset(CocoDataset):
    METAINFO = {
        'classes': ['foreground'],
        'palette': [(220, 20, 60)]
    }

@DATASETS.register_module()
class UIIS10KDataset(CocoDataset):
    """Dataset for Cityscapes."""

    METAINFO = {
        'classes': ['fish', 'reptiles', 'arthropoda', 'corals', 'mollusk',
                    'plants', 'ruins', 'garbage', 'human', 'robots'],
        'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
                     (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),]
    }

@DATASETS.register_module()
class USIS16KDataset(CocoDataset):
    """Dataset for Cityscapes."""

    METAINFO = {
        'classes': ['Diver', 'Swimmer', 'MantaRay', 'ElectricRay', 'Sawfish', 'BullheadShark', 'GreatWhiteShark',
                    'WhaleShark', 'HammerheadShark', 'ThresherShark', 'SeaDragon', 'Hippocampus', 'MorayEel',
                    'ObicularBatfish', 'Lionfish', 'Trumpetfish', 'Flounder', 'Frogfish', 'Sailfish', 'EnoplosusArmatus',
                    'PseudanthiasPleurotaenia', 'Mola', 'MoorishIdol', 'BicolorAngelfish', 'AtlanticSpadefish', 'SpottedDrum',
                    'ThreespotAngelfish', 'ChromisDimidiata', 'RedseaBannerfish', 'HeniochusVarius', 'MaldivesDamselfish',
                    'ScissortailSergeant', 'FireGoby', 'Twin-spotGoby', 'Porcupinefish', 'YellowBoxfish', 'BlackspottedPuffer',
                    'BlueParrotfish', 'StoplightParrotfish', 'PomacentrusSulfureus', 'LunarFusilier', 'OcellarisClownfish',
                    'CinnamonClownfish', 'RedSeaClownfish', 'PinkAnemonefish', 'OrangeSkunkClownfish', 'GiantWrasse',
                    'SpottedWrasse', 'AnampsesTwistii', 'Blue-spottedWrasse', 'SlingjawWrasse', 'Red-breastedWrasse',
                    'PeacockGrouper', 'PotatoGrouper', 'Graysby', 'RedmouthGrouper', 'HumpbackGrouper', 'CoralHind',
                    'Porkfish', 'AnyperodonLeucogrammicus', 'WhitespottedSurgeonfish', 'Orange-bandSurgeonfish',
                    'ConvictSurgeonfish', 'SohalSurgeonfish', 'RegalBlueTang', 'LinedSurgeonfish', 'AchillesTang',
                    'PowderBlueTang', 'WhitecheekSurgeonfish', 'SaddleButterflyfish', 'MirrorButterflyfish', 'BluecheekButterflyfish',
                    'BlacktailButterflyfish', 'RaccoonButterflyfish', 'ThreadfinButterflyfish', 'EritreanButterflyfish',
                    'PyramidButterflyfish', 'CopperbandButterflyfish', 'GiantClams', 'Scallop', 'Abalone', 'QueenConch',
                    'Nautilus', 'TritonSTrumpet', 'SeaSlug', 'DumboOctopus', 'Blue-ringedOctopus', 'CommonOctopus', 'Squid',
                    'Cuttlefish', 'SeaAnemone', 'LionSManeJellyfish', 'MoonJellyfish', 'FriedEggJellyfish', 'FanCoral',
                    'ElkhornCoral', 'BrainCoral', 'SeaUrchin', 'SeaCucumber', 'Crinoid', 'OreasterReticulatus', 'ProtoreasterNodosus',
                    'KillerWhale', 'SpermWhale', 'HumpbackWhale', 'Seal', 'Manatee', 'SeaLion', 'Dolphin', 'Walrus', 'Dugong',
                    'Turtle', 'Snake', 'Homarus', 'SpinyLobster', 'CommonPrawn', 'MantisShrimp', 'KingCrab', 'HermitCrab',
                    'CancerPagurus', 'SwimmingCrab', 'SpannerCrab', 'Penguin', 'Sponge', 'PlasticBag', 'PlasticBottle',
                    'PlasticCup', 'PlasticBox', 'GlassBottle', 'SurgicalMask', 'Tyre', 'Can', 'Shipwreck', 'WreckedAircraft',
                    'WreckedCar', 'WreckedTank', 'Gun', 'Phone', 'Ring', 'Boots', 'Glasses', 'Coin', 'Statue', 'Amphora',
                    'Anchor', 'ShipSWheel', 'AUV', 'ROV', 'MilitarySubmarines', 'PersonalSubmarines', 'ShipSAnode',
                    'OverBoardValve', 'Propeller', 'SeaChestGrating', 'SubmarinePipeline', 'PipelineSAnode', 'Geoduck', 'LinckiaLaevigata', ],
        'palette': [
                    (220, 20, 60), (30, 144, 255), (0, 206, 209), (255, 140, 0), (128, 0, 128),
                    (34, 139, 34), (255, 69, 0), (70, 130, 180), (255, 105, 180), (0, 100, 0),
                    (255, 215, 0), (106, 90, 205), (0, 0, 139), (255, 0, 255), (184, 134, 11),
                    (255, 20, 147), (0, 255, 127), (139, 69, 19), (255, 160, 122), (0, 191, 255),
                    (210, 105, 30), (152, 251, 152), (205, 92, 92), (46, 139, 87), (255, 182, 193),
                    (176, 224, 230), (186, 85, 211), (95, 158, 160), (100, 149, 237), (255, 99, 71),
                    (240, 230, 140), (0, 250, 154), (199, 21, 133), (233, 150, 122), (255, 228, 196),
                    (64, 224, 208), (219, 112, 147), (72, 209, 204), (127, 255, 0), (0, 255, 255),
                    (160, 82, 45), (255, 248, 220), (176, 196, 222), (244, 164, 96), (255, 228, 225),
                    (138, 43, 226), (255, 222, 173), (255, 69, 0), (32, 178, 170), (255, 255, 0),
                    (85, 107, 47), (255, 215, 0), (147, 112, 219), (244, 164, 96), (143, 188, 143),
                    (218, 112, 214), (240, 128, 128), (255, 160, 122), (255, 20, 147), (60, 179, 113),
                    (210, 180, 140), (255, 105, 180), (106, 90, 205), (135, 206, 250), (255, 127, 80),
                    (144, 238, 144), (255, 99, 71), (32, 178, 170), (189, 183, 107), (154, 205, 50),
                    (128, 128, 0), (222, 184, 135), (70, 130, 180), (95, 158, 160), (0, 255, 127),
                    (205, 133, 63), (186, 85, 211), (255, 239, 213), (127, 255, 212), (199, 21, 133),
                    (255, 255, 224), (173, 216, 230), (240, 230, 140), (250, 128, 114), (60, 179, 113),
                    (139, 0, 0), (233, 150, 122), (100, 149, 237), (255, 215, 0), (176, 224, 230),
                    (128, 0, 128), (255, 182, 193), (255, 160, 122), (255, 140, 0), (147, 112, 219),
                    (0, 206, 209), (139, 69, 19), (255, 20, 147), (0, 128, 128), (255, 0, 0),
                    (255, 215, 0), (255, 69, 0), (0, 191, 255), (34, 139, 34), (255, 228, 181),
                    (70, 130, 180), (105, 105, 105), (186, 85, 211), (135, 206, 250), (124, 252, 0),
                    (240, 128, 128), (255, 228, 225), (221, 160, 221), (144, 238, 144), (255, 160, 122),
                    (255, 250, 205), (255, 105, 180), (72, 61, 139), (0, 128, 128), (255, 192, 203),
                    (0, 100, 0), (250, 235, 215), (0, 255, 127), (128, 128, 0), (255, 222, 173),
                    (160, 82, 45), (173, 255, 47), (0, 255, 255), (255, 215, 0), (32, 178, 170),
                    (255, 20, 147), (233, 150, 122), (255, 69, 0), (147, 112, 219), (0, 206, 209),
                    (255, 105, 180), (100, 149, 237), (255, 99, 71), (255, 182, 193), (205, 133, 63),
                    (34, 139, 34), (255, 140, 0), (0, 0, 139), (0, 255, 127), (255, 215, 0),
                    (106, 90, 205), (152, 251, 152), (0, 250, 154), (255, 228, 181),
                    (255, 250, 250),(64, 224, 208),(0, 191, 255),(173, 255, 47),(255, 160, 160),
                    (112, 128, 144),(250, 250, 210),(0, 139, 139),(255, 228, 181),
        ]
    }

@DATASETS.register_module()
class SOCDataset(CocoDataset):
    """Dataset for Cityscapes."""

    METAINFO = {
        'classes': ('sal_object', ),
        'palette': [(220, 20, 60), ]
    }

@DATASETS.register_module()
class ISODDataset(CocoDataset):
    """Dataset for Cityscapes."""

    METAINFO = {
        'classes': ('sal_object', ),
        'palette': [(220, 20, 60), ]
    }


@DATASETS.register_module()
class COMEDataset(CocoDataset):
    """Dataset for Cityscapes."""

    METAINFO = {
        'classes': ('foreground', ),
        'palette': [(220, 20, 60), ]
    }


@DATASETS.register_module()
class DSISDataset(CocoDataset):
    """Dataset for Cityscapes."""

    METAINFO = {
        'classes': ('foreground', ),
        'palette': [(220, 20, 60), ]
    }


@DATASETS.register_module()
class SIPDataset(CocoDataset):
    """Dataset for Cityscapes."""

    METAINFO = {
        'classes': ('foreground', ),
        'palette': [(220, 20, 60), ]
    }

@DATASETS.register_module()
class ZeroWasteDataset(CocoDataset):
    METAINFO = {
        'classes': ['rigid_plastic', 'cardboard', 'metal', 'soft_plastic',],
        'palette': [(220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),]
    }

@DATASETS.register_module()
class USIS10K_foreground_InsSegDataset(CocoDataset):
    METAINFO = {
        'classes': ['foreground'],
        'palette': [(220, 20, 60)]
    }