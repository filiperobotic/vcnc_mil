# Copyright (c) OpenMMLab. All rights reserved.
from .bbox_head import BBoxHead
from .convfc_bbox_head import (ConvFCBBoxHead, Shared2FCBBoxHead,
                               Shared4Conv1FCBBoxHead)
from .dii_head import DIIHead
from .double_bbox_head import DoubleConvFCBBoxHead
from .multi_instance_bbox_head import MultiInstanceBBoxHead
from .sabl_head import SABLHead
from .scnet_bbox_head import SCNetBBoxHead
from .convfc_bbox_head_kl_daniel import (ConvFCBBoxHeadKL, Shared2FCBBoxHeadKL, Shared4Conv1FCBBoxHeadKL)
from .convfc_bbox_head_oamil_daniel import (ConvFCBBoxHeadOAMILDANIEL, Shared2FCBBoxHeadOAMILDANIEL, Shared4Conv1FCBBoxHeadOAMILDANIEL)


__all__ = [
    'BBoxHead', 'ConvFCBBoxHead', 'Shared2FCBBoxHead',
    'Shared4Conv1FCBBoxHead', 'DoubleConvFCBBoxHead', 'SABLHead', 'DIIHead',
    'SCNetBBoxHead', 'MultiInstanceBBoxHead', 'ConvFCBBoxHeadOAMILDANIEL', 'Shared2FCBBoxHeadOAMILDANIEL', 'Shared4Conv1FCBBoxHeadOAMILDANIEL',
    'ConvFCBBoxHeadKL', 'Shared2FCBBoxHeadKL', 'Shared4Conv1FCBBoxHeadKL'
]
