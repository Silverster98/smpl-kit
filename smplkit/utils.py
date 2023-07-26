import torch
import numpy as np
from typing import Any, Union, Tuple, List, Dict, Optional
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

from .misc import ModelOutput
from .constants import VERTEX_NUM, CONTACT_VERTEX_IDS, CONTACT_PART_NAME, KEY_VERTEX_IDS

## Some useful functions
def matrix_to_parameter(
        T: torch.Tensor,
        trans: torch.Tensor,
        orient: torch.Tensor,
        pelvis: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Convert vertice transformation matrix to smpl trans and orient parameter. The given T should be column matrices.

    Reference: https://www.dropbox.com/scl/fi/zkatuv5shs8d4tlwr8ecc/Change-parameters-to-new-coordinate-system.paper?dl=0&rlkey=lotq1sh6wzkmyttisc05h0in0

    Args:
        T: target transformation matrix, shape is <4, 4>, should be column matrices
        trans: origin trans of smpl parameters, shape is <N, 3>
        orient: origin orient of smpl parameters, shape is <N, 3>
        pelvis: origin pelvis, shape is <N, 3>
    
    Return:
        Transformed trans and orient parameters
    """
    R = T[0:3, 0:3]
    t = T[0:3, -1]

    pelvis = pelvis - trans
    trans = torch.matmul(R, (trans + pelvis).T).T - pelvis

    rot_mats = axis_angle_to_matrix(orient) # pytorch3d uses row matrices
    rot_mats = torch.matmul(R, rot_mats)
    orient = matrix_to_axis_angle(rot_mats)

    return trans + t, orient

def compute_orient(
    body: Union[torch.Tensor, np.ndarray, ModelOutput],
):
    """ Compute the body orientation with given body joints
    """
    pass

def compute_normal(
    body: Union[torch.Tensor, np.ndarray, ModelOutput],
    faces: Optional[Union[torch.Tensor, np.ndarray]]=None,
):
    """ Compute the body normal with given body vertices and faces
    """
    pass

def compute_sdf(
    
):
    """ Signed Distance Function
    """
    pass


## Some useful classes
class VertexSelector():

    @staticmethod
    def select_vertex(verts: Union[torch.Tensor, np.ndarray], index: Union[List, Dict, Tuple]):
        """ Select specific vertices with a given index

        Args:
            verts: the vertices tensor or array, shape is <..., N, 3>, where N is the number of vertices
            index: the index of vertices to be selected, can be a list, dict or tuple
        
        Return:
            The selected vertices tensor or array.
        """
        if isinstance(index, list):
            index = index
        elif isinstance(index, dict):
            index =  index.values()
        elif isinstance(index, tuple):
            index = [t[1] for t in index]
        else:
            raise Exception('[Type Error] Index type not supported! Please use list, dict or tuple.')
        
        return verts[..., index, :]

    @staticmethod
    def contact_vertex(
        body: Union[torch.Tensor, np.ndarray, ModelOutput],
        index: Optional[Union[List, Dict, Tuple]]=None,
    ):
        """ Select contact vertices from a given body

        Args:
            body: the output a body model, can be a tensor or array or ModelOutput. If it is tensor or array, the shape is <..., N, 3>, where N is the number of vertices
            index: the index of contact vertices, default is None, which means all contact vertices will be selected with default CONTACT_VERTEX_IDS
        
        Return:
            The selected contact vertices tensor or array.
        """
        if isinstance(body, ModelOutput):
            verts = body.vertices
        else:
            verts = body
        
        nverts = verts.shape[-2]
        if nverts == VERTEX_NUM.SMPL: # SMPL and SMPL+H have same body vertex number
            if index is None:
                index = []
                for part in CONTACT_PART_NAME:
                    index += CONTACT_VERTEX_IDS.SMPL[part]['verts_ind']
        elif nverts == VERTEX_NUM.SMPLX:
            if index is None:
                index = []
                for part in CONTACT_PART_NAME:
                    index += CONTACT_VERTEX_IDS.SMPLX[part]['verts_ind']
        else:
            raise Exception('[Length Error] The number of vertex is not supported! Please input an array with 6890 vertices for SMPL(+H) or 10475 for SMPL-X.')
        
        return VertexSelector.select_vertex(verts, index)
        
    @staticmethod
    def key_vertex(
        body: Union[torch.Tensor, np.ndarray, ModelOutput],
        index: Optional[Union[List, Dict, Tuple]]=None
    ):
        """ Select key vertices from a given body

        In smplx https://github.com/vchoutas/smplx/blob/main/smplx/vertex_joint_selector.py, the key vertices are selected by VertexJointSelector and are concated with the joints as a output.
        In our implementation, we provide this utility function to select key vertices from a given body, which is more flexible.

        Args:
            body: the output a body model, can be a tensor or array or ModelOutput. If it is tensor or array, the shape is <..., N, 3>, where N is the number of vertices
            index: the index of key vertices, default is None, which means all key vertices will be selected with default KEY_VERTICES_INDEX
        
        Return:
            The selected key vertices tensor or array.
        """
        if isinstance(body, ModelOutput):
            verts = body.vertices
        else:
            verts = body
        
        nverts = verts.shape[-2]
        if nverts == VERTEX_NUM.SMPL:
            if index is None:
                index = KEY_VERTEX_IDS.SMPL
        elif nverts == VERTEX_NUM.SMPLX:
            if index is None:
                index = KEY_VERTEX_IDS.SMPLX
        else:
            raise Exception('[Length Error] The number of vertex is not supported! Please input an array with 6890 vertices for SMPL(+H) or 10475 for SMPL-X.')

        return VertexSelector.select_vertex(verts, index)

def _singleton(cls):
    _instance = {}

    def inner(*args, **kwargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]
    return inner

@_singleton
class BodyModel():
    def __init__(self, model_type: str, *args, **kwargs):
        """ Create a singleton body model. """
        self.reset(model_type, *args, **kwargs)
    
    def __call__(self, *args, **kwargs) -> Any:
        pass
    
    def reset(self, model_type: Optional[str]=None, *args, **kwargs):
        """ Reset the body model. """
        if model_type is None or model_type == '':
            assert self.model is not None, '[Value Error] Body model type not specified! Please use smpl, smplh or smplx.'
            model_type = self.model.name
        
        if model_type.lower() == 'smpl':
            from .layers import SMPLLayer as SMPL
            self.model = SMPL(*args, **kwargs)
        elif model_type.lower() == 'smplh':
            from .layers import SMPLLayer as SMPLH
            self.model = SMPLH(*args, **kwargs)
        elif model_type.lower() == 'smplx':
            from .layers import SMPLXLayer as SMPLX
            self.model = SMPLX(*args, **kwargs)
        else:
            raise Exception('[Value Error] Body model type not supported! Please use smpl, smplh or smplx.')
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def to(self, **kwargs):
        return self.model.to(**kwargs)
