import os
import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, Union, NewType, Optional

Tensor = NewType('Tensor', torch.Tensor)
Array = NewType('Array', np.ndarray)

def search_model_path(model_path: str, model_type: str) -> str:
    """ Search the smpl model path automatically

    Args:
        model_path: the user given model path
        model_type: the specified model type
    
    Return:
        The computed model path for desired smpl model
    """
    if model_path is None or model_path == '':
        ## search local directory
        model_path = f"./body_models/{model_type.lower()}/"
        if os.path.exists(model_path):
            return model_path
        
        ## search home directory
        model_path = os.path.join(
            os.environs['HOME'],
            f".body_models/{model_type.lower()}/"
        )
        if os.path.exists(model_path):
            return model_path

        raise Exception('[Not Find] the body model directory!')
    elif os.path.isfile(model_path):
        if os.path.exists(model_path):
            return model_path
        
        raise Exception('[Not Find] the body model path: {model_path}!')
    elif os.path.isdir(model_path):
        if os.path.exists(model_path):
            model_type = model_type.lower()
            if model_type not in model_path.split('/'):
                model_path = os.path.join(model_path, model_type)
            return model_path
        
        raise Exception('[Not Find] the body model directory: {model_path}!')
    else:
        raise Exception('[Not Support] the input model path: {model_path}!')

def to_numpy(array: Union[Array, Tensor], dtype=np.float32) -> np.ndarray:
    """ Convert other array type to numpy.ndarray

    Args:
        array: the array data
        dtype: specified data type
    
    Return:
        The array data in numpy.ndarray data type
    """
    if torch.is_tensor(array):
        array = array.cpu().detach().numpy()
    elif 'scipy.sparse' in str(type(array)):
        array = array.todense()
    
    return np.array(array, dtype=dtype)

def to_tensor(array: Union[Array, Tensor], dtype=torch.float32, device='cpu') -> torch.Tensor:
    """ Convert other array type to torch.Tensor

    Args:
        array: the array data
        dtype: specified data type
        device: specified data type
    
    Return:
        The array data in numpy.ndarray data type
    """
    if torch.is_tensor(array):
        return array.to(dtype=dtype, device=device)
    else:
        return torch.tensor(array, dtype=dtype, device=device)

def rot_mat_to_euler(rot_mats):
    """ Calculates rotation matrix to euler angles
    Careful for extreme cases of eular angles like [0.0, pi, 0.0]

    Args:
        rot_mats: the rotation matrix
    
    Return:
        The euler angles
    """
    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] +
                    rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)

def find_joint_kin_chain(joint_id, kinematic_tree):
    """ Find the kinematic chain of a given joint"""
    kin_chain = []
    curr_idx = joint_id
    while curr_idx != -1:
        kin_chain.append(curr_idx)
        curr_idx = kinematic_tree[curr_idx]
    return kin_chain

class Struct(object):
    """ Body model parameter structure
    """
    def __init__(self, **kwargs: Dict) -> None:
        for key, val in kwargs.items():
            setattr(self, key, val)

@dataclass
class ModelOutput(object):
    """ Body model output structure
    """
    vertices: Optional[Tensor] = None
    faces: Optional[Tensor] = None
    joints: Optional[Tensor] = None

    def keys(self):
        return self.__dict__.keys()
    
    def values(self):
        return self.__dict__.values()
    
    def items(self):
        return self.__dict__.items()

@dataclass
class SMPLOutput(ModelOutput):
    """ SMPL body model output structure
    """
    body_pose: Optional[Tensor] = None

@dataclass
class SMPLHOutput(SMPLOutput):
    """ SMPLH body model output structure
    """
    left_hand_pose: Optional[Tensor] = None
    right_hand_pose: Optional[Tensor] = None

@dataclass
class SMPLXOutput(SMPLHOutput):
    """ SMPLX body model output structure
    """
    jaw_pose: Optional[Tensor] = None
    leye_pose: Optional[Tensor] = None
    reye_pose: Optional[Tensor] = None
    landmarks: Optional[Tensor] = None