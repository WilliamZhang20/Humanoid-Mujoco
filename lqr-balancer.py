import numpy as np
from typing import Callable, Optional, Union, List
import scipy.linalg

try:
    print('Checking that the installation succeeded:')
    import mujoco
    mujoco.MjModel.from_xml_string('<mujoco/>')
except Exception as e:
    raise e from RuntimeError(
        'Something went wrong during installation. Check the shell output above '
        'for more information.\n')

with open('humanoid.xml', 'r') as f:
    xml = f.read()