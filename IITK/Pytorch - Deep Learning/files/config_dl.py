import os
import sys
import errno
import requests
import subprocess
import shutil
from IPython.display import HTML, display
from tensorboard import manager

def tensorboard_cleanup():
    info_dir = manager._get_info_dir()
    shutil.rmtree(info_dir)

FOLDERS = {
    0: ['plots'],
    1: ['plots'],
    2: ['plots', 'data_generation', 'data_preparation', 'model_configuration', 'model_training'],
    21: ['plots', 'data_generation', 'data_preparation', 'model_configuration', 'stepbystep'],
    3: ['plots', 'stepbystep'],
}
FILENAMES = {
    0: ['chapter0.py'],
    1: ['chapter1.py'],
    2: ['chapter2.py', 'simple_linear_regression.py', 'v0.py', 'v0.py', 'v0.py'],
    21: ['chapter2_1.py', 'simple_linear_regression.py', 'v2.py', '', 'v0.py'],
    3: ['chapter3.py', 'v0.py'],
}

try:
    host = os.environ['BINDER_SERVICE_HOST']
    IS_BINDER = True
except KeyError:
    IS_BINDER = False
    
try:
    import google.colab
    IS_COLAB = True
except ModuleNotFoundError:
    IS_COLAB = False

IS_LOCAL = (not IS_BINDER) and (not IS_COLAB)

TB_LINK = ''
if IS_BINDER:
    TB_LINK = HTML('''
    <a href="" target="_blank" id="tb">Click here to open TensorBoard</a>
    <script>
        var address=document.location.href;
        a = document.getElementById('tb');
        a.href = address.substr(0, address.lastIndexOf("/")-9).concat("proxy/6006/");
    </script>
    ''')
    
