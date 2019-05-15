from os.path import join, sep
from os import getcwd
import platform as p

# region Params PATH
if p.system() == 'Windows':
    if 'ssarusi' in getcwd():
        home_path = join('C:', sep, 'Users', 'ssarusi', 'Desktop', 'second_degree', 'semester_d', 'NLP')
else:
    home_path = join('C:', sep, 'home', 'local', 'BGU-USERS', 'shanisa', 'NLP')
# endregion

