import numpy as np


from cryolike.stacks import Templates, Images

def validate_operation(templates: Templates, images: Images):
    if not templates.polar_grid.uniform:
        raise NotImplementedError("Non-uniform polar grid is not yet supported.")
    if not np.isclose(templates.box_size[0], templates.box_size[1], rtol=1e-6):
        raise NotImplementedError("Box size must be same in both dimensions")
    ## TODO: make sure the polar grids of the templates and images are compatible
