import numpy as np
import numpy.typing as npt

ComplexArrayType = npt.NDArray[np.complexfloating]
FloatArrayType = npt.NDArray[np.floating]
IntArrayType = npt.NDArray[np.integer]

# TODO: These actually probably belong in the grids directory

Pixels_count_type = int | list[int]   | IntArrayType
Pixel_size_type = float | list[float] | FloatArrayType

Voxels_count_type = int | list[int]   | IntArrayType
Voxel_size_type = float | list[float] | FloatArrayType

Cartesian_grid_2d_descriptor = tuple[Pixels_count_type, Pixel_size_type]

class OutputConfiguration():
    cross_correlation_pose: bool
    integrated_likelihood_fourier: bool
    optimal_phys_pose_likelihood: bool
    optimal_fourier_pose_likelihood: bool
    optimal_pose: bool
    optimal_inplane_rotation: bool
    optimal_displacement: bool
    optimal_viewing_angle: bool

    OUTPUT_CONFIG_ARG_TO_FIELD_MAP = {
        "return_cross_correlation_pose": "cross_correlation_pose",
        "return_likelihood_integrated_pose_fourier": "integrated_likelihood_fourier",
        "return_likelihood_optimal_pose_physical": "optimal_phys_pose_likelihood",
        "return_likelihood_optimal_pose_fourier": "optimal_fourier_pose_likelihood",
        "return_optimal_pose": "optimal_pose",
        "optimized_inplane_rotation": "optimal_inplane_rotation",
        "optimized_displacement": "optimal_displacement",
        "optimized_viewing_angle": "optimal_viewing_angle",
    }

    def __init__(self,
        return_cross_correlation_pose: bool = False,
        return_likelihood_integrated_pose_fourier: bool = False,
        return_likelihood_optimal_pose_physical: bool = False, # return likelihood of optimal pose in physical space
        return_likelihood_optimal_pose_fourier: bool = False, # return likelihood of optimal pose in fourier space
        return_optimal_pose: bool = True, # return optimal pose
        optimized_inplane_rotation: bool = True, # optimize inplane rotation
        optimized_displacement: bool = True, # optimize displacement
        optimized_viewing_angle: bool = True, # optimize viewing angle,
    ):
        args: dict[str, bool] = locals()
        args.pop('self', None)
        # all_keys = OutputConfiguration.OUTPUT_CONFIG_ARG_TO_FIELD_MAP.keys()
        # keyset = set(args.keys())
        # field_map_keyset = set(all_keys)
        # assert keyset == field_map_keyset

        if all([not x for x in args.values()]):
            raise ValueError("At least one type of output must be requested.")
        for (arg, val) in args.items():
            setattr(self, OutputConfiguration.OUTPUT_CONFIG_ARG_TO_FIELD_MAP[arg], val)


    @staticmethod
    def make_all_possible_configs() -> tuple[list['OutputConfiguration'], list[list[bool]]]:
        init_arg_cnt = OutputConfiguration.__init__.__code__.co_argcount # remove 1 for self
        init_arg_cnt -= 1 # remove 'self'

        # make boolean enumeration
        s = ['0', '1']
        for _ in range(init_arg_cnt - 1):   # -1 b/c we start with length one
            tmp = [orig + '0' for orig in s]
            tmp.extend([orig + '1' for orig in s])
            s = tmp

        configs: list[OutputConfiguration] = []
        argsets: list[list[bool]] = []
        for v in s:
            if v == '0' * init_arg_cnt: continue
            args = [True if c == '1' else False for c in v]
            res = OutputConfiguration(*args)
            configs.append(res)
            argsets.append(args)
        return (configs, argsets)
