from pytest import raises, mark
from inspect import getfullargspec

PKG = "cryolike.util.types"

from cryolike.util.types import OutputConfiguration


@mark.parametrize("mode", [('normal'), ('all_false'), ('check_defaults')])
def test_output_config(mode: str):
    arg_count = OutputConfiguration.__init__.__code__.co_argcount - 1 # for the 'self'

    if mode == 'check_defaults':
        res = OutputConfiguration()
        assert res.optimal_pose
        assert res.optimal_inplane_rotation
        assert res.optimal_displacement
        assert res.optimal_viewing_angle
    elif mode == 'all_false':
        with raises(ValueError, match="At least one"):
            vals = [False] * arg_count
            _ = OutputConfiguration(*vals)
    else:
        (configs, argsets) = OutputConfiguration.make_all_possible_configs()
        for c, a in zip(configs, argsets):
            for i, arg in enumerate(OutputConfiguration.OUTPUT_CONFIG_ARG_TO_FIELD_MAP.keys()):
                actual = getattr(c, OutputConfiguration.OUTPUT_CONFIG_ARG_TO_FIELD_MAP[arg])
                assert actual == a[i]


def test_map_includes_all_fields():
    spec = getfullargspec(OutputConfiguration.__init__)
    args = spec[0]
    args = args[1:] # remove 'self'
    argset = set(args)
    configset = set(OutputConfiguration.OUTPUT_CONFIG_ARG_TO_FIELD_MAP.keys())
    assert argset == configset


def test_make_all_possible_configs_returns_correct_count():
    arg_count = OutputConfiguration.__init__.__code__.co_argcount - 1
    expected_count = 2 ** arg_count
    expected_count -= 1 # for the all-false case, which is invalid
    def list_to_str(l: list[bool]) -> str:
        s = ''
        for i in l:
            s += '0' if not i else '1'
        return s

    (configs, argsets) = OutputConfiguration.make_all_possible_configs()
    assert len(configs) == expected_count
    
    argstrs = [list_to_str(x) for x in argsets]
    # assert uniqueness of true/false configurations
    assert len(set(argstrs)) == expected_count
