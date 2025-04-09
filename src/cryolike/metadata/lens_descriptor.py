import numpy as np
from typing import Any, NamedTuple, Literal
from pydantic import BaseModel, ConfigDict

from cryolike.util import FloatArrayType, IntArrayType, to_float_flatten_np_array, extract_unique_float
from .star_file import read_star_file


class RelionField(NamedTuple):
    relion_field: str
    descriptor_field: str
    description: str
    default: float | None | tuple[Literal['expand'], float]
    defaultable: bool
    required: bool


RELION_FIELDS: list[RelionField] = [
    RelionField('DefocusU', 'defocusU', '', None, False, True),
    RelionField('DefocusV', 'defocusV', '', None, False, True),
    RelionField('DefocusAngle', 'defocusAngle', '', None, False, True),
    RelionField('OriginXAngst', 'originXAngst', '', 0.0, True, False),
    RelionField('OriginYAngst', 'originYAngst', '', 0.0, True, False),
    RelionField('PhaseShift', 'phaseShift', 'CTF phase shift', ('expand', 0.), True, True),
    RelionField('SphericalAberration', 'sphericalAberration', '', None, False, True),
    RelionField('Voltage', 'voltage', '', None, False, True),
    RelionField('AmplitudeContrast', 'amplitudeContrast', 'CTF amplitude contrast', 0.1, True, True),
    RelionField('AngleRot', 'angleRotation', '', None, False, False),
    RelionField('AngleTilt', 'angleTilt', '', None, False, False),
    RelionField('AnglePsi', 'anglePsi', '', None, False, False),
    RelionField('ImagePixelSize', 'ref_pixel_size', '', None, False, False),
    RelionField('CtfBfactor', 'ctfBfactor', 'CTF B-factor', 0., True, True),
    RelionField('CtfScalefactor', 'ctfScalefactor', 'CTF scale factor', 1., True, True),
]

RELION_ANGLE_FIELDS = ['AngleRot', 'AngleTilt', 'AnglePsi']
STAR_ANGLE_FIELDS = ['PhaseShift', 'DefocusAngle']
ALL_ANGLE_FIELDS = []
ALL_ANGLE_FIELDS.extend(RELION_ANGLE_FIELDS)
ALL_ANGLE_FIELDS.extend(STAR_ANGLE_FIELDS)


class BatchableLensFields(NamedTuple):
    defocusU: FloatArrayType
    defocusV: FloatArrayType
    defocusAngle: FloatArrayType
    originXAngst: FloatArrayType
    originYAngst: FloatArrayType
    phaseShift: FloatArrayType
    angleRotation: FloatArrayType | None
    angleTilt: FloatArrayType | None
    anglePsi: FloatArrayType | None


class SerializedLensDescriptor(BaseModel):
    defocusU: FloatArrayType
    defocusV: FloatArrayType
    defocusAngle: FloatArrayType
    originXAngst: FloatArrayType
    originYAngst: FloatArrayType
    phaseShift: FloatArrayType | float
    sphericalAberration: float
    voltage: float
    amplitudeContrast: float
    angleRotation: FloatArrayType | None
    angleTilt: FloatArrayType | None
    anglePsi: FloatArrayType | None
    ref_pixel_size: FloatArrayType | None
    files: np.ndarray | None
    idxs: IntArrayType | None
    ctfBfactor: FloatArrayType | float | None
    ctfScalefactor: FloatArrayType | float | None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class LensDescriptor():
    """Class describing the properties of a device, to be used to compute
    the relevant contrast transfer function (CTF).
    
    Attributes:
        defocusU (FloatArrayType): Defocus in U-dimension, in Angstrom
        defocusV (FloatArrayType): Defocus in V-dimension, in Angstrom
        defocusAngle (FloatArrayType): Defocus angle, in radians
        originYAngst (FloatArrayType): x-shift from origin of particle.
        originYAngst (FloatArrayType): y-shift from origin of particle.
        phaseShift (FloatArrayType): phase shift, in radians
        sphericalAberration (float): Spherical aberration, in mm
        voltage (float): Voltage, in kV
        amplitudeContrast (float): Amplitude contrast
        angleRotation (FloatArrayType | None): Optional rotational angle
        angleTilt (FloatArrayType | None): Optional rotational angle
        anglePsi (FloatArrayType | None): Optional rotational angle
        ref_pixel_size (FloatArrayType | None): Recorded pixel size from
            a descriptor file
        files (np.ndarray | None): List of image files referenced in
            a combined Cryosparc descriptor file
        idxs (IntArrayType | None): Lookup table for interpreting the
            images described by a combined Cryosparc descriptor file
        ctfBfactor (float | FloatArrayType | None): B-factor of CTF,
            typically from a Starfile. For accounting/reference only.
        ctfScalefactor (float | FloatArrayType | None): Scale factor of
            CTF, typically from a Starfile. For accounting/reference only.
    """
    defocusU: FloatArrayType
    defocusV: FloatArrayType
    defocusAngle: FloatArrayType
    originXAngst: FloatArrayType
    originYAngst: FloatArrayType
    phaseShift: FloatArrayType
    sphericalAberration: float
    voltage: float
    amplitudeContrast: float
    angleRotation: FloatArrayType | None
    angleTilt: FloatArrayType | None
    anglePsi: FloatArrayType | None
    ref_pixel_size: FloatArrayType | None
    files: np.ndarray | None
    idxs: IntArrayType | None
    ctfBfactor: float | FloatArrayType | None
    ctfScalefactor: float | FloatArrayType | None


    def __init__(self, *,
        defocusU: float | np.ndarray = np.array([10200.0]), # these defaults are weird. 
        defocusV: float | np.ndarray = np.array([9800.0]),
        defocusAngle: float | np.ndarray = np.array([90.0]),
        originXAngst: float | np.ndarray = np.array([0.0]),
        originYAngst: float | np.ndarray = np.array([0.0]),
        phaseShift: float | np.ndarray = np.array([0.0]),
        sphericalAberration: float | FloatArrayType = 2.7,
        voltage: float | FloatArrayType = 300.0,
        amplitudeContrast: float | FloatArrayType = 0.1,
        angleRotation: FloatArrayType | None = None,
        angleTilt: FloatArrayType | None = None,
        anglePsi: FloatArrayType | None = None,
        ref_pixel_size: FloatArrayType | None = None,
        defocusAngle_degree: bool = True,
        phaseShift_degree: bool = True,
        files: np.ndarray | None = None,
        idxs: IntArrayType | None = None,
        ctfBfactor: float | FloatArrayType | None = None,
        ctfScalefactor: float | FloatArrayType | None = None
    ):
        """Constructor for device properties used to compute a CTF.

        Args:
            defocusU (float | np.ndarray, optional): In Angstroms. Defaults to 10200.0.
            defocusV (float | np.ndarray, optional): In Angstroms. Defaults to 9800.0.
            defocusAngle (float | np.ndarray, optional): Defocus angle, in degrees
                unless otherwise specified. Defaults to np.array([90.0]).
            phaseShift (float | np.ndarray, optional): Phase shift, in degrees unless
                otherwise specified. Defaults to 0.0.
            sphericalAberration (float | FloatArrayType, optional): Spherical aberration.
                Defaults to 2.7.
            voltage (float | FloatArrayType, optional): Voltage, in kV. Defaults to 300.0.
            amplitudeContrast (float | FloatArrayType, optional): Amplitude contrast.
                Defaults to 0.1.
            angleRotation (FloatArrayType | None): Optional per-image rotational angle
            angleTilt (FloatArrayType | None): Optional per-image rotational angle
            anglePsi (FloatArrayType | None): Optional per-image rotational angle
            ref_pixel_size: (FloatArrayType | None, optional): If set, indicates the pixel
                size recorded from a cryosparc descriptor file.
            defocusAngle_degree (bool, optional): If True (the default), defocus angle is
                presumed to be in degrees; if False, defocus angle is treated as radians.
            phaseShift_degree (bool, optional): If True, phase shift is presumed to be in
                degrees; if False, phase shift is in radians. Defaults to True.
            files (np.ndarray | None, optional): If set, contains the file list from
                a combined cryosparc descriptor file. Only used in particle stack conversion.
            idxs: (IntArrayType | None, optional): If set, indicates how to match the
                data from a combined cryosparc descriptor file to the image files. Only used
                in particle stack conversion.
            ctfBfactor (float | FloatArrayType | None, optional): If set, records the
                B-factor from a starfile. This is not used in any computation, but will
                be preserved for reference.
            ctfScalefactor (float | FloatArrayType | None, optional): If set, records the
                scale factor from a starfile. This is not used in any computation, but
                will be preserved for reference.
        """
        self.defocusU = to_float_flatten_np_array(defocusU)
        self.defocusV = to_float_flatten_np_array(defocusV)

        self.originXAngst = to_float_flatten_np_array(originXAngst)
        self.originYAngst = to_float_flatten_np_array(originYAngst)

        if self.defocusU.size != self.defocusV.size:
            raise ValueError('defocusU and defocusV must have the same size')
        self.defocusAngle = to_float_flatten_np_array(defocusAngle)
        if defocusAngle_degree:
            self.defocusAngle = np.radians(self.defocusAngle)
        self.phaseShift = to_float_flatten_np_array(phaseShift)
        if phaseShift_degree:
            self.phaseShift = np.radians(self.phaseShift)
        if self.phaseShift.size == 1:
            n_CTF = len(self.defocusU)
            self.phaseShift = self.phaseShift * np.ones(n_CTF)

        self.sphericalAberration = extract_unique_float(sphericalAberration, "spherical aberration")
        self.voltage = extract_unique_float(voltage, "voltage")
        self.amplitudeContrast = extract_unique_float(amplitudeContrast, "amplitude contrast")
        self.angleRotation = angleRotation
        self.angleTilt = angleTilt
        self.anglePsi = anglePsi
        self.ref_pixel_size = ref_pixel_size
        self.files = files
        self.idxs = idxs
        self.ctfBfactor = ctfBfactor
        self.ctfScalefactor = ctfScalefactor


    def batch_whole(self) -> BatchableLensFields:
        """Returns a tuple of all the per-image descriptors available.

        Returns:
            BatchableLensFields: This LensDescriptor's defocusU, defocusV,
                defocus angle, and phase shift parameters. 
        """
        return BatchableLensFields(
            self.defocusU,
            self.defocusV,
            self.defocusAngle,
            self.originXAngst,
            self.originYAngst,
            self.phaseShift,
            self.angleRotation,
            self.angleTilt,
            self.anglePsi
        )


    def get_slice(self, start: int, end: int) -> BatchableLensFields:
        """Returns a tuple of per-image descriptors for a particular
        range, without removing it from the descriptor.

        Args:
            start (int): Beginning index (inclusive)
            end (int): Ending index (exclusive)

        Returns:
            BatchableLensFields: Defocus and phase shfit data for the
                indicated range of images.
        """
        ar = None if self.angleRotation is None else self.angleRotation[start:end]
        at = None if self.angleTilt is None else self.angleTilt[start:end]
        ap = None if self.anglePsi is None else self.anglePsi[start:end]
        return BatchableLensFields(
            self.defocusU[start:end],
            self.defocusV[start:end],
            self.defocusAngle[start:end],
            self.originXAngst[start:end],
            self.originYAngst[start:end],
            self.phaseShift[start:end],
            ar,
            at,
            ap
        )


    def get_selections(self, selections) -> BatchableLensFields:
        """Returns a tuple of per-image descriptors for selected
        records, identified by index.

        Args:
            selections (IntArrayType): Indices to return

        Returns:
            BatchableLensFields: Defocus and phase shift data for
                the specifically requested indices.
        """
        ar = None if self.angleRotation is None else self.angleRotation[selections]
        at = None if self.angleTilt is None else self.angleTilt[selections]
        ap = None if self.anglePsi is None else self.anglePsi[selections]
        return BatchableLensFields(
            self.defocusU[selections],
            self.defocusV[selections],
            self.defocusAngle[selections],
            self.phaseShift[selections],
            ar,
            at,
            ap
        )


    def serialize(self) -> SerializedLensDescriptor:
        return SerializedLensDescriptor(
            defocusU = self.defocusU,
            defocusV = self.defocusV,
            defocusAngle = self.defocusAngle,
            originXAngst = self.originXAngst,
            originYAngst = self.originYAngst,
            phaseShift = self.phaseShift,
            sphericalAberration = self.sphericalAberration,
            voltage = self.voltage,
            amplitudeContrast = self.amplitudeContrast,
            angleRotation = self.angleRotation,
            angleTilt = self.angleTilt,
            anglePsi = self.anglePsi,
            ref_pixel_size = self.ref_pixel_size,
            files = self.files,
            idxs = self.idxs,
            ctfBfactor = self.ctfBfactor,
            ctfScalefactor = self.ctfScalefactor
        )


    def to_dict(self) -> dict:
        return self.serialize().model_dump()


    @classmethod
    def from_dict(cls, d: dict[str, Any], angles_in_degrees: bool = False):
        # TODO: queries about voltage, amp cont, phase types.
        data = SerializedLensDescriptor.model_validate(d)
        return cls(
            defocusU=data.defocusU,
            defocusV=data.defocusV,
            defocusAngle=data.defocusAngle,
            phaseShift=data.phaseShift,
            sphericalAberration=data.sphericalAberration,
            voltage=data.voltage,
            amplitudeContrast=data.amplitudeContrast,
            angleRotation=data.angleRotation,
            angleTilt=data.angleTilt,
            anglePsi=data.anglePsi,
            ref_pixel_size=data.ref_pixel_size,
            defocusAngle_degree=angles_in_degrees,
            phaseShift_degree=angles_in_degrees,
            files=data.files,
            idxs=data.idxs,
            ctfBfactor=data.ctfBfactor,
            ctfScalefactor=data.ctfScalefactor
        )


    @classmethod
    def from_cryosparc_file(cls, file: str, get_fs_data: bool = False):
        """Create a LensDescriptor from a Cryosparc file, which may optionally
        have internal filesystem data indexing images in MRC files.

        Args:
            file (str): The path to the Cryosparc file
            get_fs_data (bool, optional): If set, assume we are operating
                on a Cryosparc file with MRC indexing. In this case, we will
                look for a "blob/path" member identifying the MRC file paths and
                a "blob/idx" member mapping the records in the Cryosparc file to
                the image records. We will only attempt to read pixel size data
                from the Cryosparc file when get_fs_data is True. Defaults to False.

        Returns:
            LensDescriptor: A LensDescriptor read from the specified Cryosparc file.
        """
        # prior versions have also considered looking at:
        # - image_shape = cs_data['blob/shape']
        # - ctfBfactor = cs_data['ctf/bfactor_A2']
        # - ctfScalefactor = cs_data['ctf/scale']
        # - uid = cs_data['uid']

        cs_data = np.load(file)
        defocusU = cs_data['ctf/df1_A']
        defocusV = cs_data['ctf/df2_A']
        defocusAngle = cs_data['ctf/df_angle_rad']
        sphericalAberration = cs_data['ctf/cs_mm'][0]
        voltage = cs_data['ctf/accel_kv'][0]
        amplitudeContrast = cs_data['ctf/amp_contrast'][0]
        phaseShift = cs_data['ctf/phase_shift_rad']

        if get_fs_data:
            files = cs_data['blob/path']
            idxs = cs_data['blob/idx']
            try:
                pixel_size = cs_data['blob/psize_A'][0]
            except:
                pixel_size = None
        else:
            files = None
            idxs = None
            pixel_size = None

        descriptor = cls(
            defocusU=defocusU,
            defocusV=defocusV,
            defocusAngle=defocusAngle,
            phaseShift=phaseShift,
            sphericalAberration=sphericalAberration,
            voltage=voltage,
            amplitudeContrast=amplitudeContrast,
            ref_pixel_size = pixel_size,
            defocusAngle_degree=False,
            phaseShift_degree=False,
            files = files,
            idxs = idxs,
        )
        return descriptor

    # TODO: Rewrite following the model of indexed_starfile
    @classmethod
    def from_starfile(cls, star_file: str, defocus_is_degree: bool = True, phase_shift_is_degree: bool = True):
        """Read a Starfile and generate a LensDescriptor object

        Args:
            star_file (str): Path to the Starfile
            defocus_is_degree (bool, optional): Whether the Starfile stores the
                DefocusAngle property in degrees. Defaults to True.
            phase_shift_is_degree (bool, optional): Whether the Starfile stores the
                PhaseShift property in degrees. Defaults to True.

        Returns:
            LensDescriptor: A LensDescriptor representation of the apparatus data
                from the given Starfile.
        """
        dataBlock, _ = read_star_file(star_file)

        defocusU = dataBlock["DefocusU"]
        defocusV = dataBlock["DefocusV"]
        print('checking keys')
        if 'OriginXAngst' in dataBlock.keys() and 'OriginYAngst' in dataBlock.keys():
            originXAngst = dataBlock["OriginXAngst"]
            originYAngst = dataBlock["OriginYAngst"]
        else:
            originXAngst = np.zeros(len(dataBlock['DefocusU']))
            originYAngst = np.zeros(len(dataBlock['DefocusV']))

        defocusAngle = dataBlock["DefocusAngle"]
        sphericalAberration = dataBlock["SphericalAberration"]
        voltage = dataBlock["Voltage"]
        try:
            amplitudeContrast = dataBlock["AmplitudeContrast"]
        except:
            print("Warning: CTF amplitude contrast not found, using default value 0.1")
            amplitudeContrast = 0.1
        try:
            phaseShift = dataBlock["PhaseShift"]
        except:
            print("Warning: CTF phase shift not found, using default value 0.0")
            phaseShift = np.zeros_like(defocusU)
        try:
            ctfBfactor = dataBlock["CtfBfactor"]
        except:
            print("Warning: CTF B-factor not found, using default value 0.0")
            ctfBfactor = 0.0
        try:
            ctfScalefactor = dataBlock["CtfScalefactor"]
        except:
            print("Warning: CTF scale factor not found, using default value 1.0")
            ctfScalefactor = 1.0
        
        descriptor = cls(
            defocusU=defocusU,
            defocusV=defocusV,
            defocusAngle=defocusAngle,
            originXAngst=originXAngst,
            originYAngst=originYAngst,
            phaseShift=phaseShift,
            sphericalAberration=sphericalAberration,
            voltage=voltage,
            amplitudeContrast=amplitudeContrast,
            defocusAngle_degree=defocus_is_degree,
            phaseShift_degree=phase_shift_is_degree,
            ctfBfactor=ctfBfactor,
            ctfScalefactor=ctfScalefactor
        )
        return descriptor


    @classmethod
    def from_indexed_starfile(cls, star_file: str, persist_angles: bool = True):
        dataBlock, fields = read_star_file(star_file)
        _check_required_relion_fields(fields, star_file)
        raw_dict = {}
        shape_record = dataBlock['DefocusU']

        for x in RELION_FIELDS:
            if x.relion_field in fields:
                if not persist_angles and x.relion_field in RELION_ANGLE_FIELDS:
                    continue
                v = dataBlock[x.relion_field]
                raw_dict[x.descriptor_field] = np.radians(v) if x.relion_field in ALL_ANGLE_FIELDS else v
            elif x.defaultable:
                _handle_default(x, raw_dict, shape_record)

        files_list: list[str] = []
        indices_list: list[int] = []
        # This needs to be fixed. We should be using starfile for everything. 

        # Note: arbitrary splitting of potentially-irregular-length strings
        # on an arbitrary-positioned character is unlikely to benefit too
        # much from vectorized operations.
        # BUT if we can be certain that the int component is always a particular
        # length, we could maybe do this much faster. TODO
        # (Though I have doubts about whether the brittleness would be worth it)
        for fn in dataBlock['ImageName']:
            assert isinstance(fn, str)
            (idx, pathstr) = fn.split('@')
            idx = int(idx) - 1 # starfiles use one-based indexing for images
            files_list.append(pathstr)
            indices_list.append(idx)
        raw_dict['files'] = np.array(files_list)
        raw_dict['idxs'] = np.array(indices_list)
        base_desc = cls(**raw_dict, phaseShift_degree=False, defocusAngle_degree=False)
        if base_desc.ref_pixel_size is not None:
            base_desc.ref_pixel_size = np.unique(base_desc.ref_pixel_size)
            if len(base_desc.ref_pixel_size) > 1:
                raise ValueError("Multiple pixel sizes found in starfile")

        return base_desc


def _check_required_relion_fields(fields: list[str], filename: str):
    for x in RELION_FIELDS:
        if (not x.required or x.defaultable): continue
        if x.relion_field not in fields:
            raise ValueError(f"Unable to parse Relion-formatted starfile {filename}: field {x.relion_field} is missing.")


def _handle_default(x: RelionField, raw_dict: dict, shape_record: np.ndarray):
    if isinstance(x.default, tuple):
        # If this fails, we don't recognize this type of default
        assert x.default[0] == "expand"
        def_val = x.default[1]
        val = np.ones_like(shape_record) * def_val
    else:
        val = x.default
        def_val = val
    print(f"Warning: {x.description} not found, using default value {def_val}")
    raw_dict[x.descriptor_field] = val
