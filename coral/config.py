"""
This Python module contains utilities to parse CoRAL configuration files.
Most of this code is copied from PyRate (see https://github.com/GeoscienceAustralia/PyRate)
"""
from typing import Dict
import os


# constants for lookups
#: STR; Name of input interferogram list file
ASC_LIST = 'backscatter_data_asc'
DESC_LIST = 'backscatter_data_desc'
ASC_CR_FILE_ORIG = 'cr_file_asc'
ASC_CR_FILE_NEW = 'cr_file_asc_new'
DESC_CR_FILE_ORIG = 'cr_file_desc'
DESC_CR_FILE_NEW = 'cr_file_desc_new'
OUT_DIR = 'path_out'
TARG_WIN_SZ = 'target_window_size'
CLT_WIN_SZ = 'clutter_window_size'
SUB_IM = 'sub_image_size'


# Lookup to help convert args to correct type/defaults
# format is	key : (conversion, default value)
# None = no conversion
PARAM_CONVERSION = {
    TARG_WIN_SZ: (int, 3),
    CLT_WIN_SZ: (int, 7),
    SUB_IM: (int, 51),
}

PATHS = [
    ASC_LIST,
    DESC_LIST,
    ASC_CR_FILE_ORIG,
    ASC_CR_FILE_NEW,
    DESC_CR_FILE_ORIG,
    DESC_CR_FILE_NEW,
    OUT_DIR,
]


def get_config_params(path: str) -> Dict:
    """
    Reads the parameters file provided by the user and converts it into
    a dictionary.

    Args:
        path: Absolute path to the parameters file.
    Returns:
       A dictionary of parameters.
    """
    txt = ''
    with open(path, 'r') as inputFile:
        for line in inputFile:
            if any(x in line for x in PATHS):
                pos = line.find('~')
                if pos != -1:
                    # create expanded line
                    line = line[:pos] + os.environ['HOME'] + line[(pos + 1):]
            txt += line
    params = _parse_conf_file(txt)

    return params


def _parse_conf_file(content) -> Dict:
    """
    Converts the parameters from their text form into a dictionary.

    Args:
        content: Parameters as text.

    Returns:
        A dictionary of parameters.
    """

    def _is_valid(line):
        """
        Check if line is not empty or has % or #
        """
        return line != "" and line[0] not in "%#"

    lines = [ln.split() for ln in content.split('\n') if _is_valid(ln)]

    # convert "field:   value" lines to [field, value]
    kvpair = [(e[0].rstrip(":"), e[1]) for e in lines if len(e) == 2] + \
             [(e[0].rstrip(":"), None) for e in lines if len(e) == 1]
    parameters = dict(kvpair)
    for p in PATHS:
        if p not in parameters:
            parameters[p] = None

    if not parameters:
        raise ConfigException('Cannot parse any parameters from config file')

    return _parse_pars(parameters)


# todo: check why conversion of parameters to int is not working properly
def _parse_pars(pars) -> Dict:
    """
    Takes dictionary of parameters, converting values to required type
    and providing defaults for missing values.

    Args:
        pars: Parameters dictionary.

    Returns:
        Dictionary of converted (and optionally validated) parameters.
    """
    # Fallback to default for missing values and perform conversion.
    for k in PARAM_CONVERSION:
        if pars.get(k) is None:
            pars[k] = PARAM_CONVERSION[k][1]
            # _logger.warning(f"No value found for parameter '{k}'. Using "f"default value {pars[k]}.")
        else:
            conversion_func = PARAM_CONVERSION[k][0]
            if conversion_func:
                try:
                    pars[k] = conversion_func(pars[k])
                except ValueError as e:
                    _logger.error(
                        f"Unable to convert '{k}': {pars[k]} to " f"expected type {conversion_func.__name__}.")
                    raise e

    return pars


class ConfigException(Exception):
    """
    Default exception class for configuration errors.
    """
