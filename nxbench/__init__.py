from nxbench._version import __version__

from .data import *
from .validation import *
from .viz import *
from .log import initialize_logging

__packagename__ = "nxbench"
__url__ = "https://github.com/dpys/nxbench"

DOWNLOAD_URL = f"https://github.com/dpys/{__packagename__}/archive/{__version__}.tar.gz"

initialize_logging()
