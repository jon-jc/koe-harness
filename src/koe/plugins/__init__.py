"""Plugin discovery and lifecycle.

The kernel knows how to run a plugin; this package decides which ones exist
and whether they should be running. See :mod:`koe.plugins.loader` for the
positions that shape it -- in particular that disabling a plugin unmounts it
rather than asking it to behave as though it were off.
"""

from koe.plugins.loader import DECLARATION, STATE_FILE, PluginManager, PluginRecord

__all__ = ["DECLARATION", "STATE_FILE", "PluginManager", "PluginRecord"]
