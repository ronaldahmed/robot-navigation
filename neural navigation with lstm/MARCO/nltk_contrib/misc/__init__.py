# Natural Language Toolkit: Third-Party Contributions
# Miscellaneous contributions
#
# Copyright (C) 2003 The original contributors
# URL: <http://nltk.sf.net>
#
# $Id: __init__.py,v 1.1 2003/08/07 04:21:46 edloper Exp $

"""
Miscellaneous contributions to NLTK.
"""

# Add all subdirectories to our package contents path.  This lets us
# put modules in separate subdirectories without making them packages.
import nltk_contrib
nltk_contrib._add_subdirectories_to_package(__path__)
