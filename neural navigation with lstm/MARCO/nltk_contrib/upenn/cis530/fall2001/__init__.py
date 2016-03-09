# Natural Language Toolkit: Third-Party Contributions
# Contributions from the University of Pennsylvania: CIS-530/fall2001
#
# Copyright (C) 2003 The original contributors
# URL: <http://nltk.sf.net>
#
# $Id: __init__.py,v 1.1 2003/08/07 04:35:21 edloper Exp $

"""
Student projects from the course CIS-530, taught in Fall of 2001 at
the University of Pennsylvania.
"""

# Add all subdirectories to our package contents path.  This lets us
# put modules in separate subdirectories without making them packages.
import nltk_contrib
nltk_contrib._add_subdirectories_to_package(__path__)
