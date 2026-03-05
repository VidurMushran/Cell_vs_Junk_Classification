#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 16:59:08 2025

@author: Mushran
"""

import os
os.chdir('/mnt/deepstore/Vidur/Junk Classification/data/')
import extract_event_images

oligo_junk = 'oligo_junk.hdf5'
rare_cells_annotated = os.listdir('rare_cells_annotated')
