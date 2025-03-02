#!/usr/bin/env python3
#
# vision2b.py

import pygame as pg
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

pg.init()
windowSize = (1920,1080)
pg.display.set_mode(display, DOUBLEBUF|OPENGL)


