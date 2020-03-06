from os import system, popen
import Xlib.display # python-xlib
import PIL.Image # python-imaging
import mss
import mss.tools
import numpy as np


def get_window_coords(four_coords=True):
    txt = popen('xwininfo -id $(xdotool getactivewindow)').read()
    x_idx = txt.find('Absolute upper-left X:') + 24
    y_idx = txt.find('Absolute upper-left Y:') + 24
    
    width_idx = txt.find('Width:') + 7
    height_idx = txt.find('Height:') + 8

    x1 = int(txt[x_idx:txt.find('\n', x_idx)])
    y1 = int(txt[y_idx:txt.find('\n', y_idx)])
    
    width = int(txt[width_idx:txt.find('\n', width_idx)])
    height = int(txt[height_idx:txt.find('\n', height_idx)])
    
    if not four_coords:
        return x1, y1, width, height
    
    x2 = x1 + width
    y2 = y1 + height
        
    return x1, y1, x2, y2


def grab_area(x1, y1, x2, y2):
    with mss.mss() as sct:
        region = {'left': x1,
                  'top': y1,
                  'width': x2 - x1,
                  'height': y2 - y1}

        img = sct.grab(region)
        img = PIL.Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")
        img_arr = np.array(img)

    return img_arr


def move_mouse(x, y=None, window_offset=(0, 0)):
    if y is None:
        y = random.randint(x[1], x[3]) + window_offset[1]
        x = random.randint(x[0], x[2]) + window_offset[0]
    else:
        y += window_offset[1]
        x += window_offset[0]
    system(f'xdotool mousemove {x} {y}')


def click(x=None, y=None, window_offset=(0, 0)):
    if x is not None and y is not None:
        move_mouse(x, y, window_offset=window_offset)
    elif x is not None:
        move_mouse(x, window_offset=window_offset)
    system('xdotool click 1 &')