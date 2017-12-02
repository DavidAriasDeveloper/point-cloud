#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Generación de nube de puntos a traves de imagenes estereoscopicas
Por: Luis David Arias Manjarrez
Asignatura: High Performance Computing
Universidad Tecnológica de Pereira
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import cv2

#Encabezado de archivo point-cloud
ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

#Función para almacenar la nube de puntos en un archivo ply
def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


if __name__ == '__main__':
    print('Cargando imagenes...')
    imgSrc = 'aloe'
    imgL = cv2.pyrDown( cv2.imread('img/'+imgSrc+'L'+'.jpg'))  # downscale images for faster processing
    imgR = cv2.pyrDown( cv2.imread('img/'+imgSrc+'R'+'.jpg') )

    # Rango de disparidad ajustado para el par de imagenes
    window_size = 3
    min_disp = 16
    num_disp = 112-min_disp
    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = 16,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32
    )

    print('Calculando disparidad...')
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    print('Generando nube de puntos 3D...',)
    h, w = imgL.shape[:2]
    f = 0.8*w                          # Suposicion de distancia focal
    Q = np.float32([[1, 0, 0, -0.5*w],
                    [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                    [0, 0, 0,     -f], # so that y-axis looks up
                    [0, 0, 1,      0]])
    points = cv2.reprojectImageTo3D(disp, Q)
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]
    out_fn = 'out.ply'
    write_ply('out.ply', out_points, out_colors)
    print('%s almacenado' % 'out.ply')

    cv2.imshow('Stereo Left', imgL)
    cv2.imshow('Disparidad', (disp-min_disp)/num_disp)
    cv2.waitKey()
    cv2.destroyAllWindows()
