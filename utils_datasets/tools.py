# import math
#
# import torch
#
#
#
# #均匀生成极坐标
# def generate_uniform_distribution_polar_coordinate(num_theta, num_phi, range_r):
# # num_theta   : 1
# # num_phi     : 1
# # num_range_r : n * 1
#     polar_list=[]
#     for r in range_r:
#         for i in range(num_theta):
#             for j in range(num_phi):
#                 polar = [math.pi * 2 * i /num_theta, math.pi * 2 * j /num_phi, r]
#                 polar_list.append(polar)
#     return polar_list
#
# #极坐标转化直角坐标系
# def polar2rectangular(theta, phi, r):
#     x = r * math.sin(theta) * math.cos(phi)
#     y = r * math.sin(theta) * math.sin(phi)
#     z = r * math.cos(theta)
#     return [x,y,z]
#
# def generate_uniform_distribution_rectangular_coordinate(num_theta, num_phi, range_r):
#     polar_list = generate_uniform_distribution_polar_coordinate(num_theta, num_phi, range_r)
#     rectangular_list = []
#     for (theta, phi, r) in polar_list:
#         rectangular = polar2rectangular(theta, phi, r)
#         rectangular_list.append(rectangular)
#     return rectangular_list
#
# if __name__ == "__main__":
#     num_theta = 32
#     num_phi = 32
#     range_r = [1]
#
#     polar_list = generate_uniform_distribution_polar_coordinate(num_theta, num_phi, range_r)
#     x = []
#     y = []
#     z = []
#     for (theta, phi, r) in polar_list:
#         rectangular = polar2rectangular(theta, phi, r)
#         x.append(rectangular[0])
#         y.append(rectangular[1])
#         z.append(rectangular[2])
#
#         print(rectangular)
#
#     #生成三维点云
#     import matplotlib.pyplot as plt
#     import numpy as np
#     from mpl_toolkits.mplot3d import Axes3D
#
#     fig = plt.figure(figsize=(20, 20))
#     ax = Axes3D(fig)
#     ax.scatter3D(x, y, z, s=100)
#     plt.show()
#
#     print("finish!")

#!/usr/bin/python
# -*- coding: utf-8 -*-
import math
class Spherical(object):
    '''球坐标系'''
    def __init__(self, radial = 1.0, polar = 0.0, azimuthal = 0.0):
        self.radial = radial
        self.polar = polar
        self.azimuthal = azimuthal
    def toCartesian(self):
        '''转直角坐标系'''
        r = math.sin(self.azimuthal) * self.radial
        x = math.cos(self.polar) * r
        y = math.sin(self.polar) * r
        z = math.cos(self.azimuthal) * self.radial
        return x, y, z
def splot(r, limit):
    s = Spherical(radial=r)
    n = int(math.ceil(math.sqrt((limit - 2) / 4)))
    azimuthal = 0.5 * math.pi / n
    for a in range(-n, n + 1):
        s.polar = 0
        size = (n - abs(a)) * 4 or 1
        polar = 2 * math.pi / size
        for i in range(size):
            yield s.toCartesian()
            s.polar += polar
        s.azimuthal += azimuthal

import numpy as np
points = np.array([[0.38,
                0.385,
                0.325,
                1]
                  ,
               [-0.37,
                0.385,
                0.325,
                1]
                  ,
               [0.38,
                -0.385,
                0.325,
                1]
                  ,
               [-0.37,
                -0.385,
                0.325,
                1]
                  ,
               [0.37,
                0.30,
                0,
                1]
                  ,
               [-0.37,
                0.30,
                0,
                1]
                  ,
               [0.355,
                -0.265,
                0,
                1]
                  ,
               [-0.37,
                -0.30,
                0,
                1]
                  ,
               [0.305,
                -0.57,
                0.255,
                1]
                  ,
               [0.55,
                0.49,
                0.26,
                1]
                  ,
               [-0.53,
                0.48,
                0.26,
                1]]
                  )
#生成三维点云
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
if __name__ == "__main__":
    x=[]
    y=[]
    z=[]
    for point in splot(r=6, limit=64):
        print("%f %f %f" % point)
        x.append(point[0])
        y.append(point[1])
        z.append(point[2])


    fig = plt.figure(figsize=(9, 6))
    ax = Axes3D(fig)
    ax.scatter3D(x, y, z, s=50,marker='d',color='b')

    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = 6 * np.outer(np.cos(u), np.sin(v))
    y = 6 * np.outer(np.sin(u), np.sin(v))
    z = 6 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, rstride=1, cstride=1, color='b', shade=0,alpha=0.05)

    idx=[0,1,3,2,0,4,5,7,6,4]
    ax.plot3D(points[idx,0],points[idx,1],points[idx,2],'red',linewidth=2)
    idx=[1,5]
    ax.plot3D(points[idx,0],points[idx,1],points[idx,2],'red')
    idx=[2,6]
    ax.plot3D(points[idx,0],points[idx,1],points[idx,2],'red')
    idx=[3,7]
    ax.plot3D(points[idx,0],points[idx,1],points[idx,2],'red')
    idx=[0,9]
    ax.plot3D(points[idx,0],points[idx,1],points[idx,2],'red')
    idx=[2,8]
    ax.plot3D(points[idx,0],points[idx,1],points[idx,2],'red')
    idx=[1,10]
    ax.plot3D(points[idx,0],points[idx,1],points[idx,2],'red')

    points=points*6
    idx=[0,1,3,2,0,4,5,7,6,4]
    ax.plot3D(points[idx,0],points[idx,1],points[idx,2],'red',linestyle='--',linewidth=2)
    idx=[1,5]
    ax.plot3D(points[idx,0],points[idx,1],points[idx,2],'red',linestyle='--')
    idx=[2,6]
    ax.plot3D(points[idx,0],points[idx,1],points[idx,2],'red',linestyle='--')
    idx=[3,7]
    ax.plot3D(points[idx,0],points[idx,1],points[idx,2],'red',linestyle='--')
    idx=[0,9]
    ax.plot3D(points[idx,0],points[idx,1],points[idx,2],'red',linestyle='--')
    idx=[2,8]
    ax.plot3D(points[idx,0],points[idx,1],points[idx,2],'red',linestyle='--')
    idx=[1,10]
    ax.plot3D(points[idx,0],points[idx,1],points[idx,2],'red',linestyle='--')

    ax.set_xlabel('x-axis',size=14,labelpad=14)
    ax.set_ylabel('y-axis',size=14,labelpad=14)
    ax.set_zlabel('z-axis',size=14,labelpad=14)

    ax.tick_params(labelsize=14)
    # ax.set_xticks(fontsize=14)
    # ax.set_yticks(fontsize=14)
    # ax.set_zticks(fontsize=14)
    plt.show()



    print("finish!")