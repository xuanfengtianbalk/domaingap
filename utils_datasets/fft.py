import numpy as np
import cv2
from matplotlib import pyplot as plt

# img = cv2.imread('test.jpg', 0)
#
# region = cv2.imread('test_region.jpg', 0)
#
# # paramenter
# lambda1 = 0.7
#
# # 傅里叶变换
# # test
# f = np.fft.fft2(img)
# fshift = np.fft.fftshift(f)
#
# phase_spectrumA = np.angle(fshift)  # 相位谱
# magnitude_spectrumB = 20 * np.log(np.abs(fshift))  # 幅度谱
#
# res1 = phase_spectrumA
# # region
# f1 = np.fft.fft2(region)
# fshift1 = np.fft.fftshift(f1)
#
# phase_spectrumA1 = np.angle(fshift1)  # 相位谱
# magnitude_spectrumB1 = 20 * np.log(np.abs(fshift1))  # 幅度谱
#
# # 插值，合成图像
# fshift = lambda1 * fshift + (1 - lambda1) * fshift1
# # 一张图像的相位和不同图像的幅值融合
# combined = np.multiply(np.abs(f), np.exp(1j * np.angle(f1)))
# imgcomblined = np.real(np.fft.fft2(combined))
# # 傅里叶逆变换
# ishift = np.fft.ifftshift(fshift)
# iimg = np.fft.ifft2(ishift)
#
# iimg = np.abs(iimg)
# res2 = np.abs(imgcomblined)
# # 显示图像
# plt.subplot(131), plt.imshow(img, 'gray'), plt.title('Original Image')
# plt.axis('off')
# plt.subplot(132), plt.imshow(res1, 'gray'), plt.title('Fourier Image')
# plt.axis('off')
# plt.subplot(133), plt.imshow(iimg, 'gray'), plt.title('Inverse Fourier Image')
# plt.axis('off')
# plt.show()
# print('done')
def img_fusion(img, region, lamba):
    background_index=img<12



    img[background_index] = lamba * img[background_index] + (1 - lamba) * region[background_index]
    iimg = np.abs(img).astype('uint8')

    return iimg

# def img_fusion(img, region, lamba):
#     # #读取图像
#     #傅里叶变换
#     f = np.fft.fft2(img)
#     fshift = np.fft.fftshift(f)
#     #region
#     f1 = np.fft.fft2(region)
#     fshift1 = np.fft.fftshift(f1)
#
#     #插值，合成图像
#     fshift = lamba * fshift + (1 - lamba) * fshift1
#     #傅里叶逆变换
#     ishift = np.fft.ifftshift(fshift)
#     iimg = np.fft.ifft2(ishift)
#
#     iimg = np.abs(iimg).astype('uint8')
#
#     return iimg

# img = cv2.imread('test.jpg', 0)
#
# region = cv2.imread('test_region.jpg', 0)
# iimg = img_fusion(img,region,0.3)
# plt.subplot(111), plt.imshow(iimg, 'gray'), plt.title('Inverse Fourier Image')
# plt.axis('off')
# plt.show()
# print('done')

# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
#
# #paramenter
# lambda1 = 1
#
# img = cv2.imread('test.jpg',0)
# dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
# dftShift = np.fft.fftshift(dft)
#
# img1 = cv2.imread('test_region.jpg',0)
# dft1 = cv2.dft(np.float32(img1), flags = cv2.DFT_COMPLEX_OUTPUT)
# dftShift1 = np.fft.fftshift(dft1)
#
# res_shift = 20*np.log(cv2.magnitude(dftShift[:,:,0],dftShift[:,:,1]))
# res_shift1 = 20*np.log(cv2.magnitude(dftShift1[:,:,0],dftShift1[:,:,1]))
# res1 = res_shift
# res2 = res_shift1
# # res2 = res_shift1[:,:,0]
#
#
# # rows,cols = img.shape
# # crow,ccol = (rows/2).__int__(),(cols/2).__int__()
# # mask = np.zeros((rows,cols,2),np.uint8)
# # mask[crow-30:crow+30,ccol-30:ccol+30] = 1
# # dftShift = dftShift*mask
#
# ishift = np.fft.ifftshift(dftShift)
# iImg = cv2.idft(ishift)
# iImg= cv2.magnitude(iImg[:, :,0], iImg[:, :,1]) # 计算幅度
# plt.subplot(221), plt.imshow(img, cmap = 'gray')
# plt.title('original'), plt.axis('off')
# plt.subplot(222), plt.imshow(iImg, cmap = 'gray')
# plt.title('inverse'), plt.axis('off')
# plt.subplot(223), plt.imshow(res1, cmap = 'gray')
# plt.title('res1'), plt.axis('off')
# plt.subplot(224), plt.imshow(res2, cmap = 'gray')
# plt.title('res2'), plt.axis('off')
# plt.show()

