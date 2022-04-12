import cv2
import numpy as np
import matplotlib.pyplot as plt


'''
2022.04.08 이동한
edge 감지 / 이미지 그라디언트
subplot(nrows, ncols, index)
'''
image = cv2.imread('nft1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
hgt, wdt,_ = image.shape

#Sobel Edges
x_sobel = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
y_sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)

plt.figure(figsize=(10, 10))
# subplot 각 column에 독립된 subplot 그리기
plt.subplot(3, 2, 1)
plt.title("Original")
plt.imshow(image)
plt.title("Sobel X")
plt.imshow(x_sobel)

plt.subplot(3, 2, 3)
plt.title("Sobel Y")
plt.imshow(y_sobel)
sobel_or = cv2.bitwise_or(x_sobel, y_sobel)
plt.subplot(3, 2, 4)
plt.imshow(sobel_or)

laplacian = cv2.Laplacian(image, cv2.CV_64F)
plt.subplot(3, 2, 5)
plt.title("Laplacian")
plt.imshow(laplacian)
# There are two values: threshold1 and threshold2.
# Those gradients that are greater than threshold2 => considered as an edge
# Those gradients that are below threshold1 => considered not to be an edge.
# Those gradients Values that are in between threshold1 and threshold2 => either classiﬁed as edges or non-edges
# The first threshold gradient
canny = cv2.Canny(image, 50, 120)
plt.subplot(3, 2, 6)
plt.imshow(canny.astype('uint8'))
plt.show()

'''
팽창, 열기, 닫기, 침식
'''

image = cv2.imread('fontstyle.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 10))
plt.subplot(3, 2, 1)
plt.title("Original")
plt.imshow(image)

kernel = np.ones((5,5), np.uint8)

erosion = cv2.erode(image, kernel, iterations = 1)
plt.subplot(3, 2, 2)
plt.title("Erosion")
plt.imshow(erosion)

dilation = cv2.dilate(image, kernel, iterations = 1)
plt.subplot(3, 2, 3)
plt.title("Dilation")
plt.imshow(dilation)

opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
plt.subplot(3, 2, 4)
plt.title("Opening")
plt.imshow(opening)

closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
plt.subplot(3, 2, 5)
plt.title("Closing")
plt.imshow(closing)
plt.show()

'''
시점 변환
'''

image = cv2.imread('confirm.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(image)

points_A = np.float32([[320,15], [700,215], [85,610], [530,780]])
points_B = np.float32([[0,0], [420, 0], [0,594], [420,594]])
M = cv2.getPerspectiveTransform(points_A, points_B)
warped = cv2.warpPerspective(image, M, (420,594))
plt.subplot(1, 2, 2)
plt.title("warpPerspective")
plt.imshow(warped)
plt.show()

'''
이미지 피라미드
'''

image = cv2.imread('resume.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.title("Original")
plt.imshow(image)

smaller = cv2.pyrDown(image)
larger =  cv2.pyrUp(smaller)
'''
pyrDown = 단계적 축소, 화질 떨어짐
'''
plt.subplot(2, 2, 2)
plt.title("smaller")
plt.imshow(smaller)
'''
pyrUp = 단계적 확대, 화질 떨어짐
'''
plt.subplot(2, 2, 3)
plt.title("Larger")
plt.imshow(larger)
plt.show()

'''
자르기. 이미지 특정 부분을 가져오는 데 사용
'''
image = cv2.imread('confirm.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.title("Original")
plt.imshow(image)
hgt, wdt = image.shape[:2]
start_row, start_col = int(hgt * .25), int(wdt * .25)
end_row, end_col = int(hgt * .78), int(wdt * .75)
cropped = image[start_row:end_row, start_col:end_col]
plt.subplot(2, 2, 2)
plt.imshow(cropped)
plt.show()


