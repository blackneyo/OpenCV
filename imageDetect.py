import cv2
import numpy as np
import matplotlib.pyplot as plt


'''
2022.04.08 이동한
edge 감지 / 이미지 그라디언트
subplot(nrows, ncols, index)
'''
# image = cv2.imread('nft1.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# hgt, wdt,_ = image.shape
#
# #Sobel Edges
# x_sobel = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
# y_sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
#
# plt.figure(figsize=(20, 20))
# # subplot 각 column에 독립된 subplot 그리기
# plt.subplot(3, 2, 1)
# plt.title("Original")
# plt.imshow(image)
# plt.title("Sobel X")
# plt.imshow(x_sobel)
#
# plt.subplot(3, 2, 3)
# plt.title("Sobel Y")
# plt.imshow(y_sobel)
# sobel_or = cv2.bitwise_or(x_sobel, y_sobel)
# plt.subplot(3, 2, 4)
# plt.imshow(sobel_or)
#
# laplacian = cv2.Laplacian(image, cv2.CV_64F)
# plt.subplot(3, 2, 5)
# plt.title("Laplacian")
# plt.imshow(laplacian)
# # There are two values: threshold1 and threshold2.
# # Those gradients that are greater than threshold2 => considered as an edge
# # Those gradients that are below threshold1 => considered not to be an edge.
# # Those gradients Values that are in between threshold1 and threshold2 => either classiﬁed as edges or non-edges
# # The first threshold gradient
# canny = cv2.Canny(image, 50, 120)
# plt.subplot(3, 2, 6)
# plt.imshow(canny.astype('uint8'))
# plt.show()

'''
팽창, 열기, 닫기, 침식
'''

# image = cv2.imread('fontstyle.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.figure(figsize=(20, 20))
# plt.subplot(3, 2, 1)
# plt.title("Original")
# plt.imshow(image)
#
# kernel = np.ones((5,5), np.uint8)
#
# erosion = cv2.erode(image, kernel, iterations = 1)
# plt.subplot(3, 2, 2)
# plt.title("Erosion")
# plt.imshow(erosion)
#
# dilation = cv2.dilate(image, kernel, iterations = 1)
# plt.subplot(3, 2, 3)
# plt.title("Dilation")
# plt.imshow(dilation)
#
# opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
# plt.subplot(3, 2, 4)
# plt.title("Opening")
# plt.imshow(opening)
#
# closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
# plt.subplot(3, 2, 5)
# plt.title("Closing")
# plt.imshow(closing)
# plt.show()

'''
시점 변환
'''

# image = cv2.imread('confirm.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.figure(figsize=(20, 20))
# plt.subplot(1, 2, 1)
# plt.title("Original")
# plt.imshow(image)
#
# points_A = np.float32([[320,15], [700,215], [85,610], [530,780]])
# points_B = np.float32([[0,0], [420, 0], [0,594], [420,594]])
# M = cv2.getPerspectiveTransform(points_A, points_B)
# warped = cv2.warpPerspective(image, M, (420,594))
# plt.subplot(1, 2, 2)
# plt.title("warpPerspective")
# plt.imshow(warped)
# plt.show()

# '''
# 이미지 피라미드
# '''
#
# image = cv2.imread('resume.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.figure(figsize=(20, 20))
# plt.subplot(2, 2, 1)
# plt.title("Original")
# plt.imshow(image)
#
# smaller = cv2.pyrDown(image)
# larger =  cv2.pyrUp(smaller)
# '''
# pyrDown = 단계적 축소, 화질 떨어짐
# '''
# plt.subplot(2, 2, 2)
# plt.title("smaller")
# plt.imshow(smaller)
# '''
# pyrUp = 단계적 확대, 화질 떨어짐
# '''
# plt.subplot(2, 2, 3)
# plt.title("Larger")
# plt.imshow(larger)
# plt.show()

# '''
# 자르기. 이미지 특정 부분을 가져오는 데 사용
# '''
# image = cv2.imread('confirm.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.figure(figsize=(10, 10))
# plt.subplot(2, 2, 1)
# plt.title("Original")
# plt.imshow(image)
# hgt, wdt = image.shape[:2]
# start_row, start_col = int(hgt * .25), int(wdt * .25)
# end_row, end_col = int(hgt * .78), int(wdt * .75)
# cropped = image[start_row:end_row, start_col:end_col]
# plt.subplot(2, 2, 2)
# plt.imshow(cropped)
# plt.show()

# '''
# 2022.04.11 이동한
# 크기 조정, 보간 및 크기 조정 / 보간(interpolation) = 두 점을 연결하는 궤적 생성, 보통 정보를 압축한 것을 다시 복원하기 위함
# 이미지
# '''
# image = cv2.imread('confirm.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.figure(figsize=(10, 10))
# plt.subplot(2, 2, 1)
# plt.title("Original")
# plt.imshow(image)
# image_scaled = cv2.resize(image, None, fx=0.75, fy=0.75)
#
# plt.subplot(2, 2, 2)
# plt.title("Scaling - Linear Interpolation")
# plt.imshow(image_scaled)
# img_scaled = cv2.resize(image, None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC) #3차회선 보간법/ 16개의 픽셀 사용
#
# plt.subplot(2, 2, 3)
# plt.title("Scaling - Cubic Interpolation ")
# plt.imshow(img_scaled)
# img_scaled = cv2.resize(image, (900, 400), interpolation = cv2.INTER_AREA) #축소할때 사용
#
# plt.subplot(2, 2, 4)
# plt.title("Scaling - Skewed Size")
# plt.imshow(img_scaled)
# plt.show()

# '''
# 임계값, 적응 임계값 및 이진화(이미지의 흑백처리)
# binary_image = 흑백으로만 표현한 이미지
# Threshholding = 여러값을 임계점 기준으로 두 가지 부류로 나누는 방법
# '''
# image = cv2.imread('confirm.jpg', 0)
# plt.figure(figsize=(20, 20))
# plt.subplot(3, 2, 1)
# plt.title("Original")
# plt.imshow(image)
#
# ret, thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY) # threshold(이미지, 임계값, 임계값 이상일 경우 바꿀 최대값 보통 흰색인 255로 지정, 흑백)
# plt.subplot(3, 2, 2)
# plt.title("Threshold Binary")
# plt.imshow(thresh1)
#
# image = cv2.GaussianBlur(image, (3, 3), 0) #이미지, 커널크기(3,3), 시그마x,y값 동일
# thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 5) #adaptiveThreshold(이미지, 벨류, adaptiveMethod, thresholdType, blocksize, C) C는 보정 상수
# plt.subplot(3, 2, 3)
# plt.title("Adaptive Mean Thresholding")
# plt.imshow(thresh)
#
# _, th2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Threshold함수와 Thresh_otsu를 같이 사용한 인진화 두번째 argument인 임계값은 0으로 설정해야 노이즈까지 같이 검출 안됨/ (물체만 검출되게 하기 위함)
# plt.subplot(3, 2, 4)
# plt.title("Otsu's Thresholding")
# plt.imshow(th2)
# plt.subplot(3, 2, 5)
# blur = cv2.GaussianBlur(image, (5, 5), 0)
# _, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# plt.title("Gaussian Otsu's Thresholding")
# plt.imshow(th3)
# plt.show()

# '''
# 선명하게 하기
# '''
# image = cv2.imread('confirm.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.figure(figsize=(10, 10))
# plt.subplot(1, 2, 1)
# plt.title("Original")
# plt.imshow(image)
# kernel_sharpening = np.array([[-1, -1, -1],
#                               [-1, 9, -1],
#                               [-1, -1, -1]])
# sharpened = cv2.filter2D(image, -1, kernel_sharpening)
# plt.subplot(1, 2, 2)
# plt.title("Image Sharpening")
# plt.imshow(sharpened)
# plt.show()

# '''
# 구조의 윤곽 식별
# 물체의 모양 식별하는데 도움이 됨
# canny edge를 매개변수로 전달해야 하는 findContours함수 사용
# '''
# image = cv2.imread('confirm.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.figure(figsize=(10, 10))
# plt.subplot(2, 2, 1)
# plt.title("Original")
# plt.imshow(image)
#
# # Grayscale
# gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # Canny edges Canny(gray타입, minval=30, maxval=200) 값이 클수록 엣지 검출 어렵고 작을 수록 엣지가 검출되기 쉽다.
# edged = cv2.Canny(gray, 30, 200)
# plt.subplot(2, 2, 2)
# plt.title("Canny Edges")
# plt.imshow(edged)
#
# # Finding Contours
# contour, hier = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #cv2.findContours (이진화 이미지, 검색 방법, 근사화 방법) return 윤곽선, 계층 구조
# plt.subplot(2, 2, 3)
# plt.title("Canny Edges After Contouring")
# plt.imshow(edged)
# print("Count of Contours = " + str(len(contour))) #해당 윤곽선의 계층 구조 표시
#
# # All contours // cv2.drawContours()를 이용하여 검출된 윤곽선 그림
# cv2.drawContours(image, contour, -1, (0, 255, 0), 3) #cv2.drawContours(이미지, [윤곽선=contour], contour의 인덱스 -1{-1은 윤곽선 배열 모두를 의미}, (BGR), 두께)
# plt.subplot(2, 2, 4)
# plt.title("Contours")
# plt.imshow(image)
# plt.show()

# '''
# Hough Line을 사용한 라인 감지
# 임계값은 라인으로 간주되기 위한 최소치 
# '''
# image = cv2.imread('confirm.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.figure(figsize=(10, 10))
# 
# # Grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 
# # Canny Edges
# edges = cv2.Canny(gray, 100, 170, apertureSize = 3) #cv2.Canny(gray타입, minval=100, maxval=170, aperturesize 이미지 그레디언트 크기를 구할때 사용하는 소벨 커널의 크기
# plt.subplot(2, 2, 1)
# plt.title("edges")
# plt.imshow(edges)
# 
# # Run HoughLines Fuction
# lines = cv2.HoughLines(edges, 1, np.pi/180, 200) #cv2.HoughLines(edge를 사용, 매개변수 분해능 1을 사용, 180도를 180으로 나는 1도, threshold(만나는 점의 기준) = 200)
# 
# # Run for loop through each line
# for line in lines:
#     rho, theta = line[0]
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a * rho
#     y0 = b * rho
#     x_1 = int(x0 + 1000 * (-b))
#     y_1 = int(y0 + 1000 * (a))
#     x_2 = int(x0 - 1000 * (-b))
#     y_2 = int(y0 - 1000 * (a))
#     cv2.line(image, (x_1, y_1), (x_2, y_2), (255, 0, 0), 2) # cv2.line(이미지, 시작점좌표(x,y), 종료점좌표(x,y), 선색, 선 두깨)
# # Show Final output
# plt.subplot(2, 2, 2)
# plt.imshow(image)
# plt.show()

# '''
# 코너 찾기
# 이미지 모서리를 찾기 위해 cornerHarris 함수 사용
# '''
# image = cv2.imread('confirm.jpg')
#
# # Grayscaling
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.figure(figsize=(10, 10))
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # CornerHarris function want input to be float
# gray = np.float32(gray)
# h_corners = cv2.cornerHarris(gray, 3, 3, 0.05) #cv2.cornerHarris(gray스케일, 블록사이즈 3, 소벨 도함수 매개변수 = 3, Harris 검출기 자유 매개변수 = 0.05)
# kernel = np.ones((7, 7), np.uint8) # 1로 가득찬 7x7 행렬 datatype = uint8 타입
# h_corners = cv2.dilate(h_corners, kernel, iterations = 10) #cv2.dilate 이미지 팽창 (h_corners, ㅇ,
# image[h_corners > 0.024 * h_corners.max() ] = [256, 128, 128]
# plt.subplot(1, 1, 1)
#
# # Final Output
# plt.imshow(image)
# plt.show()

'''
2022.04.12 이동한
원, 타원 계산
SimpleBlobDetector
필터 종류
'''
image = cv2.imread('BlobTest.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 10))
detector = cv2.SimpleBlobDetector_create()

# Detect Blobs
points = detector.detect(image)
blank = np.zeros((1, 1))
blobs = cv2.drawKeypoints(image, points, blank, (0, 0, 255),
                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# Detect blobs
keypoints = detector.detect(image)

number_of_blobs = len(keypoints)
text = "Total Blobs: " + str(len(keypoints))
cv2.putText(blobs, text, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 255), 2)

plt.subplot(2, 2, 1)
plt.imshow(blobs)
# Filtering Parameters
# Initialize parameter setting using cv2.SimpleBlobDetector
params = cv2.SimpleBlobDetector_Params()

# Area filtering parameters 지역별 필터링
params.filterByArea =True
params.minCircularity = 0.9

# Convexity filtering parameters 볼록성 필터링
params.filterByConvexity = False
params.minInertiaRatio = 0.01

# detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)


# Draw blobs on our image as red circles
blank = np.zeros((1, 1))
blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 255, 0),
                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
number_of_blobs = len(keypoints)
text = "No. Circular Blobs: " + str(len(keypoints))
cv2.putText(blobs, text, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)

# Show blobs
plt.subplot(2, 2, 2)
plt.title("Filtering Circular BLobs Only")
plt.imshow(blobs)
plt.show()