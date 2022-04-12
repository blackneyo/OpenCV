import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
2022.04.11 이동한
크기 조정, 보간 및 크기 조정 / 보간(interpolation) = 두 점을 연결하는 궤적 생성, 보통 정보를 압축한 것을 다시 복원하기 위함
이미지
'''
image = cv2.imread('confirm.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.title("Original")
plt.imshow(image)
image_scaled = cv2.resize(image, None, fx=0.75, fy=0.75)

plt.subplot(2, 2, 2)
plt.title("Scaling - Linear Interpolation")
plt.imshow(image_scaled)
img_scaled = cv2.resize(image, None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC) #3차회선 보간법/ 16개의 픽셀 사용

plt.subplot(2, 2, 3)
plt.title("Scaling - Cubic Interpolation ")
plt.imshow(img_scaled)
img_scaled = cv2.resize(image, (900, 400), interpolation = cv2.INTER_AREA) #축소할때 사용

plt.subplot(2, 2, 4)
plt.title("Scaling - Skewed Size")
plt.imshow(img_scaled)
plt.show()

'''
임계값, 적응 임계값 및 이진화(이미지의 흑백처리)
binary_image = 흑백으로만 표현한 이미지
Threshholding = 여러값을 임계점 기준으로 두 가지 부류로 나누는 방법
'''
image = cv2.imread('confirm.jpg', 0)
plt.figure(figsize=(20, 20))
plt.subplot(3, 2, 1)
plt.title("Original")
plt.imshow(image)

ret, thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY) # threshold(이미지, 임계값, 임계값 이상일 경우 바꿀 최대값 보통 흰색인 255로 지정, 흑백)
plt.subplot(3, 2, 2)
plt.title("Threshold Binary")
plt.imshow(thresh1)

image = cv2.GaussianBlur(image, (3, 3), 0) #이미지, 커널크기(3,3), 시그마x,y값 동일
thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 5) #adaptiveThreshold(이미지, 벨류, adaptiveMethod, thresholdType, blocksize, C) C는 보정 상수
plt.subplot(3, 2, 3)
plt.title("Adaptive Mean Thresholding")
plt.imshow(thresh)

_, th2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Threshold함수와 Thresh_otsu를 같이 사용한 인진화 두번째 argument인 임계값은 0으로 설정해야 노이즈까지 같이 검출 안됨/ (물체만 검출되게 하기 위함)
plt.subplot(3, 2, 4)
plt.title("Otsu's Thresholding")
plt.imshow(th2)
plt.subplot(3, 2, 5)
blur = cv2.GaussianBlur(image, (5, 5), 0)
_, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.title("Gaussian Otsu's Thresholding")
plt.imshow(th3)
plt.show()

'''
선명하게 하기
'''
image = cv2.imread('confirm.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(image)
kernel_sharpening = np.array([[-1, -1, -1],
                              [-1, 9, -1],
                              [-1, -1, -1]])
sharpened = cv2.filter2D(image, -1, kernel_sharpening)
plt.subplot(1, 2, 2)
plt.title("Image Sharpening")
plt.imshow(sharpened)
plt.show()

'''
구조의 윤곽 식별
물체의 모양 식별하는데 도움이 됨
canny edge를 매개변수로 전달해야 하는 findContours함수 사용
'''
image = cv2.imread('confirm.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.title("Original")
plt.imshow(image)

# Grayscale
gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Canny edges Canny(gray타입, minval=30, maxval=200) 값이 클수록 엣지 검출 어렵고 작을 수록 엣지가 검출되기 쉽다.
edged = cv2.Canny(gray, 30, 200)
plt.subplot(2, 2, 2)
plt.title("Canny Edges")
plt.imshow(edged)

# Finding Contours
contour, hier = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #cv2.findContours (이진화 이미지, 검색 방법, 근사화 방법) return 윤곽선, 계층 구조
plt.subplot(2, 2, 3)
plt.title("Canny Edges After Contouring")
plt.imshow(edged)
print("Count of Contours = " + str(len(contour))) #해당 윤곽선의 계층 구조 표시

# All contours // cv2.drawContours()를 이용하여 검출된 윤곽선 그림
cv2.drawContours(image, contour, -1, (0, 255, 0), 3) #cv2.drawContours(이미지, [윤곽선=contour], contour의 인덱스 -1{-1은 윤곽선 배열 모두를 의미}, (BGR), 두께)
plt.subplot(2, 2, 4)
plt.title("Contours")
plt.imshow(image)
plt.show()
