## 1️⃣ SIFT를 이용한 특징점 검출 및 시각화
### 🌀 과제 설명
- 소벨(Sobel) 필터를 사용해 엣지를 검출
<br>
  
### 📌 개념
- 소벨 필터를 이용해 X, Y 방향의 기울기를 계산
- 기울기를 조합하여 에지 강도(edge magnitude) 계산
- 검출된 엣지를 시각화
<br>

### 💻 주요 코드
<p>✔ <b>이미지 불러오기 </b><code>cv.imread(image_path)</code><br></p>
<p>✔ <b>그레이스케일 변환</b> <code>cv.cvtColor(image, cv.COLOR_BGR2GRAY)</code><br>
<p>✔ <b>소벨 필터 적용</b> <code>cv.Sobel(src, ddepth, dx, dy, ksize)</code><br>
<p>✔ <b>에지 강도 계산</b> <code>edge_magnitude = cv.magnitude(sobel_x, sobel_y)</code><br>
<p>✔ <b>이미지 시각화</b> <code>cv.imshow()</code><br>
<br>

<br>



<details>
  <summary><b> 🧿 클릭해서 코드 보기 </b></summary>
  
  ```python
import cv2 as cv
import matplotlib.pyplot as plt

# 이미지 로드
image_path = 'C:/Users/82107/Desktop/cv/mot_color70.jpg'
image = cv.imread(image_path)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# SIFT 객체 생성 (특징점 개수 조절 가능)
sift = cv.SIFT_create(nfeatures=500)

# 특징점 검출 및 기술자 계산
keypoints, descriptors = sift.detectAndCompute(gray, None)

# 특징점 시각화
image_with_keypoints = cv.drawKeypoints(image, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 이미지 출력
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(image_with_keypoints, cv.COLOR_BGR2RGB))
plt.title('SIFT Keypoints')
plt.axis('off')

plt.show()

 ```
</details>

<br>

### 🕵‍♀ 결과화면
![결과이미지](./data/6_1.png)

<br>
<br>

## 2️⃣ SIFT를 이용한 두 영상 간 특징점 매칭
### 🌀 과제 설명
- 이진화된 이미지에 대해 <b>팽창(Dilation), 침식(Erosion), 열림(Opening), 닫힘(Closing) 연산</b>을 수행하여<br> 노이즈 제거 및 형태 보정
<br>

### 📌 개념
- 캐니 에지 검출을 이용해 엣지를 추출
- 허프 변환(Hough Transform)을 이용해 직선을 검출
- 검출된 직선을 원본 이미지에 표시
<br>

### 💻 주요 코드
<p>✔ <b>캐니 에지 검출</b> <code>cv.Canny(image, threshold1, threshold2)</code><br>
<p>✔ <b>허프 변환을 사용한 직선 검출</b> <code>cv.HoughLinesP(image, rho, theta, threshold, minLineLength, maxLineGap)</code><br>
<p>✔ <b>검출된 직선을 원본 이미지에 빨간색으로 표시</b> <code>cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)</code><br>
<br>

<details>
  <summary><b> 🧿 클릭해서 코드 보기 </b></summary>

  ```python
import cv2 as cv
import matplotlib.pyplot as plt

# 이미지 로드
image1_path = 'C:/Users/82107/Desktop/cv/mot_color70.jpg'
image2_path = 'C:/Users/82107/Desktop/cv/mot_color83.jpg'
image1 = cv.imread(image1_path)
image2 = cv.imread(image2_path)
gray1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

# SIFT 객체 생성
sift = cv.SIFT_create()

# 특징점 검출 및 기술자 계산
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# BFMatcher 생성 및 매칭 수행
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# 매칭 결과 정렬 (거리순)
matches = sorted(matches, key=lambda x: x.distance)

# 매칭 결과 시각화
image_matches = cv.drawMatches(image1, keypoints1, image2, keypoints2, matches[:50], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 결과 출력
plt.figure(figsize=(12, 6))
plt.imshow(cv.cvtColor(image_matches, cv.COLOR_BGR2RGB))
plt.title('SIFT Feature Matching')
plt.axis('off')
plt.show()

 ```
</details>

<br>

### 🕵‍♀ 결과화면
![결과이미지](./data/6_2.png)

<br>
<br>

## 3️⃣ 호모그래피를 이용한 이미지 정합(Image Alignment)
### 🌀 과제 설명
- GrabCut 알고리즘을 사용해 이미지에서 객체(전경)와 배경을 분리
<br>

### 📌 개념
- 초기 사각형(rect)을 설정하여 관심 영역 지정
- GrabCut 알고리즘을 사용해 배경과 전경 분리
- 마스크(mask) 처리를 통해 전경만 남김
<br>

### 💻 주요 코드
<p> ✔ <b> 초기 마스크 생성</b> <code>np.zeros(image.shape[:2], np.uint8)</code><br>
<p> ✔ <b> 배경 모델과 전경 모델 초기화</b> <code>bgdModel = np.zeros((1, 65), np.float64)</code><br>
<p> - cv.grabCut() 함수에서 사용하는 전경(foreground)과 배경(background) 모델을 저장할 배열 <br>
<p> - 65: OpenCV에서 정해진 GMM(Gaussian Mixture Model) 파라미터 개수<br>
<p> ✔ <b> 마스크 처리하여 배경 제거 </b> <code>mask2 = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype('uint8')
</code>
<p> - cv.GC_BGD(0): 확실한 배경
<p> - cv.GC_PR_BGD(2): 가능성이 높은 배경
<p> - cv.GC_FGD(1): 확실한 전경
<p> - cv.GC_PR_FGD(3): 가능성이 높은 전경
<p> - 배경 픽셀을 제거하고 전경만 남김
<br>
<br>


<details>
  <summary><b> 🧿 클릭해서 코드 보기 </b></summary>

  ```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 이미지 로드
image1_path = 'C:/Users/82107/Desktop/cv/img1.jpg'
image2_path = 'C:/Users/82107/Desktop/cv/img2.jpg'
image1 = cv.imread(image1_path)
image2 = cv.imread(image2_path)

# 이미지 로드 확인
if image1 is None or image2 is None:
    print("Error: One or both images could not be loaded. Check the file paths.")
    exit()

# 그레이스케일 변환
gray1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

# SIFT 객체 생성
sift = cv.SIFT_create()

# 특징점 검출 및 기술자 계산
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# BFMatcher 생성 및 매칭 수행
bf = cv.BFMatcher(cv.NORM_L2)
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# 좋은 매칭점 선택 (비율 테스트 적용)
good_matches = []
ratio_thresh = 0.75
for m, n in matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)

# 매칭 개수 확인
print(f"Number of good matches: {len(good_matches)}")

# 최소한의 매칭점 필요
if len(good_matches) > 10:
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # 호모그래피 계산
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

    # 호모그래피 계산 실패 시 처리
    if H is None:
        print("Error: Homography calculation failed.")
        exit()
    
    # 이미지 정합
    h, w = image1.shape[:2]
    aligned_image = cv.warpPerspective(image1, H, (w, h))
    
    # 결과 출력
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(cv.cvtColor(image1, cv.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(cv.cvtColor(image2, cv.COLOR_BGR2RGB))
    plt.title('Target Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(cv.cvtColor(aligned_image, cv.COLOR_BGR2RGB))
    plt.title('Aligned Image')
    plt.axis('off')
    
    plt.show(block=True)  # 창이 바로 닫히지 않도록 설정
else:
    print("Not enough matches found to compute homography.")

 ```
</details>

<br>

### 🕵‍♀ 결과화면
![결과이미지](./data/6_3.png)
