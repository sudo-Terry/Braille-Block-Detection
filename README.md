# 동영상에서 원형 및 선형 점자블록 검출 시스템

## 1. 프로젝트 개요

본 프로젝트의 목표는 동영상 파일에서 보도블럭 상의 원형 및 선형 점자블록을 검출하는 시스템을 개발하는 것입니다. 이 시스템은 다양한 영상 처리 기법을 활용하여 점자블록들의 종류나 방향을 탐지하고 그 결과를 시각화하여 출력합니다. 소리 출력 시스템 등 다른 하드웨어와 연결하여 시각 장애인이 보다 편리하게 점자 보도블럭을 따라 통행할 수 있는 것을 목표로 합니다.

## 2. 시스템 개요

시스템은 다음의 주요 단계로 구성됩니다:

1. 영상 각 프레임에 대한 전처리
2. 색상 기반 영역 필터링
3. 영상 내 엣지 검출
4. 2, 3 처리 결과 혼합 및 해당 구간 내에서 원형 및 선형 패턴 검출
5. 결과 시각화 및 저장

## 3. 주요 기능 설명

### 3.1 영상 내 보도블럭 구간 추출 (색상 기반)

```python
def rgb_to_yiq(img_rgb):
    img_yiq = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2YCrCb)
    return img_yiq

def increase_saturation(img, scale=1.5):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)
    s = cv2.multiply(s, scale)
    s = np.clip(s, 0, 255).astype(hsv_img.dtype)
    hsv_img = cv2.merge([h, s, v])
    img_saturated = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    return img_saturated

def filter_color(img_yiq):
    y_channel, cr_channel, cb_channel = cv2.split(img_yiq)
    
    # 노란색 영역의 Cr, Cb 임계값 설정
    # Cr 빨강 청록 140, 170
    # Cb 파랑 노랑 100, 130
    # 노란색 210 (Y) 146 (Cr) 16 (Cb)
    lower_cr = 140
    upper_cr = 170
    lower_cb = 0
    upper_cb = 140

    
    # 밝기 채널의 임계값 설정 (어두운 영역 제외)
    lower_y = 30  # 최소 밝기값 (조정 가능)
    upper_y = 170 # 최대 밝기값
    
    # Cr, Cb 채널에서 임계값에 해당하는 영역 추출
    mask_cr = cv2.inRange(cr_channel, lower_cr, upper_cr)
    mask_cb = cv2.inRange(cb_channel, lower_cb, upper_cb)
    
    # Y 채널에서 임계값에 해당하는 영역 추출
    mask_y = cv2.inRange(y_channel, lower_y, upper_y)
    
    # Cr, Cb 임계값에 해당하는 마스크 생성
    color_mask_cr_cb = cv2.bitwise_and(mask_cr, mask_cb)
    
    # Y 임계값과 Cr, Cb 임계값을 결합
    color_mask = cv2.bitwise_and(color_mask_cr_cb, mask_y)
    
    return color_mask
```

- 채도 강화를 통해 노란색 블럭들을 효과적으로 찾아내고자 하였습니다.
- YIQ 색상 공간에서 특정 범위의 색상을 필터링하여 블록 마스크를 생성했습니다.

### 결과

![Untitled](https://github.com/sudo-Terry/Braille-Block-Detection/assets/76080411/72e55142-96bf-425d-ac43-a7d823b893da)


### 3.2 전처리 (가우시안 필터링 및 히스토그램 평활화)

```python
def preprocess_image(frame):
    # 가우시안 블러링 적용
    blurred = apply_gaussian_filter(frame, 5, 1)
    
    # 히스토그램 평활화 적용 (HSI 색상 공간에서 적용)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_eq = histo_equal(v)
    hsv_eq = cv2.merge([h, s, v_eq])
    equalized = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)
    
    return equalized
```

- 가우시안 필터링을 적용하여 이미지를 부드럽게 하고 자연 환경의 노이즈를 제거했습니다.
- 히스토그램 평활화를 적용하여 명암 대비를 향상시켰습니다.

### 3.3 엣지 검출

```python
color_mask = filter_color(img_yiq)  

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blurred = gaussian_filter_grayscale(gray, 5, 1)
equalized = histo_equal(blurred)
edges = cv2.Canny(equalized, 100, 200)

mask = cv2.bitwise_and(color_mask, edges)
```

- 그레이스케일 변환 후 가우시안 블러를 적용하고, Canny 엣지 검출을 수행했습니다.
- 색상 마스크와 엣지 검출 결과를 결합하여 최종 마스크를 생성했습니다.

### 결과

![Untitled (1)](https://github.com/sudo-Terry/Braille-Block-Detection/assets/76080411/d82dc6a5-ca41-4bd4-8156-4a9b397e7e49)


### 3.4 원형 및 선형 패턴 검출

```python
def detect_circular_patterns(edges, frame):
  if edges is not None and len(edges.shape) == 2 and edges.dtype == np.uint8:
      # 원형 패턴 검출
      circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                                 param1=50, param2=30, minRadius=10, maxRadius=30)
      
      if circles is not None:
          circles = np.round(circles[0, :]).astype("int")
          for (x, y, r) in circles:
              cv2.circle(frame, (x, y), r, (0, 0, 255), 4)  # 원의 중심에 빨간색 원 그리기
  return frame
    
def detect_linear_patterns(edges, frame):
	if edges is not None and len(edges.shape) == 2 and edges.dtype == np.uint8:
	    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=30, maxLineGap=20)
	    if lines is not None:
	        for line in lines:
	            x1, y1, x2, y2 = line[0]
	            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 4) 
return frame
```

- Hough Transform을 사용하여 원형 및 선형 패턴을 검출했습니다.
- 검출된 원형 및 선형 패턴을 원본 프레임에 시각화했습니다.

### 결과

<img width="604" alt="스크린샷 2024-06-21 오후 8 37 07" src="https://github.com/sudo-Terry/Braille-Block-Detection/assets/76080411/1e1f0068-d6fe-4288-adfc-5b9710ab2d66">
<img width="488" alt="스크린샷 2024-06-21 오후 8 38 07" src="https://github.com/sudo-Terry/Braille-Block-Detection/assets/76080411/3f3e4d6c-206b-494e-97b0-f0e66065a58f">


# 4. 시행착오

### 4.1 형태학적 특징 문제

프로젝트를 진행하면서 시각장애인이 활용하기 위해 어깨 높이에서 도로를 촬영하는 카메라를 가정하였으나, 개인별 시선의 높이에 편차가 있고 타일의 방향에 따라 타일의 형태가 달라지는 등의 문제가 있었습니다. 사다리꼴 형태의 타일을 민감도를 높여 형태를 추출하려고 하면 엉뚱한 오브젝트까지 인지하게 되고, 민감도를 낮추면 타일조차 인지하지 못하는 문제가 발생했습니다.

![Untitled (2)](https://github.com/sudo-Terry/Braille-Block-Detection/assets/76080411/1a36b3dc-d656-41f0-a106-59000fc2c953)


### 4.2 색상 특징 문제

실제 영상 내에서는 타일이 노란색이라는 정보를 활용하기가 어려웠습니다. 날씨에 따른 자연광의 변화, 시간에 따른 타일 색의 변색 등 다양한 환경 변수로 인해 이미지 처리로는 노란색 타일 부분만을 정확히 추출해내기가 어려웠습니다.

![Untitled (3)](https://github.com/sudo-Terry/Braille-Block-Detection/assets/76080411/e62003b5-6164-4f30-a595-60e2ce717efe)
![Untitled (4)](https://github.com/sudo-Terry/Braille-Block-Detection/assets/76080411/f98620d8-3b00-4226-8463-7f61863273fd)
![Untitled (5)](https://github.com/sudo-Terry/Braille-Block-Detection/assets/76080411/68920cc2-2ccc-4691-926e-3569732d8d2a)
![Untitled (6)](https://github.com/sudo-Terry/Braille-Block-Detection/assets/76080411/35ea375f-9c36-4a16-af8f-56a0110c50b9)
