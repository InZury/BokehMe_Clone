# BokehMe_Clone

---

BokehMe Clone Repository [[Bokeh Origin]]

Neural Rendering과 Classical Rendering의 성능을 비교한 코드입니다.  

Bokeh 효과(보케 효과)는 카메라로 보여지는 영상의 초점을 조절하여,  
강조하는 부분을 제외하고,부드럽고 흐릿하게 만들어주는 효과를 말한다.

해당 코드의 원본은 "BokehMe: When Neural Rendering Meets Classical Rendering (CVPR 2022 Oral)"이며,  
IEEE CVPR에 제출된 논문의 학습 모델 (checkpoint)를 남긴 코드이다.

---

본 코드는 원본 코드를 재구성하여, 현재 기준(2024.06.10)에 맞는  
파이썬 패키지 버전으로 코드 리팩토링을 진행한 코드이다.  

수정된 requirements.txt의 패키지 버전은 다음과 같다.
```angular2html
| Package         | Old Version | New Version |
|-----------------|-------------|-------------|
| cupy            | 10.5.0      | 13.1.0      |
| cupy_cuda90     | 7.7.0       |     ---     | Package is Removed due to cuda version
| matplotlib      | 3.5.1       | 3.8.4       |
| numpy           | 1.18.5      | 1.26.4      |
| opencv_python   | 4.2.0.34    | 4.7.0       |
| pillow          | 9.1.1       | 10.3.0      |
| torch           | 1.8.1       | 2.3.0       |
| torchvision     | 0.9.1       | 0.18.0      |
```


### Origin code 달라진 부분
- code convention 준수 (PEP 8 Style)
- 단어 맞춤법 수정 (고유 단어 제외)
- 코드 경고 및 오류 수정 (Warning code)
- deprecated 코드 수정 [[code 1]] &nbsp; [[code 2]]
- demo.py 추가 코드 작성 -->> **추가 작업 부분**

### 코드 실행
```angular2html
python demo.py --image_path 'inputs/[image file]' --disp_path 'inputs/[dist file]' --save_dir 'outputs' --K [blur value] --disp_focus [focus value] --gamma [gamma value] --highlight
```

위의 사용 코드에 있는 [&emsp;] 부분을 수정하여 사용한다.  
- [image file] \: "21.jpg", "3.jpg"와 같이 inputs 폴더 내부에 있는 원본 이미지 (jpg 파일) 명을 그대로 사용한다
- [dist file] \: "21.png', "3.png"와 같이 inputs 폴더 내부에 있는 깊이 이미지 (png 파일) 명을 그대로 사용한다
- [blur value] \: 흐릿함 강도를 조절하는 값으로, 기본 값으로 60이 지정 되어 있다 (0 ~ 120 범위 추정)
- [focus value] \: bokeh 효과를 주는 범위(위치)를 조절하는 값으로, 0 ~ 1 사이의 값을 받는다 (기본 값은 0.35)
- [gamma value] \: 이미지의 선명도를 조절하는 값으로, 1 ~ 5 사이의 값을 받는다 (기본 값은 4)

---

### 추가 작업
입력 데이터로 사용되는 이미지는 원본 이미지와 깊이 이미지 총 두 개로, 원본 이미지는 어떤 이미지를 사용해도 무방하지만, 깊이 이미지는 특정 방법을 이용하여 찍는 것이 아닌 이상, 구할 방법이 없다.  
따라서, 본 논문에서도 사용한 DPT 네트워크를 이용하여 깊이 이미지를 입력 데이터로 구한다.  

DPT(Dense Prediction Transformer) 네트워크 [[DPT Origin]]에서 사용한 ViT(Vision Transformer)는 resnet50 기반 384 x 384 이미지를 입력 데이터로 하는 ViT를 사용한다.
> DPT network : dpt-hybrid-midas --> (Hybrid DPT-base and DPT-large)  
> DPT 코드에 사용된 학습 모델은 [[데이터 파일]]을 통하여 사용할 수 있다.  
> 해당 데이터 파일은 ./weights 폴더에 넣어 사용한다.

DPT 포함한 코드를 demo.py에 포함했기 때문에 실행 코드는 기존과 동일하게 사용한다.  
DPT 네트워크를 추가한 코드로 동작하기 때문에 .jpg 형의 원본 데이터 하나로 Bokeh 효과를 적용시킬 수 있다.

[Bokeh Origin]: https://github.com/JuewenPeng/BokehMe
[code 1]: https://github.com/InZury/BokehMe_Clone/blob/c963476e6654d8ae24b72c1b014c055011f648db/classical_renderer/scatter.py#L125
[code 2]: https://github.com/InZury/BokehMe_Clone/blob/c963476e6654d8ae24b72c1b014c055011f648db/classical_renderer/scatter.py#L133
[DPT Origin]: https://github.com/isl-org/DPT
[데이터 파일]: https://drive.google.com/file/d/181AErGHS8YUrpsyUco-eNYXs6NfEoylg/view?usp=drive_link  