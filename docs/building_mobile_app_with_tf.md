---
layout: page
group: "mobile_app_with_tf"
title: "Building Mobile Applications with TensorFlow"
link_url: http://www.oreilly.com/data/free/building-mobile-applications-with-tensorflow.csp
---

저자 : [Pete Warden](https://www.linkedin.com/in/petewarden/){:target="_blank"}

## 개론

- 모바일 환경에 적합한 TensorFlow 활용처. (조금은 뻔한)
    - Speech recognition
    - Image recognition
    - Object localization
    - Gesture recognition
    - OCR
    - Translation
    - Text classification
    - Voice synthesis
    
## 사용 가능한 Platform

#### Android

- TensorFlow 가 가장 먼저 지원하는 것은 (당연히) Android.
- 개발 Tool은 Android Studio 쓰면 된다.
    - 가장 손쉬운 방법은 새로운 프로젝트를 생성후 `build.gradle` 에 다음 두 줄만 추가하면 된다.
    - 진짜 2줄이 아닌 것은 그냥 넘어가자.
    
```
allprojects {
  repositories {
    jcenter()
  }
}
dependencies {
  compile 'org.tensorflow:tensorflow-android:+'
}
```

- Android Studio 에서는 Bazel도 지원한다.
    - PC 환경에서는 TensorFlow의 기본 building tool은 Bazel이다.
    - 근데 Java랑 기타 여러 가지 dependency 가 있는 tool이다.
    - 처음 실행시 메모리를 디따 많이 쓴다.
    - 따라서 Rasperry Pi 같은 저사양 환경에서 컴파일하기 힘들다.
    - 뭐 cross-compilation이 해결 방법이긴 한데 TensorFlow에 다른 tool이 있으니 그걸 쓰자. (뒤에 설명)
- 책에는 Android Studio에서 Bazel을 이용한 TensorFlow Build 코드가 포함되어 있다.
    - 우리에게는 별로 중요한 내용은 아니니 넘기도록 하자.
- Java 환경에서 어떻게 TensorFlow로 Inferece를 하는지 궁금할 수 ㅇ도 있겠다.
    - 그런 경우 [이 코드](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/android/java/org/tensorflow/contrib/android/TensorFlowInferenceInterface.java)를 살펴보자. Java에서 TensorFlow를 사용하는 예제 코드이다.
    - 물론 Inference만 들어있다.
    - (참고) Java 코드가 꽤나 잘 되어 있는데 pytnon 환경에서 `sess.run()` 을 활용해서 호출하는 정도의 API는 이미 구현되어 있다.
    
### iOS

- 구글이 직접 만들어낸 프로덕트는 아닐지라도 외부에서 만들어낸 좋은 iOS Application 이 많다. (TensorFlow를 이용해서 만든)
- cocoapods.org 같은 site에 가보면 TensorFlow pod 를 쉽게 다운로드 할 수 있다.
    - XCode 프로젝트와의 연동이 쉽다.
    - 시작은 쉽지만 universal binary framework 형태라서 customize는 어렵다. (binary 크기를 줄인다거나 등)

#### Makefile

- 진짜 구닥다리 build tool이지만 생각보다 좋다. (유연한데다가 cross-compile이 잘 된다.)
- 그래서 Tensorflow 에서도 이런 방식의 컴파일을 지원하기 시작했다.
    - `tensorflow/contrib/makefile` 을 살펴보면 된다.
    - 이걸 이용하면 iOS 뿐만 아니라 Linux, Android, Raspberry-Pi 도 모두 사용 가능하다.
    
- **한번에 Build 하기.**
    - iOS 에서 TensorFlow를 compile 하고 싶다면 다음 스크립트를 활용하자.
    - `tensorflow/contrib/makefile/build_all_ios.sh`
    - 2013 Macbook-Pro에서 한 20분 걸린다.
    
- **수동으로 Build 하기**
    - dependency를 사용자가 고려해서 직접 따로따로 build할 수 있다.
    - 먼저 dependency libarary 를 다운받자.
        - `tensorflow/contib/makefile/download_dependencies.sh`
    - 자, 이제 build한다.
        - `tensorflow/contrib/makefile/compile_ios_protobuf.sh`
    - 이제 depencency libarary를 build하였으니 최종 build를 수행한다.
        - `make -f tensorflow/contrib/makefile/Makefile TARGET=IOS IOS_ARCH=ARM64`
    - 이렇게 하면 최종적으로 다음 파일을 얻을 수 있다.
        - `tensorflow/contrib/makefile/gen/lib/libtensorflow-core.a`
        
#### Optimization

- compile 옵션을 통해 compile 시 사용될 옵션 값들을 조정 가능하다.

```
$ compile_ios_tensorflow.sh "-Os"
```

- 좀 더 자세한 내용은 옵션 flag 를 참고하도록 하자.

#### iOS example

- 이미 TensorFlow 에 iOS용 예제가 들어있다. ( `tensorflow/contrib/ios_example` )


## Raspberry Pi

- TensorFlow 팀에서는 Pi 에서 쉽게 TensorFlow 를 돌릴 수 있는 방법을 제공한다.
    - `pip install` 을 이용해서 pre-built 된 바이너리 파일을 제공한다.
- 다음과 같이 작업을 해도 된다.
    - 일단 dependency 라이브러리를 다운받고 필요한 tool 들을 설치한다. (linux 장비에...)
    - 그 다음에 PI 용 컴파일을 수행한다.
    
```
$ tensorflow/contrib/makefile/download_dependencies.sh
$ sudo apt-get install -y autoconf automake libtool gcc-4.8 g++-4.8
$ cd tensorflow/contrib/makefile/downloads/protobuf/
$ ./autogen.sh
$ ./configure
$ make
$ sudo make install
$ sudo ldconfig # refresh
```

```
make -f tensorflow/contrib/makefile/Makefile HOST_OS=PI TARGET=PI OPTFLAGS="-Os" CXX=g++-4.8
```

- 라즈베리2 혹은 라즈베리3 만을 다룬다면 optimizer 를 켤 수 있다.

```
make -f tensorflow/contrib/makefile/Makefile HOST_OS=PI TARGET=PI OPTFLAGS="-Os -mfpu=neon-vfpv4 -funsafe-math-optimizations -ftree-vectorize" CXX=g++-4.8
```

- 한가지 주의할 점은 GCC 4.9 에서는 약간 문제가 있다는 것이다.
    - 참고로 라즈베리파이 Jessie 버전 이미지는 GCC 4.9 가 default이다.
    - `__atomic_compare_exhange` 에러.
    - 그래서 위에서는 명확하게 `g++4.8` 을 명시하여 사용한다.
    - 4.8 써라.

- 라즈베리파이용 카메라나 이미지 처리 작업은 다음 경로에서 받도록 하자.
    - `tensorflow/contrib/pi_example`
    
    
## TenorFlow Library 를 응용 어플리케이션에 넣어보자.

- 모델을 이용해서 학습을 하고 난 뒤 해야하는 당연한 수순은 자신의 어플리케이션에 이를 통합하는 것.
- 이 말은 TensorFlow 를 하나의 라이브러리로 보고 적당한 헤더를 넣어 컴파일하는 것.
- "하지만 이를 위한 C++ 헤더는 TensorFlow Core에 없다."
    - 즉, (라이브러리+노출헤더) 와 같이 간단한 형태로 구성된 산출물들을 만들어 낼 수 없다.
    
