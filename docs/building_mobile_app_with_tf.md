---
layout: page
group: "mobile_app_with_tf"
title: "Building Mobile Applications with TensorFlow"
link_url: http://www.oreilly.com/data/free/building-mobile-applications-with-tensorflow.csp
---

저자 : [Pete Warden](https://www.linkedin.com/in/petewarden/){:target="_blank"}

## 개론

- TensorFlow를 모바일에서 구동시킬 수 있는 방법을 이야기하고 있는 아주 간단한 책자이다.
- 하지만 Mobile 위주보다는 TensorFlow를 어떻게 Application에 탑재하는지를 살펴보도록 하자.
- 참고로 TensorFlow-Lite 가 나오기 이전에 출판된 책이다.
    - 이 부분은 뒤에 조금 더 언급하도록 하자.
- 참고로 초반부에 각 OS별 컴파일 방법은 공식 [README.md](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/makefile/README.md){:target="_blank"} 문서가 더 잘 정리되어 있다.

### 모바일 환경에 적합한 TensorFlow 활용처.

- Speech recognition
- Image recognition
- Object localization
- Gesture recognition
- OCR
- Translation
- Text classification
- Voice synthesis
    
## 사용 가능한 Platform

### Android

- TensorFlow 가 가장 먼저 지원하는 것은 (당연히) Android.
- 개발 Tool은 Android Studio 쓰면 된다.
    - 가장 손쉬운 방법은 새로운 프로젝트를 생성후 `build.gradle` 에 다음 두 줄만 추가하면 된다.
    - 진짜 두 줄이 아닌 것은 그냥 넘어가자.
    
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

- Android Studio 에서는 (당연히) [Bazel](https://bazel.build/){:target="_blank"} 도 지원한다.
    - TensorFlow의 기본 Building Tool은 **Bazel** 이다. (그 모든 짜증의 근원.)
        - Java랑 기타 여러 가지 다른 라이브러리와 Dependency 가 있는 Tool이다.
        - 처음 실행시 메모리를 디따 많이 쓴다.
        - 따라서 Rasperry Pi 같은 저사양 환경에서 다이렉트로 컴파일하기 힘들다. (뭔들)
        - 뭐 cross-compilation이 해결 방법이긴 한데 TensorFlow에 다른 Tool이 있으니 그걸 쓰자. (뒤에 설명된다.)
- 책에는 Android Studio에서 Bazel을 이용한 TensorFlow Build 코드가 포함되어 있다.
    - 우리에게는 별로 중요한 내용은 아니니 넘기도록 하자.
    - 이미 TensorFlow 매뉴얼에 추가되어 있다.

- 추가로 cross-complie 로 생성하는 것도 그리 어렵지 않다.

```sh
$ tensorflow/contrib/makefile/download_dependencies.sh
$ tensorflow/contrib/makefile/compile_android_protobuf.sh -c
$ export HOST_NSYNC_LIB=`tensorflow/contrib/makefile/compile_nsync.sh`
$ export TARGET_NSYNC_LIB=`CC_PREFIX="${CC_PREFIX}" NDK_ROOT="${NDK_ROOT}" tensorflow/contrib/makefile/compile_nsync.sh -t android -a armeabi-v7a`
$ make -f tensorflow/contrib/makefile/Makefile TARGET=ANDROID
```
    
- Android는 Java 베이스이고 Java 환경에서 어떻게 TensorFlow로 Inferece를 하는지 궁금할 수 있겠다.
    - 그런 경우 [이 코드](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/android/java/org/tensorflow/contrib/android/TensorFlowInferenceInterface.java)를 살펴보자. Java에서 TensorFlow를 사용하는 예제 코드이다.
    - 당근 Inference만 들어있다.
        - Java 코드가 꽤나 잘 되어 있는 편이지만 제공되는 함수의 갯수가 아쉽다.
        - pytnon 환경에서 `sess.run()` 을 활용해서 호출하는 정도의 API만 사용 가능하다.
    
### iOS

- 구글이 직접 만들어낸 프로덕트는 아닐지라도 외부에서 만들어낸 좋은 iOS Application 이 많다. (TensorFlow를 이용해서 만든)
    - 자신(구글러)들이 직접 만들지는 않는다는 말과 같다.
- [cocoapods.org](https://cocoapods.org){:target="_blank"} 같은 site에 가보면 TensorFlow pod 를 쉽게 다운로드 할 수 있다.
    - 즉, XCode 프로젝트와의 연동이 쉽다.
    - 시작은 쉽지만 universal binary framework 형태라서 customize는 어렵다. (binary 크기를 줄인다거나 하는.)

#### Makefile을 활용하기.

- 진짜 구닥다리 build tool이지만 생각보다 좋다. (유연한데다가 cross-compile이 잘 된다.)
- 그래서 Tensorflow 에서도 이런 방식의 컴파일을 지원하기 시작했다.
    - 사실 많은 사람들이 요청을 했는데도 무시하다가 도입된지는 얼마 안된다. ("Bazel 써라. 두번 써라" 드립만 냅다 했었다.)
    - 이것은 `tensorflow/contrib/makefile` 을 살펴보면 된다.
        - 관련해서 [README.md](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/makefile/README.md){:target="_blank"} 에 잘 기술되어 있다.
    - 이걸 이용하면 iOS 뿐만 아니라 Linux, Android, Raspberry-Pi 도 모두 사용 가능하다.
    
- **한번에 Build 하기.**
    - iOS 에서 TensorFlow를 compile 하고 싶다면 다음 스크립트를 활용하자.
    - `tensorflow/contrib/makefile/build_all_ios.sh`
    - 참고로 2013 Macbook-Pro에서 한 20분 걸린다.
    
- **수동으로 Build 하기**
    - dependency 가 존재하는 3-party library를 사용자가 고려해서 직접 따로따로 build할 수 있다.
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


### Raspberry Pi

- TensorFlow 팀에서는 라즈베리파이에서 쉽게 TensorFlow를 돌릴 수 있는 방법을 제공한다
    - `pip install` 을 이용해서 pre-built 된 바이너리 파일을 제공한다.

- 다음과 같이 작업을 해도 된다.

- 일단 dependency 라이브러리를 다운받고 필요한 tool 들을 설치한다. (Linux 장비에 받는다.)
    
```sh
$ tensorflow/contrib/makefile/download_dependencies.sh
$ sudo apt-get install -y autoconf automake libtool gcc-4.8 g++-4.8
$ cd tensorflow/contrib/makefile/downloads/protobuf/
$ ./autogen.sh
$ ./configure
$ make
$ sudo make install
$ sudo ldconfig # refresh
```

- 그 다음에 라즈베리파이 모드로 컴파일을 수행한다. (즉, cross-compile을 수행하는 것이다.)

```sh
$ make -f tensorflow/contrib/makefile/Makefile HOST_OS=PI TARGET=PI OPTFLAGS="-Os" CXX=g++-4.8
```

- 라즈베리2 혹은 라즈베리3 만을 다룬다면 추가적인 optimizer 를 켤 수 있다.

```sh
$ make -f tensorflow/contrib/makefile/Makefile HOST_OS=PI TARGET=PI \\
  OPTFLAGS="-Os -mfpu=neon-vfpv4 -funsafe-math-optimizations -ftree-vectorize" CXX=g++-4.8
```

- 한가지 주의할 점은 GCC 4.9 에서는 약간 문제가 있다는 것이다.
    - 참고로 라즈베리파이 Jessie 버전의 이미지는 GCC 버전 중 4.9 가 default 버전이다.
    - 이 경우에는 `__atomic_compare_exhange` 에서 에러가 발생한다.
    - 그래서 위에서는 명확하게 `g++4.8` 을 명시하여 사용한다.
    - 4.8 써라. 꼭 이거 써라.

- 라즈베리파이용 카메라나 이미지 처리 작업은 다음 경로에서 받도록 하자.
    - `tensorflow/contrib/pi_example`
    
    
## 두둥. TenorFlow Library 를 나만의 응용 어플리케이션에 넣어보자.

- 모델을 이용해서 학습을 하고 난 뒤 해야하는 당연한 수순은 자신의 어플리케이션에 이를 통합하는 것.
- 이 말은 TensorFlow 를 하나의 라이브러리로 보고 적당한 헤더를 넣어 컴파일하는 것. 
    - 즉, libarary linking 단계가 필요하다.
- "하지만 이를 위한 C++ 헤더는 TensorFlow Core에 없다."
    - 즉, (라이브러리+노출헤더) 와 같이 간단한 형태로 구성된 산출물들을 만들어 낼 수 없다. (헉!)

- 하지만 앞서 공개된 iOS Build 버전이 힌트가 될 수 있다.

- 앞에서 언급한 `tensorflow/contrib/makefile/gen/lib/libtensorflow-core.a` 파일을 다음과 같이 링크한다.
    - `-L/your/path/tensorflow/contrib/makefile/genlib/` & `-ltensorflow-core`
- 이것만으로는 해결되지 않는다.
    - 프로토콜 버퍼, 너가 필요하다.
    - `-L/your/path/tensorflow/contrib/makefile/gen/protobuf_ios/lib` & `-lprotobuf-lite`
- link시 헤더파일이 문제인데 앞서 언급한 3-party lib 의 헤더도 필요한 상황이다. 다음을 추가하자.
    - **include** 경로에 추가하면 된다.
    - `tensorflow/contrib/makefile/downloads/protobuf/src`
    - `tensorflow/contrib/makefile/downloads`
    - `tensorflow/contrib/makefile/downloads/eigen`
    - `tensorflow/contrib/makefile/gen/proto`
- 모든 binary 를 강제로 포함하여 빌드하는 옵션이 필요하다.
    - iOS 에서는 `force_load` 라는 옵션이 필요하다.
    - 이는 각 os 마다 지원되는 방식이 상이하므로 자신의 os 컴파일러에 맞게 찾아봐야 한다.
    - Linux 같은 경우 다음과 같이 표현 가능하다.
        - `-Wl, --allow-multiple-definition -Wl, --whole-archive`
- 안드로이드는 좀 쉬운데 빌드 jar 파일이 이미 TensorFlow 에 포함되어 있다.
    - 만약 없다면 nightly precompiled version 을 찾아보자.
    
## Global Constructor Magic

- 링크시 가장 많이 발생하는 문제가 `No session factory registered for the given session options` 이다.
- 이런 에러가 나는 이유를 이해하려면 TensorFlow 내부를 조금 이해하고 있어야 한다.
- TensorFlow는 기본적으로 매우 가벼운 Core 라이브러리를 중심으로 여러 기능들이 모듈러한 형태로 개발되어 있다.
    - 필요할 때마다 여러 모듈을 쉽게 섞고 합칠 수 있어야 한다.
    - 이를 위해 C++ 코드 패턴을 이용해서 각각의 모듈 구현 방식을 정형화시켜놓았다.
    - 그리고 이에 맞추어 각각의 라이브러리도 따로 구현해서 업데이트하는 방식을 취한다.

- 최종적으로 `resistration` 패턴을 이용해서 코드들이 구현되어 있다.
    - 아래 코드를 보자.
    
```cpp
class MulKernel : OpKernel {
  Status Compute(OpKernelContext* context) { … }
};
REGISTER_KERNEL(MulKernel, “Mul”);
```

- 이 코드는 독립적인 하나의 `.cc` 파일에 기술되어 최종적으로 여러분의 라이브러리에 링크된다.
- 여기서 중요한 부분은 `REGISTER_KERNEL()` 부분으로 MACRO로 구현되어 있다.
- 이 매크로에 기술된 함수는 반드시 동일한 파일에 구현체(implements)가 있어야 한다.
- 이 매크로는 대략적으로 다음과 같이 번역된다.

```cpp
class RegisterMul {
 public:
  RegisterMul() {
  global_kernel_registry()->Register(“Mul”, []({
    return new MulKernel()});
  }
};
RegisterMul g_register_mul;
```

- - -

# 실습

```
$ ./tensorflow/contrib/makefile/download_dependencies.sh
$ apt-get install install -y autoconf automake libtool 
$  cd tensorflow/contrib/makefile/downloads/protobuf/
$ ./autogen.sh
$ ./configure
$ make
$ sudo make install
$ sudo ldconfig # refresh
```
