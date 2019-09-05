PG2019 코드 정리 v1

x : 포함 안됨
n : 솔루션에 포함 안됨

 모델 생성
DoubleFusion(x)
RGBDCapture(n)
MakeMeshFilefromScans(x)
L0Registration(x)

 전처리
InputPreprocessing(n)
    - raw color -> bilateral filtering
    - model -> rendered depth
    
UVAtlas(x)

 텍스쳐 생성
TextureMappingNonRigid
    configuration 파일 및 shader들은 프로젝트 내에서 수정하면 자동으로 복사됨
    - conf.json : 전체 설정 파일, 각 모듈에 해당하는 설정값을 할당
    - Mapper : projection, keyframe sampling, leyer간 값 전달
    - Simplifier : decimation, vertex tree
    - Optimizer : layer당 최적화
    - Renderer : atlas / sub-atlas rendering, model rendering

    conf.json
        먼저 큰 범주로 case를 나누어 저장해둘 수 있음
        - "main"
            "data_root_path" : parameterized mesh (test_tex.obj), mesh 및 RGB-D stream이 있어야함
            "is_viwer" : 최적화를 생략하고 만들어져 있는 아틀라스를 사용할것인지
        - "mapper4D"
            "depth_test" : valid projection threshold
        - "simplifier"
            "div_per_layer" : 이전 층 vertex / 다음 층 vertex
        - "optimizer"
            "from_file" : 이전에 만들어둔 mapping결과를 사용 (구현안됨)
        - "renderer"
            "rendering_spots" : 모델 렌더링 시 해당하는 숫자키를 누르면 카메라와 모델의 포즈가 설정된 값으로 변경
        - "any"
            "mesh_extension" : mesh stream의 확장자
            "face_normal_clockwise" : normal이 반대로 감겨진 경우 false
  
    실행 예시 :  ./TextureMappingNonRigid.exe "./conf.json", "case_template"
                ./TextureMappingNonRigid.exe "./myConf.json", "case_hyomin"

    viewer key setting
        - F1, F2 : model, camera 이동 모드 변환
        - w,s,a,d : 앞,뒤 이동, 회전
        - mouse : 회전(arcball)
        - c : camera pose capture (cmd 창 출력) -> 복사하여 conf.json(rendering_spots)에 넣으면 됨
        - o : object pose capture
        - space bar : play & stop
        - n,m : geometry, naive, single, multi mode변환
        - r : 화면 캡쳐
        - 0~9 : 스팟 변경 (1 : 원위치)
        - esc : 종료