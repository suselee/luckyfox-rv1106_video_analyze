# daughter_watch 板端服务 - RV1106 交叉编译
# 产物: daughter_watch (主服务) + enroll (底库注册) + decode_test (硬解验证)

TOOLCHAIN_DIR ?= /home/suse/arm-rockchip830-linux-uclibcgnueabihf
CC  = $(TOOLCHAIN_DIR)/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc
CXX = $(TOOLCHAIN_DIR)/bin/arm-rockchip830-linux-uclibcgnueabihf-g++

# RKNN SDK (与现有 rknn_face_emotion 工程一致; 如路径不同请改这里)
RKNN_SDK_DIR ?= /mnt/f/rknn/rknn-toolkit2/rknpu2
RKNN_API_INC = $(RKNN_SDK_DIR)/runtime/Linux/librknn_api/include
RKNN_API_LIB = $(RKNN_SDK_DIR)/runtime/Linux/librknn_api/armhf-uclibc
THIRDPARTY   = $(RKNN_SDK_DIR)/examples/3rdparty

# RockIt/MPP 硬解: 直接指向克隆的 luckfox-pico SDK 内路径 (无需拷贝)
# 若之后把头文件/库拷进 third_party 或删了仓库, 改这两行即可
LUCKFOX_SDK_DIR ?= /home/suse/arm-rockchip830-linux-uclibcgnueabihf/daughter_watch/board_service/luckfox-pico
ROCKIT_INC = $(LUCKFOX_SDK_DIR)/media/rockit/rockit/mpi/sdk/include
ROCKIT_LIB = $(LUCKFOX_SDK_DIR)/media/rockit/rockit/lib/lib32
ROCKIVA_DIR = $(LUCKFOX_SDK_DIR)/media/iva/iva
ROCKIVA_INC = $(ROCKIVA_DIR)/librockiva/rockiva-rv1106-Linux/include
ROCKIVA_LIB = $(ROCKIVA_DIR)/librockiva/rockiva-rv1106-Linux/lib
ROCKIVA_MODEL_DIR = $(ROCKIVA_DIR)/models/rockiva_data_rv1106
FACE_MODEL_DIR ?= /home/suse/arm-rockchip830-linux-uclibcgnueabihf/daughter_watch/models

SRC = src
CXXFLAGS = -O2 -Wall -std=c++11
CXXFLAGS += -I$(SRC) -I$(RKNN_API_INC) -I$(THIRDPARTY) -I$(ROCKIT_INC) -I$(ROCKIVA_INC)

RKNN_LDFLAGS   = -L$(RKNN_API_LIB) -lrknnmrt
# VDEC 的 RK_MPI_VDEC_* 符号由 librockit_full.so 导出；普通 librockit.so 不导出。
# librockit_full.so 依赖 librockchip_mpp.so / librga.so (板子 /oem/usr/lib 里有);
# 链接期允许这些来自共享库的未定义符号, 运行时再解析。
ROCKIT_LDFLAGS = -L$(ROCKIT_LIB) -lrockit_full -Wl,--allow-shlib-undefined
ROCKIVA_LDFLAGS = -L$(ROCKIVA_LIB) -lrockiva
COMMON_LDFLAGS = -lm -ldl -lpthread

INSTALL_DIR = install

# 公共对象 (检测/识别相关)
COMMON_OBJ = $(SRC)/rknn_engine.o $(SRC)/face_detect.o $(SRC)/face_recog.o \
             $(SRC)/facedb.o $(SRC)/stb_impl.o

MAIN_OBJ   = $(SRC)/main.o $(SRC)/config.o $(SRC)/h264_source.o $(SRC)/mqtt_publisher.o \
             $(SRC)/mpp_decoder.o $(SRC)/rockiva_detector.o $(SRC)/track_fusion.o \
             $(SRC)/system_monitor.o $(SRC)/schedule.o
ENROLL_OBJ = $(SRC)/enroll.o
DECTEST_OBJ = $(SRC)/decode_test.o $(SRC)/mpp_decoder.o $(SRC)/stb_impl.o
PROBE_OBJ = $(SRC)/rockiva_probe.o $(SRC)/rockiva_detector.o $(SRC)/system_monitor.o \
            $(SRC)/config.o $(SRC)/h264_source.o $(SRC)/mpp_decoder.o
FUSION_TEST_OBJ = $(SRC)/track_fusion_test.o $(SRC)/track_fusion.o
SCHEDULE_TEST_OBJ = $(SRC)/schedule_test.o $(SRC)/schedule.o

.PHONY: all clean install

all: daughter_watch enroll decode_test rockiva_probe track_fusion_test schedule_test

daughter_watch: $(COMMON_OBJ) $(MAIN_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(RKNN_LDFLAGS) $(ROCKIT_LDFLAGS) $(ROCKIVA_LDFLAGS) $(COMMON_LDFLAGS)

enroll: $(COMMON_OBJ) $(ENROLL_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(RKNN_LDFLAGS) $(COMMON_LDFLAGS)

# 硬解验证工具: 只依赖 rockit + stb, 不需要 rknn
decode_test: $(DECTEST_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(ROCKIT_LDFLAGS) $(COMMON_LDFLAGS)

rockiva_probe: $(PROBE_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(ROCKIVA_LDFLAGS) $(RKNN_LDFLAGS) \
		$(ROCKIT_LDFLAGS) $(COMMON_LDFLAGS)

track_fusion_test: $(FUSION_TEST_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(COMMON_LDFLAGS)

schedule_test: $(SCHEDULE_TEST_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(COMMON_LDFLAGS)

$(SRC)/%.o: $(SRC)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

install: all
	@mkdir -p $(INSTALL_DIR)
	cp daughter_watch enroll decode_test rockiva_probe track_fusion_test schedule_test $(INSTALL_DIR)/
	cp $(RKNN_API_LIB)/librknnmrt.so $(INSTALL_DIR)/
	cp $(ROCKIT_LIB)/librockit_full.so $(INSTALL_DIR)/
	cp $(ROCKIVA_LIB)/librockiva.so $(INSTALL_DIR)/
	@mkdir -p $(INSTALL_DIR)/models/rockiva
	cp $(ROCKIVA_MODEL_DIR)/object_detection_pfp.data $(INSTALL_DIR)/models/rockiva/
	cp config.example.ini $(INSTALL_DIR)/config.ini
	cp ../deploy/run.sh ../deploy/S98daughter_watch \
	   ../deploy/install_on_board.sh ../deploy/rollback_on_board.sh $(INSTALL_DIR)/
	@if [ -f $(FACE_MODEL_DIR)/face_detector.rknn ] && [ -f $(FACE_MODEL_DIR)/mobilefacenet.rknn ]; then \
		mkdir -p $(INSTALL_DIR)/models; \
		cp $(FACE_MODEL_DIR)/face_detector.rknn $(FACE_MODEL_DIR)/mobilefacenet.rknn \
		   $(FACE_MODEL_DIR)/conversion_report.json \
		   $(INSTALL_DIR)/models/; \
		awk '/\.rknn$$/' $(FACE_MODEL_DIR)/checksums.sha256 > $(INSTALL_DIR)/models/checksums.sha256; \
		if [ -f $(FACE_MODEL_DIR)/daughter.db ]; then cp $(FACE_MODEL_DIR)/daughter.db $(INSTALL_DIR)/models/; fi; \
	fi
	@echo ""
	@echo "=== Build complete ==="
	@echo "确保 LD_LIBRARY_PATH 同时包含安装目录和板子的 /oem/usr/lib"
	@echo "先验证硬解:  ./decode_test test.h264 out.jpg"

clean:
	rm -f $(SRC)/*.o daughter_watch enroll decode_test rockiva_probe track_fusion_test schedule_test
	rm -rf $(INSTALL_DIR)
