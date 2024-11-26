include ../mkenv.mk

include $(SDK_SRC_ROOT_DIR)/.config

dir-y :=

dir-$(CONFIG_RTT_AI_EXAMPLES_ANOMALY_DET) += anomaly_det
dir-$(CONFIG_RTT_AI_EXAMPLES_BYTETRACK) += bytetrack
dir-$(CONFIG_RTT_AI_EXAMPLES_CROSSWALK_DETECT) += crosswalk_detect
dir-$(CONFIG_RTT_AI_EXAMPLES_DEC_AI_ENC) += dec_ai_enc
dir-$(CONFIG_RTT_AI_EXAMPLES_DEMO_MIX) += demo_mix
dir-$(CONFIG_RTT_AI_EXAMPLES_DISTRACTION_REMINDER) += distraction_reminder
dir-$(CONFIG_RTT_AI_EXAMPLES_DMS_SYSTEM) += dms_system
dir-$(CONFIG_RTT_AI_EXAMPLES_DYNAMIC_GESTURE) += dynamic_gesture
dir-$(CONFIG_RTT_AI_EXAMPLES_EYE_GAZE) += eye_gaze
dir-$(CONFIG_RTT_AI_EXAMPLES_FACE_ALIGNMENT) += face_alignment
dir-$(CONFIG_RTT_AI_EXAMPLES_FACE_DETECTION) += face_detection
dir-$(CONFIG_RTT_AI_EXAMPLES_FACE_EMOTION) += face_emotion
dir-$(CONFIG_RTT_AI_EXAMPLES_FACE_GENDER) += face_gender
dir-$(CONFIG_RTT_AI_EXAMPLES_FACE_GLASSES) += face_glasses
dir-$(CONFIG_RTT_AI_EXAMPLES_FACE_LANDMARK) += face_landmark
dir-$(CONFIG_RTT_AI_EXAMPLES_FACE_MASK) += face_mask
dir-$(CONFIG_RTT_AI_EXAMPLES_FACE_MESH) += face_mesh
dir-$(CONFIG_RTT_AI_EXAMPLES_FACE_PARSE) += face_parse
dir-$(CONFIG_RTT_AI_EXAMPLES_FACE_POSE) += face_pose
dir-$(CONFIG_RTT_AI_EXAMPLES_FACE_VERIFICATION) += face_verification
dir-$(CONFIG_RTT_AI_EXAMPLES_FALLDOWN_DETECT) += falldown_detect
dir-$(CONFIG_RTT_AI_EXAMPLES_FINGER_GUESSING) += finger_guessing
dir-$(CONFIG_RTT_AI_EXAMPLES_FITNESS) += fitness
dir-$(CONFIG_RTT_AI_EXAMPLES_HEAD_DETECTION) += head_detection
dir-$(CONFIG_RTT_AI_EXAMPLES_HELMET_DETECT) += helmet_detect
dir-$(CONFIG_RTT_AI_EXAMPLES_KWS) += kws
dir-$(CONFIG_RTT_AI_EXAMPLES_LICENCE_DET) += licence_det
dir-$(CONFIG_RTT_AI_EXAMPLES_LICENCE_DET_REC) += licence_det_rec
dir-$(CONFIG_RTT_AI_EXAMPLES_NANOTRACKER) += nanotracker
dir-$(CONFIG_RTT_AI_EXAMPLES_OBJECT_DETECT_YOLOV8N) += object_detect_yolov8n
dir-$(CONFIG_RTT_AI_EXAMPLES_OCR) += ocr
dir-$(CONFIG_RTT_AI_EXAMPLES_PERSON_ATTR) += person_attr
dir-$(CONFIG_RTT_AI_EXAMPLES_PERSON_DETECT) += person_detect
dir-$(CONFIG_RTT_AI_EXAMPLES_PERSON_DISTANCE) += person_distance
dir-$(CONFIG_RTT_AI_EXAMPLES_POSE_DETECT) += pose_detect
dir-$(CONFIG_RTT_AI_EXAMPLES_PPHUMANSEG) += pphumanseg
dir-$(CONFIG_RTT_AI_EXAMPLES_PUZZLE_GAME) += puzzle_game
dir-$(CONFIG_RTT_AI_EXAMPLES_SEGMENT_YOLOV8N) += segment_yolov8n
dir-$(CONFIG_RTT_AI_EXAMPLES_SELF_LEARNING) += self_learning
dir-$(CONFIG_RTT_AI_EXAMPLES_SMOKE_DETECT) += smoke_detect
dir-$(CONFIG_RTT_AI_EXAMPLES_SPACE_RESIZE) += space_resize
dir-$(CONFIG_RTT_AI_EXAMPLES_SQ_HAND_DET) += sq_hand_det
dir-$(CONFIG_RTT_AI_EXAMPLES_SQ_HANDKP_CLASS) += sq_handkp_class
dir-$(CONFIG_RTT_AI_EXAMPLES_SQ_HANDKP_DET) += sq_handkp_det
dir-$(CONFIG_RTT_AI_EXAMPLES_SQ_HANDKP_FLOWER) += sq_handkp_flower
dir-$(CONFIG_RTT_AI_EXAMPLES_SQ_HANDKP_OCR) += sq_handkp_ocr
dir-$(CONFIG_RTT_AI_EXAMPLES_SQ_HANDRECO) += sq_handreco
dir-$(CONFIG_RTT_AI_EXAMPLES_TRAFFIC_LIGHT_DETECT) += traffic_light_detect
dir-$(CONFIG_RTT_AI_EXAMPLES_TRANSLATE_EN_CH) += translate_en_ch
dir-$(CONFIG_RTT_AI_EXAMPLES_TTS_ZH) += tts_zh
dir-$(CONFIG_RTT_AI_EXAMPLES_VEHICLE_ATTR) += vehicle_attr
dir-$(CONFIG_RTT_AI_EXAMPLES_VIRTUAL_KEYBOARD) += virtual_keyboard
dir-$(CONFIG_RTT_AI_EXAMPLES_YOLOP_LANE_SEG) += yolop_lane_seg

# Add directories to the build system
dirs := $(sort $(dir-y))

.PHONY: all clean

# Default build arguments
BUILD_ARG ?= "hdmi"
BUILD_OUT ?= k230_bin

# Change BUILD_OUT if LCD mode is enabled
ifeq ($(CONFIG_RTT_AI_EXAMPLES_MODE_LCD),y)
	BUILD_ARG := "lcd"
	BUILD_OUT := k230_bin_lcd
endif

# List of directories to process (define your directories here)
dirs := $(dir-y)

# Default target
all:
ifeq ($(CONFIG_RTT_ENABLE_BUILD_AI_EXAMPLES),y)
	@$(foreach dir,$(dirs), \
		echo "Building in directory: $(dir) with BUILD_ARG=$(BUILD_ARG)"; \
		./build_app_sub_rtos_only.sh $(dir) $(BUILD_ARG);\
		if [ -d $(BUILD_OUT) ]; then \
			echo "Copying $(BUILD_OUT) to $(RTT_EXAMPLES_ELF_INSTALL_PATH)"; \
			cp -r $(BUILD_OUT)/* $(RTT_EXAMPLES_ELF_INSTALL_PATH); \
		else \
			echo "No build output found in $(dir)/$(BUILD_OUT), skipping copy."; \
		fi;)
endif
	@echo "Make AI samples done."

clean:
	@rm -rf out
	@rm -rf k230_bin, k230_bin_lcd
	@echo "Clean ai samples done."
