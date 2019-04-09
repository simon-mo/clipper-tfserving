# This Makefile is for development purpose
all: build
build:
	# TF Serving
	docker build -t gcr.io/clipper-model-comp/tf-serving-base:bench -f TFServingBaseDockerfile .
	docker build -t gcr.io/clipper-model-comp/tf-serving-inception:bench -f InceptionV3Dockerfile .
	docker build -t gcr.io/clipper-model-comp/tf-serving-res50:bench -f Res50Dockerfile .

	# Clipper
	docker build -t gri.io/clipper-model-comp/clipper-serving-inception:bench -f clipper/InceptionV3Dockerfile .
	docker build -t gri.io/clipper-model-comp/clipper-serving-res50:bench -f clipper/Res50Dockerfile .
	