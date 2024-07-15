#!/bin/sh

wget --content-disposition -P "$(dirname $0)" https://github.com/timmh/DPT/releases/download/onnx_v0.1/dpt_hybrid-midas-6c3ec701.onnx
wget --content-disposition -P "$(dirname $0)" https://github.com/timmh/MegaDetectorLite/releases/download/v0.2/md_v5a.0.0.onnx
wget --content-disposition -P "$(dirname $0)" https://github.com/timmh/segment-anything/releases/download/v1.0.0/sam_vit_b_01ec64_encoder.onnx
wget --content-disposition -P "$(dirname $0)" https://github.com/timmh/segment-anything/releases/download/v1.0.0/sam_vit_b_01ec64_decoder.onnx
wget --content-disposition -P "$(dirname $0)" https://github.com/timmh/Depth-Anything/releases/download/onnx_v0.1/depth_anything_metric_depth_outdoor.onnx
wget --content-disposition -P "$(dirname $0)" https://github.com/timmh/Metric3D/releases/download/v0.1/metric3d_vit_small.onnx