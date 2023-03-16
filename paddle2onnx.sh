paddle2onnx --model_dir ./weights/rec_ppocr_0.948 \
--model_filename inference.pdmodel \
--params_filename inference.pdiparams \
--save_file ./weights/plate_rec_onnx/plate_rec.onnx \
--opset_version 10 \
--input_shape_dict="{'x':[-1,3,-1,-1]}" \
--enable_onnx_checker True