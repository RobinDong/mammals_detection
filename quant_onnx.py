import subprocess

from onnxruntime.quantization import quantize_dynamic, QuantType

output_model_name = "best"

subprocess.run(f"python3 -m onnxruntime.quantization.preprocess --input {output_model_name}.onnx --output {output_model_name}_infer.onnx", shell=True)

quantize_dynamic(f"{output_model_name}_infer.onnx", "quant.onnx", weight_type=QuantType.QUInt8)
