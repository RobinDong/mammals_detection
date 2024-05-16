import cv2
import torch
import time
import openvino as ov

img = cv2.imread("/home/sanbai/dabailu.jpg")
img = cv2.resize(img, (640, 640))
tensor_img = torch.from_numpy(img).unsqueeze(0).permute(0,3,1,2)

core = ov.Core()
model = core.read_model(model="runs/detect/train15/weights/best_int8_openvino_model/best.xml")
cmodel = core.compile_model(model=model, device_name="CPU")
begin = time.time()
for _ in range(10):
	results = cmodel({0: tensor_img})
print(results, time.time() - begin)
