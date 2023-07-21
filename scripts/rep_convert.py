import torch
from thop import profile
from copy import deepcopy
from models.experimental import attempt_load

def model_print(model, img_size):
    # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients

    stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32
    img = torch.zeros((1, model.yaml.get('ch', 3), stride, stride), device=next(model.parameters()).device)  # input
    flops = profile(deepcopy(model), inputs=(img,), verbose=False)[0] / 1E9 * 2  # stride GFLOPS
    img_size = img_size if isinstance(img_size, list) else [img_size, img_size]  # expand if int/float
    fs = ', %.6f GFLOPS' % (flops * img_size[0] / stride * img_size[1] / stride)  # imh x imw GFLOPS

    print(f"Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}")

if __name__ == '__main__':
    #load = r'F:\Anker\python\YOLOv5-Lite\weights/v5lite-e.pt'
    load = r'C:\Pro\pythonProject\Yolo-reim\yolo-fastest\YOLOv5-Lite\best.pt'
    save = r'C:\Pro\pythonProject\Yolo-reim\yolo-fastest\YOLOv5-Lite\Frepv5lite-e.pt'
    test_size = 416
    print(f'Done. Befrom weights:({load})')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1 = attempt_load(load, map_location=device)  # load FP32 model
    torch.save(model1, save)
    model2 = attempt_load(save, map_location=device,way=2)  # load FP32 model

    model_print(model2, test_size)
    print(model2)


# def attempt_load(weights, map_location=None):
#     # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
#     model = Ensemble()
#     D = {} # 修改1
#     for w in weights if isinstance(weights, list) else [weights]:
#     attempt_download(w)
#     ckpt = torch.load(w, map_location=map_location) # load
#     D['model'] = ckpt # 修改2
#     #model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval()) # FP32 model
#     model.append(D['model'].float().fuse().eval()) # FP32 model # 修改3
