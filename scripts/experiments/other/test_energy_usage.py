import pynvml
import time
import threading

from reconsnet.model.diffusion import DiffusionModule
from reconsnet.data.dataset import ClinicalDataset


STEPS = 50

def energy_consumption(model: DiffusionModule, sample, dt=0.1):
    pynvml.nvmlInit()
    
    (bp, _, p0, p1), _, _ = sample
    bp = bp.to(model.device).unsqueeze(0)
    p0 = p0.to(model.device).unsqueeze(0).unsqueeze(0)
    p1 = p1.to(model.device).unsqueeze(0).unsqueeze(0)
 
    handle = pynvml.nvmlDeviceGetHandleByIndex(model.device.index)
    
    work = 0
    stop = False
    
    def sample_power():
        nonlocal work
        while not stop:
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            work += power * dt # simple rectangle method integration
            time.sleep(0.1)
    
    
    thread = threading.Thread(target=sample_power)
    thread.start()
    
    t0 = time.time()
    model.fast_reconstruct(
        bp, p0, p1,
        num_inference_steps=STEPS
    )
    elapsed = time.time() - t0
    
    stop = True
    thread.join()

    pynvml.nvmlShutdown()
    return work, elapsed


if __name__ == "__main__":
    model = DiffusionModule.load_from_checkpoint("/home/shared/model-weights/both.ckpt", lr=1e-4)
    model.eval()
    ds = ClinicalDataset("/home/shared/uck-right")
    ec, t = energy_consumption(model, ds[0])
    print(f"Energy consuption={ec} [J], in time={t} [s], Average power={ec/t} [W]")
