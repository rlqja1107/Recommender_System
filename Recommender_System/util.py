import torch

default_device='cpu'
 
def set_device_cuda(): 
        device='cuda' if torch.cuda.is_available() else 'cpu'
        assert device =='cuda', 'Cuda is not available'
        torch.cuda.device(device)
        

def cuda_is_avail(): 
        if torch.cuda.is_available():
                gpu_num=torch.cuda.device_count()-1
                print("Cuda is Available, Use Cuda {:d}".format(gpu_num))
                return True
        else:
                print("Cuda is not Available")
                return False
                
