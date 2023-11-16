import torch

input_lists = []
all_samples_list = []
for i in range(0,1000,100):
    data_path = 'imagenet_input_50steps_{}_{}_sd.pth'.format(i, i+100)
    input_list = torch.load(data_path, map_location='cpu')
    input_lists.extend(input_list)
    data_path = 'imagenet_samples_ddim_50steps_{}_{}_sd.pth'.format(i, i+100)
    all_samples = torch.load(data_path, map_location='cpu')
    all_samples_list.extend(all_samples)
    
torch.save(all_samples_list, 'imagenet_samples_ddim_{}steps_sd.pth'.format(50))
torch.save(input_lists, 'imagenet_input_{}steps_sd.pth'.format(50))