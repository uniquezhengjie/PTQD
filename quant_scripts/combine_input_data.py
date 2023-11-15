import torch

# input_lists = []
# all_samples_list = []
# for i in range(0,1000,100):
#     data_path = 'generated/imagenet_input_{}steps_{}eta_{}_{}.pth'.format(100, 1.0, i, i+100)
#     input_list = torch.load(data_path, map_location='cpu')
#     input_lists.extend(input_list)
#     data_path = 'generated/imagenet_samples_ddim_{}steps_{}eta_{}_{}.pth'.format(100, 1.0, i, i+100)
#     all_samples = torch.load(data_path, map_location='cpu')
#     all_samples_list.extend(all_samples)
    
# torch.save(all_samples_list, 'generated/imagenet_samples_ddim_100steps_1.0eta.pth'.format(20))
# torch.save(input_lists, 'generated/imagenet_input_100steps_1.0eta.pth')

data_error_list = []
for i in range(49,1000,50):
    print(i)
    data_path = 'generated/data_error_t_w8a8_scale1.5_eta1.0_step100_{}.pth'.format(i)
    input_list = torch.load(data_path, map_location='cpu')
    data_error_list.extend(input_list)

data_path = 'generated/data_error_t_w8a8_scale1.5_eta1.0_step100_final.pth'
input_list = torch.load(data_path, map_location='cpu')
data_error_list.extend(input_list)

print(len(data_error_list))
torch.save(data_error_list, 'generated/data_error_t_w8a8_scale1.5_eta1.0_step100.pth')