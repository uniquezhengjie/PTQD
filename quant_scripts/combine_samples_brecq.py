import numpy as np

all_images_list = []
all_labels_list = []

files = [1103, 1108, 1109]
for name in files:
    # data_path = 'generated/brecq_w8a8_50000steps20eta0.0scale3.0_{}_fp32.npz'.format(name)
    # data_path = 'generated/brecq_w8a8_50000steps20eta0.0scale3.0_{}_qdecoder.npz'.format(name)
    data_path = 'generated/brecq_w8a8_50000steps20eta0.0scale3.0_{}.npz'.format(name)
    print(data_path)
    ckpt = np.load(data_path)
    all_images = ckpt['arr_0']
    all_labels = ckpt['arr_1']
    print(len(all_images))
    all_images_list.extend(all_images)
    all_labels_list.extend(all_labels)

arr = np.array(all_images_list)
label_arr = np.array(all_labels_list)

# out_path = 'generated/brecq_w8a8_50000steps20eta0.0scale3.0_{}_fp32.npz'.format(len(all_images_list))
# out_path = 'generated/brecq_w8a8_50000steps20eta0.0scale3.0_{}_qdecoder.npz'.format(len(all_images_list))
out_path = 'generated/brecq_w8a8_50000steps20eta0.0scale3.0_{}.npz'.format(len(all_images_list))
np.savez(out_path, arr, label_arr)