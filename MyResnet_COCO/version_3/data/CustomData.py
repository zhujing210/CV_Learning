from PIL import Image
from torch.utils.data import Dataset
# use PIL Image to read image
def default_loader(path):
    try:
        img = Image.open(path)
        return img.convert('RGB')
    except:
        print("Cannot read image: {}".format(path))


# define your Dataset. Assume each line in your .txt file is [name/tab/label], for example:0001.jpg 1
class customData(Dataset):
    def __init__(self, img_path, txt_path, data_transforms=None, loader = default_loader):
        with open(txt_path, 'r') as input_file:
            lines = input_file.readlines()
            # self.img_name = [os.path.join(img_path, line.strip().split('\t')[0]) for line in lines]
            self.img_name = [line.strip().split('\t')[0] for line in lines]
            self.img_label = [int(line.strip().split('\t')[-1]) for line in lines]
        self.data_transforms = data_transforms
        self.loader = loader

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        label = self.img_label[item]
        img = self.loader(img_name)

        if self.data_transforms is not None:
            try:
                img = self.data_transforms(img)
            except:
                print("Cannot transform image: {}".format(img_name))
        return img, label