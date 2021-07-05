# Serve model as a flask application
import os
import torch
import aiohttp
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from flask import Flask, request
import numpy as np
import random
from PIL import Image
import PIL.ImageOps
from werkzeug.utils import secure_filename


matchingModel = None
UPLOAD_FOLDER = '/path/uploads/testing'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

model_file_url = 'https://drive.google.com/file/d/1Pbl-Un8Uoz8IrHcgc-tPx1L1gbVXyNTl/view?usp=sharing'
model_file_name = 'match-mode.pt'


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)

download_file(model_file_url, UPLOAD_FOLDER / model_file_name)
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8*100*100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

class SiameseNetworkDataset(Dataset):
    def __init__(self,imageFolderDataset,transform=None,should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        #we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1)
        if should_get_same_class:
            while True:
                #keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1]==img1_tuple[1]:
                    break
        else:
            while True:
                #keep looping till a different class image is found

                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] !=img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)

class Config():
    testing_dir = "/path/uploads/testing"

def load_matchingModel():
    global matchingModel
    with open('match-model.pt', 'rb') as f:
        matchingModel = SiameseNetwork()
        matchingModel.load_state_dict(torch.load(f))
        matchingModel.eval()


@app.route('/')
def home_endpoint():
    return 'Hello World!'

@app.route('/match', methods=['POST'])
def Matching():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return str("No file part")

        file = request.files['file']

        if file.filename == '':
            return str("No selected file")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        folder_dataset_test = dset.ImageFolder(root=Config.testing_dir)
        siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test, transform=transforms.Compose([transforms.Resize((100,100)), transforms.ToTensor()]) ,should_invert=False)
        test_dataloader = DataLoader(siamese_dataset,num_workers=10,batch_size=1,shuffle=True)
        dataiter = iter(test_dataloader)
        x0,_,_ = next(dataiter)
        x0
        for i in range(29):
            _,x1,_ = next(dataiter)
            output1,output2 = matchingModel(Variable(x0),Variable(x1))
            euclidean_distance = F.pairwise_distance(output1, output2)
    return str(euclidean_distance.item())


if __name__ == '__main__':
    load_matchingModel()  # load model at the beginning once only
    app.run(host='0.0.0.0', port=80)
