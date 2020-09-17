import os
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import argparse
import numpy as np
from missing_data import MissingData
from net import InpaintNet
import cv2
from utils import get_part_mask, get_part_patch_box
from torch.nn import functional as F
def get_loss(img_mask, img_input, img_complete, output):
    # print(img_complete)
    # loss = torch.sum(torch.abs((img_complete - output) * img_mask)) /( torch.sum(img_mask))
    ## change loss
    loss = F.smooth_l1_loss(img_complete[img_mask], output[img_mask])
    return loss


def train(net, epochs, dataloader, optimizer, criterion, start, dataset_size_train, part_id):
    # criterion = torch.nn.L1Loss()
    for epoch in range(start, epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        net.train()
        running_loss = 0
        for img_mask, img_input, img_complete, path in dataloader:
            # print(path)
            img_mask = torch.unsqueeze(img_mask, 1)
            img_mask = img_mask.repeat(1, 3, 1, 1)
            img_mask = img_mask.to(device)
            img_input = img_input.to(device)
            img_input = img_input * (1 - img_mask)
            img_complete = img_complete.to(device)
        
            optimizer.zero_grad()

            output = net(img_input)
            loss = criterion(img_mask, img_input, img_complete, output)
            # loss = criterion(output, img_complete)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # print(loss.item())
        epoch_loss = running_loss # / dataset_size_train
        print('epoch:', epoch, 'epoch_loss', epoch_loss)
        torch.save(net, 'model_res/' + str(part_id) + '/1model' + str(epoch) + '.pth')

def test(eval_model_path, dataloader, img_size):
    net = torch.load(eval_model_path)
    net.eval()
    for img_input, path in dataloader:
        '''
        img_mask1 = torch.unsqueeze(img_mask, 1)
        img_mask1 = img_mask1.repeat(1, 3, 1, 1)
        img_mask1 = img_mask1.to(device)
        '''
        img_input1 = img_input.to(device)
        # img_input1 = img_input1 * (1 - img_mask1)
        output = net(img_input1)

        for ids in range(output.size(0)):
           
            unloader = transforms.ToPILImage()
            '''
            image = img_input1[ids].cpu().clone()
            image = image.squeeze(0)
            image = unloader(image)
            image.save('input.png')

            image = img_complete[ids].cpu().clone()
            image = image.squeeze(0)
            image = unloader(image)
            image.save('complete.png')

            image = img_mask[ids].detach().numpy() * 255
            cv2.imwrite('mask.png', image)
            '''
            # img_res = output[ids].permute(1, 2, 0).cpu().detach().numpy() * 255
            # cv2.imwrite('out.png', img_res)
            trans = transforms.Compose([transforms.Resize((235, 533))])
            image = output[ids].cpu().clone()
            image = image.squeeze(0)
            image = trans(unloader(image))
            inpatinting(image, path['path'][0])

def test_train_set(eval_model_path, dataloader, img_size):
    net = torch.load(eval_model_path)
    net.eval()
    for img_mask, img_input, img_complete, path in dataloader:
        img_mask = torch.unsqueeze(img_mask, 1)
        img_mask = img_mask.repeat(1, 3, 1, 1)
        img_mask = img_mask.to(device)
        img_input = img_input.to(device)
        img_input = img_input * (1 - img_mask)
        # img_input1 = img_input1 * (1 - img_mask1)
        output = net(img_input)

        for ids in range(output.size(0)):
           
            unloader = transforms.ToPILImage()
            '''
            image = img_input1[ids].cpu().clone()
            image = image.squeeze(0)
            image = unloader(image)
            image.save('input.png')

            image = img_complete[ids].cpu().clone()
            image = image.squeeze(0)
            image = unloader(image)
            image.save('complete.png')

            image = img_mask[ids].detach().numpy() * 255
            cv2.imwrite('mask.png', image)
            '''
            # img_res = output[ids].permute(1, 2, 0).cpu().detach().numpy() * 255
            # cv2.imwrite('out.png', img_res)
            
            trans = transforms.Compose([transforms.Resize((235, 533))])
            image = output[ids].cpu().clone()
            image = image.squeeze(0)
            image = trans(unloader(image))
        
            inpatinting2(image, path['path'][0])
            
def inpatinting(img_output, img_raw_path):
    img_output = cv2.cvtColor(np.asarray(img_output), cv2.COLOR_RGB2BGR)
    print(img_output.shape)
    print(img_raw_path)
    img_raw = cv2.imread(img_raw_path)
    part_id = int(img_raw_path.split('/')[1])
    texture_map_path = 'Template18_new.PNG'
    texture_mask_dict = get_part_mask(texture_map_path)
    texture_bbox_dict = get_part_patch_box(texture_map_path)
    texture_part_bbox = texture_bbox_dict[part_id]
    texture_part_mask = texture_mask_dict[part_id][texture_part_bbox[0]:texture_part_bbox[2], texture_part_bbox[1]:texture_part_bbox[3]]
    missing_region = texture_part_mask & (img_raw[:,:,0] == 0) & (img_raw[:,:,1] == 0) & (img_raw[:,:,2] == 0)
    img_raw[:,:,0][missing_region] = img_output[:,:,0][missing_region]
    img_raw[:,:,1][missing_region] = img_output[:,:,1][missing_region]
    img_raw[:,:,2][missing_region] = img_output[:,:,2][missing_region]
    cv2.imwrite(os.path.join('part_inpainting_res', str(part_id), img_raw_path.split('/')[-1]), img_raw)

def inpatinting2(img_output, img_raw_path):
    img_output = cv2.cvtColor(np.asarray(img_output), cv2.COLOR_RGB2BGR)
    print(img_output.shape)
    print(img_raw_path)
    img_raw = cv2.imread(img_raw_path)
    part_id = int(img_raw_path.split('/')[1])
    texture_map_path = 'Template18_new.PNG'
    texture_mask_dict = get_part_mask(texture_map_path)
    texture_bbox_dict = get_part_patch_box(texture_map_path)
    texture_part_bbox = texture_bbox_dict[part_id]
    texture_part_mask = texture_mask_dict[part_id][texture_part_bbox[0]:texture_part_bbox[2], texture_part_bbox[1]:texture_part_bbox[3]]
    missing_region = texture_part_mask & (img_raw[:,:,0] == 255) & (img_raw[:,:,1] == 255) & (img_raw[:,:,2] == 255)
    img_raw[:,:,0][missing_region] = img_output[:,:,0][missing_region]
    img_raw[:,:,1][missing_region] = img_output[:,:,1][missing_region]
    img_raw[:,:,2][missing_region] = img_output[:,:,2][missing_region]
    # cv2.imwrite(os.path.join('part_inpainting_res', str(part_id), img_raw_path.split('/')[-1]), img_raw)
    cv2.imwrite(os.path.join('part_inpainting_res', '5_2', img_raw_path.split('/')[-1]), img_raw)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--part_id', default='5')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--img_size', type=tuple, default=(256, 512))
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--resume', default = None)
    parser.add_argument('--eval_only', action='store_true')
    # parser.add_argument('--eval_model_path', default='model_output/model129.pkl')
    # parser.add_argument('--eval_model_path', default='model_res/8/model59.pkl')
    # parser.add_argument('--eval_model_path', default='model_res/4/model1.pth')
    # parser.add_argument('--eval_model_path', default='model_res/0/model25.pth')
    # parser.add_argument('--eval_model_path', default='model_res/1/model25.pth')
    # parser.add_argument('--eval_model_path', default='model_res/2/model25.pth')
    # parser.add_argument('--eval_model_path', default='model_res/3/model25.pth')
    parser.add_argument('--eval_model_path', default='model_res/5/model999.pth')

    
    opt = parser.parse_args()
    print(opt)

    part_id = opt.part_id
    dataset_train = MissingData('part_train_data', part_id, transform=transforms.Compose([
        transforms.Resize(opt.img_size),
        transforms.ToTensor()]))
    dataset_size_train = len(os.listdir(os.path.join('part_train_data', part_id)))
    dataloader_train = data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=False, drop_last=True, num_workers=4)

    dataset_test = MissingData('part_test_data', part_id, transform=transforms.Compose([
        transforms.Resize(opt.img_size),
        transforms.ToTensor()]), train_flag=False)
    dataloader_test = data.DataLoader(dataset_test, batch_size=1, shuffle=False, drop_last=True, num_workers=4)

    dataset_train_test = MissingData('part_train_data', part_id, transform=transforms.Compose([
        transforms.Resize(opt.img_size),
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    dataloader_train_test = data.DataLoader(dataset_train, batch_size=1, shuffle=False, drop_last=True, num_workers=4)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if opt.eval_only:
        # test(opt.eval_model_path, dataloader_test, opt.img_size)
        test_train_set(opt.eval_model_path, dataloader_train_test, opt.img_size)
    else:
        if opt.resume is not None:
            net = torch.load(opt.resume)
            start = int(opt.resume.split('/')[-1].split('.')[0].split('model')[-1]) + 1
        else:
            net = InpaintNet()
            start = 0
        net = net.to(device)

        optimizer = optim.Adam(net.parameters(), lr=0.00001, betas=(0.5, 0.999))
        epochs = opt.epochs
        criterion = get_loss

        
        train(net, epochs, dataloader_train, optimizer, criterion, start, dataset_size_train, part_id)

