import torch
from models.erfnet_road import Net
#from models.erfnet import Net
import os
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage
import numpy as np

def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict keys are there
	own_state = model.state_dict()
	for name, param in state_dict.items():
		if name not in own_state:
			print("Not loaded", name)
			continue
		try:
			own_state[name].copy_(param)
			print("copied" + name)
		except Exception as e:
			print("Not copied" + name)
	return model

def main():
	model = Net()
	model = torch.nn.DataParallel(model).cuda()
		
	model = load_my_state_dict(model, torch.load('weights/weights_erfnet_road.pth'))
	#model = load_my_state_dict(model, torch.load('weights/weights_erfnet.pth'))
	#model.load_state_dict(torch.load('weights/weights_erfnet.pth'))
	#model.load_state_dict(torch.load('weights/weights_erfnet_road.pth'))
	model.eval()
	bdd_dir = '/falstaff/imaged2/BDD-100K/images/100k/test/'
	for i, file in enumerate(os.listdir(bdd_dir)):
		print(i)
		im = Image.open(bdd_dir + file)
		input = ToTensor()(im)
		input = input.unsqueeze(0)
		input = F.interpolate(input, scale_factor=0.5, mode='bilinear')
		input = input.cuda()
		output, _ = model(input)
		#output = model(input)
		output = output.max(dim=1)[1]
		output = output.float().unsqueeze(0)
		output = F.interpolate(output, scale_factor=2, mode='nearest')
		output = output.long()		
		output = ToPILImage()(output.squeeze().unsqueeze(0).cpu().byte())
		#output.save('outputs/'+file.replace('.jpg','.png'))
		output.save('outputs_road/'+file.replace('.jpg','.png'))

if __name__ == '__main__':
	main()
