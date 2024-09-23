from options import test_options
from model import create_model
from util import visualizer
from PIL import Image
import torchvision.transforms as transforms
import torch
from util import task  # Make sure this import matches your project structure

def load_single_image(image_path, opt):
    """Load and transform a single image."""
    transform = get_transform(opt)
    img_pil = Image.open(image_path).convert('RGB')
    
    # Resize if necessary
    if img_pil.size[0] < 64 or img_pil.size[1] < 64:
        img_pil = img_pil.resize((max(64, img_pil.size[0]), max(64, img_pil.size[1])), Image.BICUBIC)
    
    img = transform(img_pil)
    img_pil.close()
    return img

def get_transform(opt):
    """Basic process to transform PIL image to torch tensor"""
    transform_list = []
    osize = [opt.loadSize[0], opt.loadSize[1]]
    fsize = [opt.fineSize[0], opt.fineSize[1]]
    if opt.isTrain:
        if opt.resize_or_crop == 'resize_and_crop':
            transform_list.append(transforms.Resize(osize))
            transform_list.append(transforms.RandomCrop(fsize))
        elif opt.resize_or_crop == 'crop':
            transform_list.append(transforms.RandomCrop(fsize))
        if not opt.no_augment:
            transform_list.append(transforms.ColorJitter(0.0, 0.0, 0.0, 0.0))
        if not opt.no_flip:
            transform_list.append(transforms.RandomHorizontalFlip())
        if not opt.no_rotation:
            transform_list.append(transforms.RandomRotation(3))
    else:
        transform_list.append(transforms.Resize(fsize))

    transform_list += [transforms.ToTensor()]

    return transforms.Compose(transform_list)

def generate_mask(img, mask_type):
    """Generate a mask for the input image."""
    if mask_type == 0:
        return task.center_mask(img)
    elif mask_type == 1:
        return task.random_regular_mask(img)
    elif mask_type == 2:
        return task.random_irregular_mask(img)
    else:
        raise ValueError("Unsupported mask type")

if __name__=='__main__':
    # get testing options
    opt = test_options.TestOptions().parse()
    # specify the path to the input image
    input_image_path = 'input/input.png'
    
    # load the image
    img = load_single_image(input_image_path, opt)
    img = img.unsqueeze(0)  # add batch dimension

    # generate a mask
    mask_type = 2  # for example, use random irregular mask
    mask = generate_mask(img, mask_type).unsqueeze(0)

    # create a model
    model = create_model(opt)
    model.eval()
    # create a visualizer
    visualizer = visualizer.Visualizer(opt)

    # prepare the data dictionary
    data = {'img': img, 'img_path': input_image_path, 'mask': mask}

    # set the input and test the model
    model.set_input(data)
    model.test()
    
    # visualize or save the result if needed
    visuals = model.get_current_visuals()
    visualizer.display_current_results(visuals, 0, opt)
