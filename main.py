from models.yolo import *
from models.ssd import *
from util import *
from collections import defaultdict
import argparse
import time
import pickle as pkl
import random
import pdb
import os.path as osp

#Path vars
dirname = os.path.dirname(__file__)
CLASS_NAMES_PATH = os.path.join(dirname, 'data/image/coco/coco.names')
TEST_IMG_PATH =  os.path.join(dirname, 'data/image/samples/dog-cycle-car.png')
PALETTE_PATH = os.path.join(dirname, 'misc/pallete')

def arg_parse():
    """
    Parse arguements to the detect module
    """
    parser = argparse.ArgumentParser(description='Object Detection Module')
    parser.add_argument("--images", dest = 'images', help =
                        "Image / Directory containing images to perform detection upon",
                        default = "data/image/samples", type = str)
    parser.add_argument("--video", dest = 'video', help =
                        "Video / Directory containing videos to perform detection upon",
                        default = "data/video/samples", type = str)
    parser.add_argument("--det", dest = 'det', help =
                        "Image / Directory to store detections to",
                        default = "detections", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--arch", dest = 'arch', help =
                        "Config file for network architecture",
                        default = "configs/archs/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help =
                        "weightsfile",
                        default = "configs/weights/yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help =
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "608", type = str)
    return parser.parse_args()

#parse arguments
args = arg_parse()
images = os.path.join(dirname, args.images)
video = os.path.join(dirname, args.video)
det = os.path.join(dirname, args.det)
model_name = 'yolov3'
arch = args.arch
weights = os.path.join(dirname, args.weightsfile)
batch_size = int(args.bs)
confidence = float(args.confidence)
nms = float(args.nms_thresh)

#setup variables
start = 0
CUDA = torch.cuda.is_available()
num_classes = 80
classes = load_classes(CLASS_NAMES_PATH)

print(model_name)

def init_network():
    """
    Creates the model to be used for detection

    Returns:
       model [pytorch nn.Module] -- pytorch model
       inp_dim [int] -- the input dimension for images in this network
       ]
    """

    print ("Network is loading.......")

    if model_name == 'yolov3':
        model = Darknet(arch)
        if CUDA: model = model.cuda()
        model.load_weights(weights)
        print("Network is loaded.")
        model.net_info["height"] = args.reso
        inp_dim = int(model.net_info["height"])
        assert inp_dim % 32 == 0
        assert inp_dim > 32

    elif model_name == "ssd":
        model = VGG16()
        if CUDA: model = model.cuda()
        print("Network is loaded.")

    elif model_name == "frcnn":
        model = FRCNN()
        if CUDA: model = model.cuda()
        print("Network is loaded.")


    return model, inp_dim

def read_image_dir():
    print ("Reading images directory.......")
    try:
        imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
    except NotADirectoryError:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), images))
    except FileNotFoundError:
        print ("No file or directory with the name {}".format(images))
        exit()
    print ("Read images directory.")
    return imlist

def create_image_batches(imlist):
    print("Creating image batches.......")
    #load read images
    loaded_images = [cv2.imread(x) for x in imlist]
    #process images
    img_batches = list(map(process_image, loaded_images,
                    [inp_dim for x in range(len(imlist))]))

    #list with og dimensions
    og_img_dims = [(x.shape[1], x.shape[0]) for x in loaded_images]

    #convert img dims list to pytorch tensor, such that we repeat in dir of axis=1
    og_img_dims = torch.FloatTensor(og_img_dims).repeat(1,2)
    if CUDA: og_img_dims = og_img_dims.cuda()
    leftover = 0

    #if we cant divide images into batches evenly
    if (len(og_img_dims) % batch_size):
        leftover = 1

    num_batches = 1 #holder
    typ_len = len(og_img_dims)

    #if there is more than one image (Create batches)
    if batch_size != 1:
        num_batches = len(imlist) // batch_size + leftover
        img_batches = [torch.cat((img_batches[i*batch_size : min((i+1)*batch_size,
                    len(img_batches))])) for i in range(num_batches)]
        typ_len = len(img_batches[0])

    print(f"Batches created: Batch size:{num_batches}, each of avg length: {typ_len}")
    return og_img_dims, loaded_images, img_batches

def image_detection(model, inp_dim, img_batches, imlist, ):
    """
    Performs image detection on a single or dataset of images
    """
    #set model in eval mode
    model.eval()
    write = 0
    batch_times = []
    for i, batch in enumerate(img_batches):
        #checkpoint time
        start_det = time.time()
        if CUDA: batch = batch.cuda()

        with torch.no_grad():
            pred = model(Variable(batch), CUDA)
            result_process_start = time.time()
            pred = process_results(pred, confidence, num_classes, nms_conf=nms)
            result_process_end = time.time()
        #checkpoint time and add checkpoints to list
        end_det = time.time()
        batch_times += [start_det, end_det]

        #if there is no detections
        if type(pred)==int:
            for img_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
                img_id = i*batch_size + img_num
                print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end_det - start_det)/batch_size))
                print("{0:20s} {1:s}".format("Objects Detected:", ""))
                print("----------------------------------------------------------")
            continue#skip rest of loop

        #match prediction indexes to imlist indexes
        pred[:,0] += i*batch_size

        if not write:
            output = pred
            write = 1

        else:
            output = torch.cat((output, pred))

        for img_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
            img_id = i*batch_size + img_num
            objs = [classes[int(x[-1])] for x in output if int(x[0]) == img_id]
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end_det - start_det)/batch_size))
            print("{:25s}: {:2.3f}".format("Output Processing", result_process_end - result_process_start))
            print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
            print("----------------------------------------------------------")

        if CUDA:
            #ensures cuda kernel synced to CPU
            #for accurate time (gpu completion time equals cpu return)
            torch.cuda.synchronize()

    #check if dets were made
    try:
        output
        return output, pred, batch_times
    except:
        print("no detections were made")
        exit()

def draw_frame_boxes(output, frame, show=True):
    c1 = tuple(output[1:3].int())
    c2 = tuple(output[3:5].int())
    clas = int(output[-1])
    label = "{0}".format(classes[clas])
    cv2.rectangle(frame, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(frame, c1, c2,color, -1)
    cv2.putText(frame, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return frame

def draw_boxes(inp_dim, imlsit, img_batches, output, loaded_images, og_img_dims,show=True):
    #keep the og dims for images we have detections for
    og_img_dims = torch.index_select(og_img_dims, 0, output[:,0].long())

    #determine what scaling factor was to put into network
    scaling_factor = torch.min(inp_dim/og_img_dims, 1)[0].view(-1,1)

    #update our coordinates to match the boundaries of the padded image
    output[:,[1,3]] -= (inp_dim - scaling_factor*og_img_dims[:,0].view(-1,1))/2
    output[:,[2,4]] -= (inp_dim - scaling_factor*og_img_dims[:,1].view(-1,1))/2

    #undo the scaling done on images
    output[:,1:5] /= scaling_factor

    #clip bboxes with boundaries outside image to edges of image (due to padding)
    for i in range(output.shape[0]):
        output[i, [1,3]] = torch.clamp(output[i,[1,3]], 0.0, og_img_dims[i,0])
        output[i,[2,4]] = torch.clamp(output[i,[2,4]], 0.0, og_img_dims[i,1])

    #load color palette
    output_recast = time.time()
    class_load = time.time()
    colors = pkl.load(open(PALETTE_PATH, "rb"))

    img_detections = defaultdict(list)
    #performed for each detection
    for x in output:
        #choose random color to draw
        color = random.choice(colors)
        #create tuples for box coords
        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        #retrieve image based on the id
        img_ind = int(x[0])
        img_id = imlsit[img_ind]
        img = loaded_images[img_ind]
        #show retrieved image
        if show:
            cv2.imshow('image',img)
            cv2.waitKey(0)
        clas = int(x[-1])
        label = "{0}".format(classes[clas])
        cv2.rectangle(img, c1, c2, color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1]+t_size[1]+4
        cv2.rectangle(img, c1, c2, color, -1) #filled rectangle
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
        #shows final image
        if show:
            cv2.imshow('image', img)
            cv2.waitKey(0)
        img_detections[img_id] += [img]
    return img_detections

def save_image_dets(img_dets, det_dir):
    for img in img_dets.keys():
        det_img = img_dets[img][-1]
        if not os.path.exists(det_dir):
            os.makedirs(det_dir)
        img = img.split('\\')[-1]
        fname = img[:-4] + "_" + model_name + ".jpg"
        cv2.imwrite(os.path.join(det, fname), det_img)

def print_summary(time_stash, imlist):
    print("SUMMARY")
    print("----------------------------------------------------------")
    print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
    print()
    print("{:25s}: {:2.3f}".format("Loading Network", time_stash['end_net_load'] -  time_stash['start_net_load']))
    print("{:25s}: {:2.3f}".format("Reading Images Directory", time_stash['end_dir_read'] -  time_stash['start_dir_read']))
    print("{:25s}: {:2.3f}".format("Loading batch", time_stash['end_batch_create'] - time_stash['start_batch_create']))
    print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) +  " images)", time_stash['end_det_loop'] - time_stash['start_det_loop']))
    print("{:25s}: {:2.3f}".format("Drawing Boxes",  time_stash['end_box_draw'] -  time_stash['start_box_draw']))
    print("{:25s}: {:2.3f}".format("Average time_per_img",(time_stash['end_det_loop'] - time_stash['start_det_loop'])/len(imlist)))
    print("----------------------------------------------------------")

IMAGE_MODE = True
VIDEO_MODE = False
Live = False

if IMAGE_MODE:
    time_stash = {}
    #network init
    time_stash['start_net_load'] = time.time()
    model, inp_dim = init_network()
    time_stash['end_net_load'] = time.time()

   #read images dir
    time_stash['start_dir_read'] = time.time()
    image_names_list = read_image_dir()
    time_stash['end_dir_read'] = time.time()

   #create minibatches from read images
    time_stash['start_batch_create'] = time.time()
    og_img_dims, loaded_images, img_batches = create_image_batches(image_names_list)
    time_stash['end_batch_create'] = time.time()

    #begin detection loop
    time_stash['start_det_loop'] = time.time()
    output, pred, batch_times = image_detection(model, inp_dim, img_batches, image_names_list)
    time_stash['end_det_loop'] = time.time()

    #transform boxes to original images and draw them
    time_stash['start_box_draw'] = time.time()
    img_dets = draw_boxes(inp_dim, image_names_list, img_batches, output, loaded_images, og_img_dims, show=False)
    time_stash['end_box_draw']= time.time()

    #save image detections
    time_stash['start_saving'] = time.time()
    save_image_dets(img_dets, det)
    time_stash['end_saving'] = time.time()

    #print detection summary times
    print_summary(time_stash, image_names_list)

    torch.cuda.empty_cache()

elif VIDEO_MODE:
    if LIVE:
        cap = cv2.VideoCapture(0)
        assert cap.isOpened(), 'Cannot capture source'
        frames = 0
        start_vid_record = time.time()

        while cap.isOpened():
            returned, frame = cap.read()

            if returned:
                img = process_image(frame, inp_dim)
                #cv2.imshow("frame", frame)
               #original dimensions
                im_dim = frame.shape[1], frame.shape[0]
                im_dim = torch.FloatTensor(im_dim).repeat(1,2)

                #gpu acceleration
                if CUDA:
                    im_dim = im_dim.cuda()
                    img = img.cuda()

                with torch.no_grad():
                    output = model(Variable(img), CUDA)
                    output = process_results(output, confidence, num_classes, nms_conf = nms)

                #if no dets
                if type(output) == int:
                    #iterate
                    frames += 1
                    print("FPS of the video is {:5.4f}".format( frames / (time.time() - start_vid_record)))
                    cv2.imshow("frame", frame)
                    cv2.waitKey(1)
                    if key & 0xFF == ord('q'):
                        break
                    continue

                #else if there are detections
                #make sure bboxes dont go off image and rescale
                output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(inp_dim))
                im_dim = im_dim.repeat(output.size(0), 1)/inp_dim
                output[:,1:5] *= im_dim
                classes = load_classes(CLASS_NAMES_PATH)
                colors = pkl.load(open(PALETTE_PATH, "rb"))


                frame =  draw_frame_boxes(output, frame)

                #show dets
                cv2.imshow("frame", frame)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break

                frames += 1
                print(time.time() - start_vid_record)
                print("FPS of the video is {:5.2f}".format( frames / (time.time() - start_vid_record)))
            else:
                break

