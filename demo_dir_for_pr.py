""" a demo for testing the images in a directory (for computing precision and recall).
author: meringue
date: 2018/6/9
"""
import sys
import os
sys.path.append('./')

import time
from yolo.net.yolo_tiny_net import YoloTinyNet 
import tensorflow as tf 
import cv2
import numpy as np
np.set_printoptions(suppress=True)

classes_name = ["stick", "cup", "pen"]
classes_dict = {"stick": 0, "cup": 1, "pen": 2}

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multi-threading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff



def process_predicts(resized_img, predicts, thresh=0.2):
	"""
	process the predicts of object detection with one image input.
	
	Args:
		resized_img: resized source image.
		predicts: output of the model.
		thresh: thresh of bounding box confidence.
	Return:
		predicts_dict: {"stick": [[x1, y1, x2, y2, scores1], [...]]}.
	"""
	p_classes = predicts[0, :, :, 0:3] # one class.
	C = predicts[0, :, :, 3:5] # two bounding boxes in one cell.
	coordinate = predicts[0, :, :, 5:] # all bounding boxes position.

	height, width, _ = resized_img.shape
	p_classes = np.reshape(p_classes, (7, 7, 1, 3))
	C = np.reshape(C, (7, 7, 2, 1))
	
	P = C * p_classes # confidence for all classes of all bounding boxes (cell_size, cell_size, bounding_box_num, class_num) = (7, 7, 2, 1).
	
	predicts_dict = {}
	for i in range(7):
		for j in range(7):
			temp_data = np.zeros_like(P, np.float)
			temp_data[i, j, :, :] = P[i, j, :, :]
			position = np.argmax(temp_data) # refer to the class num (with maximum confidence) for every bounding box.
			index = np.unravel_index(position, P.shape)
			
			if P[index] > thresh:
				class_num = index[-1]
				coordinate = np.reshape(coordinate, (7, 7, 2, 4)) # (cell_size, cell_size, bbox_num_per_cell, coordinate)[xmin, ymin, xmax, ymax]
				max_coordinate = coordinate[index[0], index[1], index[2], :]
				
				xcenter = max_coordinate[0]
				ycenter = max_coordinate[1]
				w = max_coordinate[2]
				h = max_coordinate[3]
				
				xcenter = (index[1] + xcenter) * (width/7.0)
				ycenter = (index[0] + ycenter) * (height/7.0)
				
				w = w * 448.0 # ???
				h = h * 448.0
				xmin = 0 if (xcenter - w/2.0 < 0) else (xcenter - w/2.0)
				ymin = 0 if (ycenter - h/2.0 < 0) else (ycenter - h/2.0)
				xmax = resized_img.shape[0] if (xmin + w) > resized_img.shape[0] else (xmin + w)
				ymax = resized_img.shape[1] if (ymin + h) > resized_img.shape[1] else (ymin + h)
				
				class_name = classes_name[class_num]
				predicts_dict.setdefault(class_name, [])
				predicts_dict[class_name].append([int(xmin), int(ymin), int(xmax), int(ymax), P[index]])
				
	return predicts_dict
	

def non_max_suppress(predicts_dict, threshold=0.2):
    """
    implement non-maximum supression on predict bounding boxes.
    Args:
        predicts_dict: {"stick": [[x1, y1, x2, y2, scores1], [...]]}.
        threshhold: iou threshold
    Return:
        predicts_dict processed by non-maximum suppression
    """
    for object_name, bbox in predicts_dict.items():
        bbox_array = np.array(bbox, dtype=np.float)
        x1, y1, x2, y2, scores = bbox_array[:,0], bbox_array[:,1], bbox_array[:,2], bbox_array[:,3], bbox_array[:,4]
        areas = (x2-x1+1) * (y2-y1+1)
        #print "areas shape = ", areas.shape
        order = scores.argsort()[::-1]
        #print "order = ", order
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter = np.maximum(0.0, xx2-xx1+1) * np.maximum(0.0, yy2-yy1+1)
            iou = inter/(areas[i]+areas[order[1:]]-inter)
            #print np.where(iou<=threshold)
            indexs = np.where(iou<=threshold)[0]
            order = order[indexs+1]
        bbox = bbox_array[keep]
        predicts_dict[object_name] = bbox.tolist()
        predicts_dict = predicts_dict
    return predicts_dict


def write_predicts_to_txt(img_path, predicts_dict, file_writer):
	"""write the predict results to a txt file.
	Args:
		img_path: image path.
		predicts_dict: {"stick": [[x1, y1, x2, y2, scores1], [...]]}.
		file_writer: open(txt_file, a)
	"""
	row_data = img_path + " "
	for obj_name, bbox in predicts_dict.items():
		cls_num = classes_dict[obj_name]
		for coord in bbox:
			x1, y1, x2, y2 = np.int32(coord[:-1])
			row_data += str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + " " + str(cls_num) + " "
	row_data += "\n"
	file_writer.write(row_data)


def plot_result(src_img, predicts_dict, filename="result.jpg"):
    """
    plot bounding boxes on source image.
    Args:
        src_img: source image
        predicts_dict: {"stick": [[x1, y1, x2, y2, scores1], [...]]}.
    """
    for object_name, bbox in predicts_dict.items():
        for box in bbox:
            xmin, ymin, xmax, ymax, score = box
            score = float("%.2f" %score)

            cv2.rectangle(src_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255))
            cv2.putText(src_img, object_name + str(score), (int(xmin), int(ymin)), 1, 1.5, (0, 0, 255))

    cv2.imshow("result", src_img)
    #cv2.imwrite(filename, src_img)


def scale_coordinate(predicts_dict, src_shape=(480, 640), resized_shape=(448, 448)):
	"""convert the resized box coords to source size coords.
	Args:
		predicts_dict: {"stick": [[x1, y1, x2, y2, scores1], [...]]} on resized image..
		src_shape: source image shape.
		resized_shape: resized image shape.
	Return:
		predict_dict: predict dict on source image.
	"""
	height_ratio = 1.0 * src_shape[0] / resized_shape[0]
	width_ratio = 1.0 * src_shape[1] / resized_shape[1]
	scale_array = np.array([width_ratio, height_ratio, width_ratio, height_ratio, 1.0])
	for object_name in predicts_dict.keys():	
		temp_array = predicts_dict[object_name] * scale_array
		temp_array = np.round(temp_array, 2)
		temp_array[:, 0][temp_array[:, 0]>src_shape[1]] = src_shape[1]
		temp_array[:, 1][temp_array[:, 1]>src_shape[0]] = src_shape[0]
		temp_array[:, 2][temp_array[:, 2]>src_shape[1]] = src_shape[1]
		temp_array[:, 3][temp_array[:, 3]>src_shape[0]] = src_shape[0]
		predicts_dict[object_name] = temp_array
	return predicts_dict
	


if __name__ == '__main__':
	common_params = {'image_size': 448, 'num_classes': 3, 'batch_size': 1}
	net_params = {'cell_size': 7, 'boxes_per_cell': 2, 'weight_decay': 0.0005}
	
	root_dir = "/home/meringue/Documents/stick_cup_pen_detection_tensorflow/tensorflow-yolo-python2.7" #change the root_dir
	os.chdir(root_dir)
    
	test_img_folder = os.path.join(root_dir, "./data/test_image/JPEGImages") #copy the test images to this directory
	#test_label_txt = "./data/test.txt"
	#print "labels dict =  ", labels_dict 
	img_names = os.listdir(test_img_folder)
	img_names = [os.path.join(test_img_folder, img_name) for img_name in img_names if img_name.split(".")[-1] == "jpg"]
	net = YoloTinyNet(common_params, net_params, test=True)

	image = tf.placeholder(tf.float32, (1, 448, 448, 3))
	predicts = net.inference(image)

	sess = tf.Session()
	saver = tf.train.Saver(net.trainable_collection)
	saver.restore(sess, 'models/train20180410/model.ckpt-100000') #choose the trained model

	timer1 = Timer()
	timer1.tic()

	print "Procession detection..."
	predict_txt = "./predict_result.txt"
	if os.path.isfile(predict_txt):
		os.remove(predict_txt)
	f = open(predict_txt, "a")
	img_dex = 0
	for img_name in img_names:
		img_dex += 1
		## read and preprocess input image
		src_img = cv2.imread(img_name)
		resized_img = cv2.resize(src_img, (448, 448))

		np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
		np_img = np_img.astype(np.float)
		np_img = np_img / 255.0 * 2 - 1
		np_img = np.reshape(np_img, (1, 448, 448, 3))

		## prediction process
		#print "current test image = ", os.path.split(img_name)[-1]
		np_predict = sess.run(predicts, feed_dict={image: np_img})

		## process the prediction result
		current_predicts_dict = process_predicts(resized_img, np_predict)
		current_predicts_dict = non_max_suppress(current_predicts_dict)
		current_predicts_dict = scale_coordinate(current_predicts_dict, src_img.shape[:-1])
		#print "current predict dict = ", current_predicts_dict
		write_predicts_to_txt(img_name, current_predicts_dict, f)

		
		if img_dex % 10 == 0:
			print "total image number = ", len(img_names), "current image index = ", img_dex

		## show the detection result
		#plot_result(src_img, current_predicts_dict, str(img_dex)+".jpg")
		#cv2.waitKey(0)
	f.close()
	timer1.toc()
	print('total dedetction time =  {:.3f}s in average'.format(timer1.total_time))
