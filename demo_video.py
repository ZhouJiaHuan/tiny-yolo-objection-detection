""" a demo for testing a video.
author: meringue
date: 2018/6/9
"""
import sys

sys.path.append('./')

import time
from yolo.net.yolo_tiny_net import YoloTinyNet 
import tensorflow as tf 
import cv2
import numpy as np

classes_name = ["stick", "cup", "pen"]

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



def process_predicts(resized_img, predicts, thresh=0.12):
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
  
  p_classes = np.reshape(p_classes, (7, 7, 1, 3))
  C = np.reshape(C, (7, 7, 2, 1))
  
  P = C * p_classes # confidencefor all classes of all bounding boxes (cell_size, cell_size, bounding_box_num, class_num) = (7, 7, 2, 1).
  
  predicts_dict = {}
  for i in range(7):
    for j in range(7):
      temp_data = np.zeros_like(P, np.float32)
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
        
        xcenter = (index[1] + xcenter) * (448/7.0)
        ycenter = (index[0] + ycenter) * (448/7.0)
        
        w = w * 448.0 # ???
        h = h * 448.0
        xmin = 0 if (xcenter - w/2.0 < 0) else (xcenter - w/2.0)
        ymin = 0 if (xcenter - w/2.0 < 0) else (ycenter - h/2.0)
        xmax = resized_img.shape[0] if (xmin + w) > resized_img.shape[0] else (xmin + w)
        ymax = resized_img.shape[1] if (ymin + h) > resized_img.shape[1] else (ymin + h)
        
        class_name = classes_name[class_num]
        predicts_dict.setdefault(class_name, [])
        predicts_dict[class_name].append([int(xmin), int(ymin), int(xmax), int(ymax), P[index]])
        
  return predicts_dict
  

def non_max_suppress(predicts_dict, threshold=0.15):
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
            print np.where(iou<=threshold)
            indexs = np.where(iou<=threshold)[0]
            order = order[indexs+1]
        bbox = bbox_array[keep]
        predicts_dict[object_name] = bbox.tolist()
        predicts_dict = predicts_dict
    return predicts_dict


def plot_result(src_img, predicts_dict):
    """
    plot bounding boxes on source image.
    Args:
        src_img: source image
        predicts_dict: {"stick": [[x1, y1, x2, y2, scores1], [...]]}.
    """
    height_ratio = src_img.shape[0]/448.0
    width_ratio = src_img.shape[1]/448.0
    for object_name, bbox in predicts_dict.items():
        for box in bbox:
            xmin, ymin, xmax, ymax, score = box
            src_xmin = xmin * width_ratio
            src_ymin = ymin * height_ratio
            src_xmax = xmax * width_ratio
            src_ymax = ymax * height_ratio
            score = float("%.3f" %score)

            cv2.rectangle(src_img, (int(src_xmin), int(src_ymin)), (int(src_xmax), int(src_ymax)), (0, 0, 255), 2)
            cv2.putText(src_img, object_name + str(score), (int(src_xmin), int(src_ymin)), 1, 2, (0, 0, 255), thickness=2)

    cv2.imshow("result", src_img)
    return src_img

common_params = {'image_size': 448, 'num_classes': 3, 
                'batch_size':1}
net_params = {'cell_size': 7, 'boxes_per_cell':2, 'weight_decay': 0.0005}

net = YoloTinyNet(common_params, net_params, test=True)

image = tf.placeholder(tf.float32, (1, 448, 448, 3))
predicts = net.inference(image)

sess = tf.Session()
saver = tf.train.Saver(net.trainable_collection)

#saver.restore(sess, 'models/train/model.ckpt-95000')
saver.restore(sess, 'models/train/model.ckpt-100000')


cap = cv2.VideoCapture("test_video16.mp4")
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out_video = cv2.VideoWriter("test_out_video16.mp4", fourcc, 25.0, (640,480))

src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#src_width = 480
#src_height = 480
print('Procession detection...')
frame_num = 0
time1 = time.time()
while (1):
  ret, frame = cap.read()
  #src_img = cv2.imread(img_name)
  #src_height = frame.shape[0]
  #src_width = frame.shape[1]

  """
  M = cv2.getRotationMatrix2D((src_width/2,src_height/2), -90, 1)
  frame = cv2.warpAffine(frame, M, (src_width,src_height))
  """
  if ret == True:
  	  frame_num += 1
	  resized_img = cv2.resize(frame, (448, 448))
	  np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)


	  np_img = np_img.astype(np.float32)

	  np_img = np_img / 255.0 * 2 - 1
	  np_img = np.reshape(np_img, (1, 448, 448, 3))

	  np_predict = sess.run(predicts, feed_dict={image: np_img})
	  predicts_dict = process_predicts(resized_img, np_predict)
	  print "predict dict = ", predicts_dict
	  predicts_dict = non_max_suppress(predicts_dict)
	  print "predict dict = ", predicts_dict


  	  frame_with_bbx = plot_result(frame, predicts_dict)
  	  out_video.write(frame_with_bbx)

  	  if cv2.waitKey(20) & 0xFF==ord("q"):
    		break
  if ret == False:
  	break
time2 = time.time()
speed = frame_num/(time2-time1)
print "speed = ", speed
cap.release()
sess.close()
