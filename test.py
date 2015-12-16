from argparse import ArgumentParser
import tempfile
import sys
import caffe
import net as N
import six
import numpy as np
from collections import OrderedDict
from caffe.proto import caffe_pb2 as PB
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import matplotlib.pyplot as plt
import pylab
import cv2
import os
import subprocess as sp

def tag(i):
  return '-{:0>2d}'.format(i)

def post_process(data, mean, scale):
  t = data.copy().squeeze()
  t /= scale
  t += mean
  t = t.clip(0, 255)
  return t.astype('uint8').squeeze().transpose([1, 0, 2]).transpose([0, 2, 1])

def main(model, weights, K, num_act, num_step, num_iter,
        gpu, data, mean, video):
  font = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans.ttf', 20)
  caffe.set_mode_gpu()
  caffe.set_device(gpu)

  if model == 1:
    data_net_file, net_proto = N.create_netfile(model, 
        data, mean, K + num_step, K, 1, num_act, num_step=num_step, mode='data')
    test_net_file, net_proto = N.create_netfile(model, data, mean, K, K, 
        1, num_act, num_step=1, mode='test')
    data_net = caffe.Net(data_net_file, caffe.TEST)
    test_net = caffe.Net(test_net_file, caffe.TEST)
    test_net.copy_from(weights)
  else:
    data_net_file, net_proto = N.create_netfile(model, 
        data, mean, K + num_step, K, 1, num_act, num_step=num_step, mode='data')
    encoder_net_file, net_proto = N.create_netfile(model, data, mean, K, K, 
        1, num_act, num_step=0, mode='test_encode')
    decoder_net_file, net_proto = N.create_netfile(model, data, mean, 1, 0, 
        1, num_act, num_step=1, mode='test_decode')
    data_net = caffe.Net(data_net_file, caffe.TEST)
    encoder_net = caffe.Net(encoder_net_file, caffe.TEST)
    decoder_net = caffe.Net(decoder_net_file, caffe.TEST)
    decoder_net.copy_from(weights)
    encoder_net.share_with(decoder_net)

  mean_blob = caffe.proto.caffe_pb2.BlobProto()
  mean_bin = open(mean, 'rb').read()
  mean_blob.ParseFromString(mean_bin)
  mean_arr = caffe.io.blobproto_to_array(mean_blob).squeeze()

  if video:
    sp.call(['rm', '-rf', video])
    sp.call(['mkdir', '-p', video])
  
  for i in range(0, num_iter):
    print("iteration " + str(i) + "/" + str(num_iter)) 
    data_net.forward()
    data_blob = data_net.blobs['data'].data
    act_blob = data_net.blobs['act'].data
    if model == 1:
      test_net.blobs['data'].data[:] = data_blob[:, 0:K, :, :, :]
      test_net.blobs['act'].data[:] = act_blob[:, K-1, :]
      net = test_net
    elif model == 2:
      clip_blob = data_net.blobs['clip']
      encoder_net.blobs['data'].data[:] = data_blob[0:K, :, :, :, :]
      encoder_net.blobs['clip'].data[:] = 1
      encoder_net.blobs['clip'].data[0, :] = 0
      encoder_net.blobs['h-00'].data[:] = 0
      encoder_net.blobs['c-00'].data[:] = 0
      encoder_net.forward()
      decoder_net.blobs['h-00'].data[:] = encoder_net.blobs['h'+tag(K)].data[:]
      decoder_net.blobs['c-00'].data[:] = encoder_net.blobs['c'+tag(K)].data[:]
      decoder_net.blobs['clip'].data[:] = 1
      decoder_net.blobs['act'].data[:] = act_blob[K-1, :, :]
      net = decoder_net

    pred_data = np.zeros((3, 210, 160), np.float)
    true_data = np.zeros((3, 210, 160), np.float)
    for step in range(0, num_step):
      net.forward()
      if model == 1:
        pred_data[:] = net.blobs['x_hat'+tag(K+1)].data[:]
        true_data[:] = data_net.blobs['data'].data[:, K+step, :, :, :]
      elif model == 2:
        pred_data[:] = net.blobs['x_hat'+tag(1)].data[:]
        true_data[:] = data_net.blobs['data'].data[K+step, :, :, :, :]
      pred_img = post_process(pred_data, mean_arr, 1./255)
      true_img = post_process(true_data, mean_arr, 1./255)
      
      # display
      show_img = np.hstack((pred_img, true_img))
      top_pad = np.zeros((35, show_img.shape[1], show_img.shape[2]), np.uint8)
      show_img = np.vstack((top_pad, show_img))
      img = Image.fromarray(show_img)
      draw = ImageDraw.Draw(img)
      draw.text((10, 10), 'Step:' + str(step), fill=(255, 255, 255), font=font)
      cv2.imshow('Display', np.array(img))
      key = cv2.waitKey(40)

      if video:
        file_name = video+'/{:0>3d}-{:0>5d}.png'.format(i, step)
        b, g, r = img.split()
        Image.merge("RGB", (r, g, b)).save(file_name)

      if step < num_step - 1:
        if model == 1:
          net.blobs['data'].data[:, 0:K-1, :, :, :] = net.blobs['data'].data[:, 1:K, :, :, :]
          net.blobs['data'].data[:, K-1, :, :, :] = pred_data[:]
          net.blobs['act'].data[:] = act_blob[:, K+step, :]
        elif model == 2:
          net.blobs['h-00'].data[:] = net.blobs['h-01'].data[:]
          net.blobs['c-00'].data[:] = net.blobs['c-01'].data[:]
          net.blobs['data'].data[:] = pred_data[:]
          net.blobs['act'].data[:] = act_blob[K+step, :, :]

  if video:
    sp.call(['ffmpeg', '-pattern_type', 'glob', '-r', '15', '-i', video+'/*.png', '-qscale', '0', video+'.mp4'])

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--model", type=int, dest="model",
                      default=1, help="1:CNN 2:RNN")
  parser.add_argument("--weights", type=str, dest="weights",
                      default="", help="Pre-trained caffemodel")
  parser.add_argument("--data", type=str, dest="data",
                      default="test", help="Test data directory")
  parser.add_argument("--K", type=int, dest="K",
                      default=11, help="Number of initial frames")
  parser.add_argument("--mean", type=str, dest="mean",
                      default="mean.binaryproto", help="Mean file")
  parser.add_argument("--num_act", type=int, dest="num_act",
                      default=0, help="Number of actions")
  parser.add_argument("--num_step", type=int, dest="num_step",
                      default=1, help="Number of steps")
  parser.add_argument("--num_iter", type=int, dest="num_iter",
                      default=30, help="Number of iterations")
  parser.add_argument("--gpu", type=int, dest="gpu",
                      default=0, help="GPU device id")
  parser.add_argument("--video", type=str, dest="video",
                      default="", help="Output video directory")
  args = parser.parse_args()
  main(**vars(args))
