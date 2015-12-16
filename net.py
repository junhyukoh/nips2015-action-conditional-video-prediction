import tempfile
from collections import OrderedDict
from caffe import layers as L
from caffe import params as P
import caffe

def tag(i):
  return '-{:0>2d}'.format(i)

def atari_data(source, mean, batch_size, num_act, num_input, streaming=False, 
               load_to_mem=False, out_clip=False, out_act=False):
  ntop = 1;
  atari_data_param = dict(num_frame=num_input, num_act=num_act,
            channels=3, streaming=streaming,
            out_clip=out_clip, out_act=out_act,
            load_to_memory=load_to_mem)
  if out_act:
    ntop += 1
  if out_clip:
    ntop += 1

  return L.AtariData(
          data_param=dict(source=source, batch_size=batch_size),
          transform_param=dict(scale=1./255, mean_file=mean),
          atari_data_param=atari_data_param, ntop=ntop)

def conv(bottom, ks_h, ks_w, nout, stride, pad_h, pad_w, param_name=''):
  return L.Convolution(bottom, kernel_h=ks_h, kernel_w=ks_w, 
                      stride=stride, num_output=nout, 
                      pad_h=pad_h, pad_w=pad_w,
                      weight_filler=dict(type='xavier'),
                      param=[dict(name=param_name+'-w', lr_mult=1),
                            dict(name=param_name+'-b', lr_mult=2, decay_mult=0)])

def deconv(bottom, ks_h, ks_w, nout, stride, pad_h, pad_w, param_name=''):
  return L.Deconvolution(bottom, convolution_param=dict(
                      kernel_h=ks_h, kernel_w=ks_w, 
                      stride=stride, num_output=nout, 
                      pad_h=pad_h, pad_w=pad_w,
                      weight_filler=dict(type='xavier')),
                      param=[dict(name=param_name+'-w', lr_mult=1),
                            dict(name=param_name+'-b', lr_mult=2, decay_mult=0)])

def reshape(bottom, c, h, w):
  return L.Reshape(bottom, shape=dict(dim=[-1, c, h, w]))

def fc(bottom, nout, weight_filler={}, bias_filler={}, param_name='', axis=1, bias=True): 
  inner_param = dict(num_output=nout, axis=axis, bias_term=bias)
  if weight_filler:
    inner_param['weight_filler'] = weight_filler
  else:
    inner_param['weight_filler'] = dict(type='xavier')
  if bias_filler and bias:
    inner_param['bias_filler'] = bias_filler
  if not param_name:
    if bias:
      param = [dict(lr_mult=1), dict(lr_mult=2, decay_mult=0)]
    else:
      param = [dict(lr_mult=1)]
  else:
    if bias:
      param=[dict(lr_mult=1, name=param_name+'-w'),dict(lr_mult=2, name=param_name+'-b', decay_mult=0)]
    else:
      param=[dict(lr_mult=1, name=param_name+'-w')]
  return L.InnerProduct(bottom, inner_product_param=inner_param, param=param)

def relu(bottom):
  return L.ReLU(bottom, in_place=True)

def eltwise(bottom1, bottom2, op, coeff_blob=False):
  return L.Eltwise(bottom1, bottom2, operation=op, coeff_blob=coeff_blob)

def euclidean_loss(bottom, label):
  return L.EuclideanLoss(bottom, label)

def flatten(bottom, bottom2):
  return L.Flatten(bottom, bottom2, axis=1, end_axis=2, ntop=2)

def lstm(bottom, clip, nout, param_name=''):
  recurrent_param = dict(num_output=nout,
                  weight_filler=dict(type='uniform', min=-0.08, max=0.08))
  return L.LSTM(bottom, clip, recurrent_param=recurrent_param,
                        param=[dict(name=param_name+'-w'),
                        dict(name=param_name+'-u'),
                        dict(name=param_name+'-b', decay_mult=0)])
  
def add_conv_enc(n, bottom, tag=''):
  n.tops['conv1'+tag] = conv(bottom, 8, 8, 64, 2, 0, 1, 'conv1')
  n.tops['relu1'+tag] = relu(n.tops['conv1'+tag]) 
  n.tops['conv2'+tag] = conv(n.tops['relu1'+tag], 6, 6, 128, 2, 1, 1, 'conv2')
  n.tops['relu2'+tag] = relu(n.tops['conv2'+tag])
  n.tops['conv3'+tag] = conv(n.tops['relu2'+tag], 6, 6, 128, 2, 1, 1, 'conv3')
  n.tops['relu3'+tag] = relu(n.tops['conv3'+tag])
  n.tops['conv4'+tag] = conv(n.tops['relu3'+tag], 4, 4, 128, 2, 0, 0, 'conv4')
  n.tops['relu4'+tag] = relu(n.tops['conv4'+tag])
  n.tops['ip1'+tag]   = fc(n.tops['relu4'+tag], 2048, param_name='ip1')
  n.tops['relu5'+tag] = relu(n.tops['ip1'+tag])
  return n.tops['relu5'+tag]

def add_transform(n, bottom, act, tag=''):
  n.tops['enc-factor'+tag] = fc(bottom, 2048, 
                              weight_filler=dict(type='uniform', min=-1, max=1), 
                              param_name='enc')
  n.tops['act-embed'+tag] = fc(act, 2048, 
                              weight_filler=dict(type='uniform', min=-0.1, max=0.1),
                              param_name='act-embed')
  n.tops['dec-factor'+tag] = eltwise(n.tops['enc-factor'+tag], 
                              n.tops['act-embed'+tag],
                              P.Eltwise.PROD)
  n.tops['dec'+tag] = fc(n.tops['dec-factor'+tag], 2048, param_name='dec')
  return n.tops['dec'+tag]

def add_deconv(n, bottom, tag=''):
  n.tops['ip4'+tag] = fc(bottom, 11264, param_name='dec-reshape')
  n.tops['relu6'+tag] = relu(n.tops['ip4'+tag])
  n.tops['reshape'+tag] = reshape(n.tops['relu6'+tag], 128, 11, 8)
  n.tops['deconv4'+tag] = deconv(n.tops['reshape'+tag], 4, 4, 128, 2, 0, 0, 'deconv4')
  n.tops['relu7'+tag] = relu(n.tops['deconv4'+tag])
  n.tops['deconv3'+tag] = deconv(n.tops['relu7'+tag], 6, 6, 128, 2, 1, 1, 'deconv3')
  n.tops['relu8'+tag] = relu(n.tops['deconv3'+tag])
  n.tops['deconv2'+tag] = deconv(n.tops['relu8'+tag], 6, 6, 128, 2, 1, 1, 'deconv2')
  n.tops['relu9'+tag] = relu(n.tops['deconv2'+tag])
  n.tops['deconv1'+tag] = deconv(n.tops['relu9'+tag], 8, 8, 3, 2, 0, 1, 'deconv1')
  return n.tops['deconv1'+tag]

def add_lstm_init(n, batch_size, lstm_dim):
  n.tops['h-00'] = caffe.net_spec.Input('h-00', [1, batch_size, lstm_dim])
  n.tops['c-00'] = caffe.net_spec.Input('c-00', [1, batch_size, lstm_dim])

def add_lstm_encoder(n, bottom, batch_size, lstm_dim, flatten=True, t='', tag=''):
  bot = bottom
  if flatten:
    n.tops['data_flat'+tag] = L.Flatten(bottom, axis=0, end_axis=1)
    bot = n.tops['data_flat'+tag]
  top = add_conv_enc(n, bot, tag)
  n.tops['x_reshape'+tag] = L.Reshape(top, shape=dict(dim=[-1, batch_size, 2048]))
  n.tops['x_gate'+t] = fc(n.tops['x_reshape'+tag], 4 * lstm_dim, 
      weight_filler=dict(type='uniform', min=-0.08, max=0.08), param_name='Wx', axis=2)
  return n.tops['x_gate'+t]

def add_decoder(n, bottom, act=None, flatten=True, tag=''):
  h = bottom
  if flatten:
    n.tops['h_flat'+tag] = L.Flatten(bottom, axis=0, end_axis=1)
    h = n.tops['h_flat'+tag]
  if act:
    a = act
    if flatten:
      n.tops['act_flat'+tag] = L.Flatten(act, axis=0, end_axis=1)
      a = n.tops['act_flat'+tag]
    top = add_transform(n, h, a, tag)
  else:
    top = h
  return add_deconv(n, top, tag)

def add_cnn(n, data, act, batch_size, T, K, num_step, mode='train'):
  n.x_flat = L.Flatten(data, axis=1, end_axis=2)
  n.act_flat = L.Flatten(act, axis=1, end_axis=2)
  if mode == 'train':
    x = L.Slice(n.x_flat, axis=1, ntop=T)
    act_slice = L.Slice(n.act_flat, axis=1, ntop=T-1)
    x_set = ()
    label_set = ()
    x_hat_set = ()
    silence_set = ()
    for i in range(T):
      t = tag(i+1)
      n.tops['x'+t] = x[i]
      if i < K:
        x_set += (x[i],)
      if i < T - 1:
        n.tops['act'+t] = act_slice[i]
      if i < K - 1:
        silence_set += (n.tops['act'+t],)
      if i >= K:
        label_set += (x[i],)
    n.label = L.Concat(*label_set, axis=0)
    input_list = list(x_set)
    for step in range(0, num_step):
      step_tag = tag(step + 1) if step > 0 else ''
      t = tag(step + K)
      tp = tag(step + K + 1)
      input_tuple = tuple(input_list)
      n.tops['input'+step_tag] = L.Concat(*input_tuple, axis=1)
      top = add_conv_enc(n, n.tops['input'+step_tag], tag=step_tag)
      n.tops['x_hat'+tp] = add_decoder(n, top, n.tops['act'+t], flatten=False, 
          tag=step_tag)
      input_list.pop(0)
      input_list.append(n.tops['x_hat'+tp]) 
  else:
    top = add_conv_enc(n, n.x_flat)
    n.tops['x_hat'+tag(K+1)] = add_decoder(n, top, n.act_flat, flatten=False)
  if mode == 'train':
    x_hat = ()
    for i in range(K, T):
      t = tag(i+1)
      x_hat += (n.tops['x_hat'+t],)
    n.x_hat = L.Concat(*x_hat, axis=0)
    n.silence = L.Silence(*silence_set, ntop=0)
    n.l2_loss = L.EuclideanLoss(n.x_hat, n.label)
  return n

def cnn_proto(source, mean, batch_size, T, K, num_act, num_step, 
    load_to_mem=False, mode='train'):
  n = caffe.NetSpec()
  if mode == 'train' or mode == 'data':
    n.data, n.act = atari_data(source, mean, batch_size, num_act, T, 
        out_act=True, load_to_mem=load_to_mem)
  else:
    n.data = caffe.net_spec.Input('data', [batch_size, T, 3, 210, 160])
    n.act = caffe.net_spec.Input('act', [batch_size, 1, num_act])
  if mode is not 'data':
    add_cnn(n, n.data, n.act, batch_size, T, K, num_step,
        mode=mode)
  return n.to_proto()

def add_rnn(n, data, act, clip, batch_size, T, K, num_step, lstm_dim=2048, 
    mode='train'):
  add_lstm_init(n, batch_size, lstm_dim)
  n.clip_reshape = L.Reshape(clip, shape=dict(dim=[1, T, batch_size]))
  if mode is 'train' or mode is 'test_encode':
    clip_slice = L.Slice(n.clip_reshape, ntop=T, axis=1)
    if mode == 'train':
      act_slice = L.Slice(act, ntop=T-1, axis=0)
      x = L.Slice(data, axis=0, ntop=T)
      x_set = ()
      label_set = ()
      silence_set = ()
    for i in range(T):
      t = tag(i+1)
      n.tops['clip'+t] = clip_slice[i]
      if mode == 'train':
        n.tops['x'+t] = x[i]
        if i < T - 1:
          n.tops['act'+t] = act_slice[i]
        if i < T-num_step:
          x_set = x_set + (x[i],)
        if i < K-1:
          silence_set += (act_slice[i], )
        if i >= K:
          label_set = label_set + (x[i],)
    if mode == 'train':
      n.x = L.Concat(*x_set, axis=0)
      n.label = L.Concat(*label_set, axis=0)
      add_lstm_encoder(n, n.x, batch_size, lstm_dim) 
    else:
      add_lstm_encoder(n, data, batch_size, lstm_dim) 
  if T > num_step:
    x_gate = L.Slice(n.x_gate, ntop=T-num_step, axis=0)
    if type(x_gate) is caffe.net_spec.Top:
      x_gate = (x_gate,)
  else:
    x_gate = ()

  for i in range(0, T):
    t_1 = tag(i)
    t = tag(i+1)
    
    clip_t = n.tops['clip'+t] if mode == 'train' or mode == 'test_encode' else n.clip_reshape
    n.tops['h_conted'+t_1] = eltwise(n.tops['h'+t_1], clip_t, 
        P.Eltwise.SUM, True)
    # Decoding
    if i == T - num_step:
      if mode == 'train':
        h_set = ()
        act_set = ()
        for j in range(K, T - num_step + 1):
          t_j = tag(j)
          h_set = h_set + (n.tops['h_conted'+t_j],)
          act_set = act_set + (n.tops['act'+t_j],)
        n.h = L.Concat(*h_set, axis=0)
        n.act_concat = L.Concat(*act_set, axis=0)
        top = add_decoder(n, n.h, n.act_concat)
      else:
        top = add_decoder(n, n.tops['h_conted'+t_1], act)
      x_outs = L.Slice(top, axis=0, ntop=T-num_step-K+1)
      if type(x_outs) is caffe.net_spec.Top:
        x_outs = [x_outs]
      for j in range(K, T - num_step+1):
        n.tops['x_hat'+tag(j+1)] = x_outs[j-K]
      dec_tag = tag(2) if mode == 'train' else ''
      if mode == 'test_decode':
        add_lstm_encoder(n, n.tops['x_hat'+t], batch_size, lstm_dim=lstm_dim, flatten=False) 
        x_gate = x_gate + (n.tops['x_gate'],)
      elif num_step > 1:
        add_lstm_encoder(n, n.tops['x_hat'+t], batch_size, lstm_dim=lstm_dim, t=t, 
            tag=dec_tag, flatten=False) 
        x_gate = x_gate + (n.tops['x_gate'+t],)

    if i > T - num_step: 
      dec_t = tag(i-T+num_step+1)
      dec_tp = tag(i-T+num_step+2)
      top = add_decoder(n, n.tops['h_conted'+t_1], n.tops['act'+t_1], tag=dec_t)
      n.tops['x_hat'+t] = top
      if i < T-1:
        add_lstm_encoder(n, n.tops['x_hat'+t], batch_size, lstm_dim=lstm_dim, t=t, 
            tag=dec_tp, flatten=False) 
        x_gate = x_gate + (n.tops['x_gate'+t],)

    if i < T-1 or mode is not 'train':
      # H-1 to H
      if mode is not 'test_decode':
        n.tops['x_gate'+t] = x_gate[i] 
      n.tops['h_gate'+t] = fc(n.tops['h_conted'+t_1], 4 * lstm_dim,
          weight_filler=dict(type='uniform', min=-0.08, max=0.08), param_name='Wh', 
          axis=2, bias=False)
      n.tops['gate'+t] = eltwise(x_gate[i], n.tops['h_gate'+t], P.Eltwise.SUM)
      n.tops['c'+t], n.tops['h'+t] = L.LSTMUnit(n.tops['c'+t_1], 
          n.tops['gate'+t], clip_t, ntop=2, clip_gradients=[0, 0.1, 0])

  # Define Loss functions
  if mode == 'train':
    x_hat = ()
    for i in range(K, T):
      t = tag(i+1)
      x_hat = x_hat + (n.tops['x_hat'+t],)
    silence_set += (n.tops['c'+tag(T-1)],)
    n.silence = L.Silence(*silence_set, ntop=0)
    n.x_hat = L.Concat(*x_hat, axis=0)
    n.label_flat = L.Flatten(n.label, axis=0, end_axis=1)
    n.l2_loss = L.EuclideanLoss(n.x_hat, n.label_flat)
  return n

def rnn_proto(source, mean, batch_size, T, K, num_act, num_step, 
    load_to_mem=False, mode='train'):
  n = caffe.NetSpec()
  if mode == 'train' or mode == 'data':
    n.data, n.act, n.clip = atari_data(source, mean, batch_size, num_act, T, 
        streaming=True, out_act=True, out_clip=True, load_to_mem=load_to_mem)
  else:
    n.data = caffe.net_spec.Input('data', [T, batch_size, 3, 210, 160])
    n.clip = caffe.net_spec.Input('clip', [T, batch_size]) 
    n.act = caffe.net_spec.Input('act', [T, batch_size, num_act])
  if mode is not 'data':
    add_rnn(n, n.data, n.act, n.clip, batch_size, T, K, num_step, mode=mode)
  return n.to_proto()

def create_netfile(model_type, source, mean, T, K, batch_size, num_act, 
                 num_step=1, file_name='', load_to_mem=False,
                 mode='train'):
  if model_type == 1: # cnn
    n = cnn_proto(source, mean, batch_size, T, K, num_act, num_step, 
        load_to_mem=load_to_mem, mode=mode)
  elif model_type == 2: # rnn
    n = rnn_proto(source, mean, batch_size, T, K, num_act, num_step, 
        load_to_mem=load_to_mem, mode=mode)
  else:
    assert false

  if file_name:
    f = open(file_name, 'w')
    f.write(str(n))
    f.close()
    return file_name, n
  else:
    f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    f.write(str(n))
    f.close()
    return f.name, n
