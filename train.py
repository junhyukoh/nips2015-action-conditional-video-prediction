from argparse import ArgumentParser
import tempfile
import sys
import caffe
import net as N
from caffe.proto import caffe_pb2 as PB

def create_solver(solver_param, file_name=""):
  if file_name:
    f = open(file_name, 'w')
  else:
    f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
  f.write(str(solver_param))
  f.close()
  solver = caffe.get_solver(f.name)
  return solver

def create_solver_proto(train_net, test_net, lr, prefix, 
      test_iter=300, test_interval=10000,
      max_iter=2e6, snapshot=100000, gpu=0, debug_info=False):
  solver = PB.SolverParameter()
  solver.train_net = train_net
  solver.test_net.extend([test_net])
  solver.test_iter.extend([test_iter])
  solver.test_interval = test_interval
  solver.display = 1000
  solver.max_iter = max_iter
  solver.snapshot = snapshot
  solver.snapshot_prefix = prefix
  solver.snapshot_format = PB.SolverParameter.HDF5
  solver.solver_mode = PB.SolverParameter.GPU
  solver.solver_type = PB.SolverParameter.ADAM
  solver.base_lr = lr
  solver.lr_policy = "fixed"
  solver.average_loss = 10000
  solver.momentum = 0.9
  solver.momentum2 = 0.999
  solver.delta= 1e-08
  solver.debug_info = debug_info
  return solver

def main(model, lr, prefix, weights, snapshot, mean, batch_size,
        test_batch_size, num_act, T, K, num_step, num_iter,
        gpu, debug_info, train_data, test_data, load_to_mem):
  caffe.set_mode_gpu()
  caffe.set_device(gpu[0])
  train_net_file = prefix + '_train.prototxt'
  test_net_file = prefix + '_test.prototxt'
  solver_file_name= prefix + '_solver.prototxt'
  train_net_file, train_proto = N.create_netfile(model, train_data, 
      mean, T, K, batch_size, num_act, num_step=num_step, file_name=train_net_file,
      load_to_mem=load_to_mem)
  test_net_file, test_proto= N.create_netfile(model, test_data, 
      mean, T, K, test_batch_size, num_act, num_step=num_step, file_name=test_net_file,
      load_to_mem=load_to_mem)
  solver_proto = create_solver_proto(train_net_file, test_net_file,
      lr, prefix, max_iter=num_iter, debug_info=debug_info)
  solver = create_solver(solver_proto, file_name=solver_file_name) 
  if snapshot:
    solver.restore(snapshot)
  elif weights:
    solver.net.copy_from(weights)
    solver.test_nets[0].copy_from(weights)
  solver.solve()

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--model", type=int, dest="model",
                      default=1, help="1:CNN 2:RNN")
  parser.add_argument("--lr", type=float, dest="lr",
                      default=0.0001, help="Base learning rate")
  parser.add_argument("--prefix", type=str, dest="prefix",
                      default="", help="Path for results")
  parser.add_argument("--weights", type=str, dest="weights",
                      default="", help="Pre-trained caffemodel")
  parser.add_argument("--mean", type=str, dest="mean",
                      default="mean.binaryproto", help="Mean proto file")
  parser.add_argument("--snapshot", type=str, dest="snapshot",
                      default="", help="Pre-trained solverstate")
  parser.add_argument("--batch_size", type=int, dest="batch_size",
                      default=4, help="Batch size")
  parser.add_argument("--test_batch_size", type=int, dest="test_batch_size",
                      default=30, help="Batch size for test") 
  parser.add_argument("--train_data", type=str, dest="train_data",
                      default="train", help="Directory for training data")
  parser.add_argument("--test_data", type=str, dest="test_data",
                      default="test", help="Directory for test data")
  parser.add_argument("--T", type=int, dest="T",
                      default=21, help="Number of unrolled time steps")
  parser.add_argument("--K", type=int, dest="K",
                      default=11, help="Number of initial frames")
  parser.add_argument("--num_act", type=int, dest="num_act",
                      default=0, help="Number of actions")
  parser.add_argument("--num_step", type=int, dest="num_step",
                      default=1, help="Number of prediction steps")
  parser.add_argument("--num_iter", type=int, dest="num_iter",
                      default=2000000, help="Number of iterations")
  parser.add_argument("--gpu", type=int, nargs='+', dest="gpu", help="GPU device id")
  parser.add_argument("--debug_info", dest="debug_info", action="store_true")
  parser.add_argument("--load_to_mem", dest="load_to_mem", action="store_true")

  parser.set_defaults(debug_info=False, load_to_mem=False)

  args = parser.parse_args()
  main(**vars(args))
