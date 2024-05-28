# -*- coding: utf-8 -*-
import numpy as np

import torch
import torch.distributed as dist

from parameters import get_args
from pcode.master import Master as Master_HOG
from pcode.master_lmf import Master as Master_LMF
from pcode.worker import Worker as Worker_HOG
from pcode.worker_lmf import Worker as Worker_LMF
import pcode.utils.topology as topology
import pcode.utils.checkpoint as checkpoint
import pcode.utils.logging as logging
import pcode.utils.param_parser as param_parser
import time
import os


def main(conf):
    # init the distributed world.
    start_time = time.time()
    try:
        dist.init_process_group("mpi")
    except AttributeError as e:
        print(f"failed to init the distributed world: {e}.")
        conf.distributed = False

    # init the config.
    init_config(conf)

    # start federated learning.
    # init server and clients
    if "semi_supervised" in conf.fl_aggregate["scheme"]:
        process = Master_LMF(conf) if conf.graph.rank == 0 else Worker_LMF(conf)
    else:
        process = Master_HOG(conf) if conf.graph.rank == 0 else Worker_HOG(conf)
    process.run()
    print("end2end time cost:", time.time() - start_time)


def init_config(conf):
    # define the graph for the computation.
    conf.graph = topology.define_graph_topology(
        world=conf.world,
        world_conf=conf.world_conf,
        n_participated=conf.n_participated,
        on_cuda=conf.on_cuda,
    )
    conf.graph.rank = dist.get_rank()

    # init related to randomness on cpu.
    if not conf.same_seed_process:
        conf.manual_seed = 1000 * conf.manual_seed + conf.graph.rank
    conf.random_state = np.random.RandomState(conf.manual_seed)
    torch.manual_seed(conf.manual_seed)

    # configure cuda related.
    if conf.graph.on_cuda:
        assert torch.cuda.is_available()
        torch.cuda.manual_seed(conf.manual_seed)
        torch.cuda.set_device(conf.graph.primary_device)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True if conf.train_fast else False

    # init the model arch info.
    conf.arch_info = (
        param_parser.dict_parser(conf.complex_arch)
        if conf.complex_arch is not None
        else {"master": conf.arch, "worker": conf.arch}
    )
    conf.arch_info["worker"] = conf.arch_info["worker"].split(":")

    # parse the fl_aggregate scheme.
    conf._fl_aggregate = conf.fl_aggregate
    conf.fl_aggregate = (
        param_parser.dict_parser(conf.fl_aggregate) if conf.fl_aggregate is not None else conf.fl_aggregate
    )
    [setattr(conf, f"fl_aggregate_{k}", v) for k, v in conf.fl_aggregate.items()]

    # define checkpoint for logging (for federated learning server).
    checkpoint.init_checkpoint(conf, rank=str(conf.graph.rank))

    # configure logger.
    conf.logger = logging.Logger(conf.checkpoint_dir)

    # display the arguments' info.
    if conf.graph.rank == 0:
        logging.display_args(conf)

    # sync the processes.
    dist.barrier()


if __name__ == "__main__":
    conf = get_args()
    main(conf)



# import os
# import torch.distributed as dist
# import torch.multiprocessing as mp
#
# # -*- coding: utf-8 -*-
# import numpy as np
#
# import torch
#
# from parameters import get_args
# from pcode.master import Master
# from pcode.master import Master as Master_HOG
# from pcode.master_ours_method import Master as Master_Ours
# from pcode.master_lmf import Master as Master_LMF
# from pcode.original_worker import Worker
# from pcode.worker import Worker as Worker_HOG
# from pcode.worker_lmf import Worker as Worker_LMF
# from pcode.worker import Worker
# import pcode.utils.topology as topology
# import pcode.utils.checkpoint as checkpoint
# import pcode.utils.logging as logging
# import pcode.utils.param_parser as param_parser
# import random
#
#
# def main(rank,size,conf,port):
#     # init the distributed world.
#     try:
#         os.environ['MASTER_ADDR'] = '127.0.0.1'
#         os.environ['MASTER_PORT'] = port
#         dist.init_process_group("gloo",rank=rank,world_size=size)
#     except AttributeError as e:
#         print(f"failed to init the distributed world: {e}.")
#         conf.distributed = False
#
#     # init the config.
#     init_config(conf)
#
#     # start federated learning.
#     if "ours_method" in conf.fl_aggregate["scheme"]:
#         process = Master_Ours(conf) if conf.graph.rank == 0 else Worker_HOG(conf)
#     elif "semi_supervised" in conf.fl_aggregate["scheme"]:
#         process = Master_LMF(conf) if conf.graph.rank == 0 else Worker_LMF(conf)
#     else:
#         process = Master_HOG(conf) if conf.graph.rank == 0 else Worker_HOG(conf)
#     process.run()
#
#
# def init_config(conf):
#     # define the graph for the computation.
#     conf.graph = topology.define_graph_topology(
#         world=conf.world,
#         world_conf=conf.world_conf,
#         n_participated=conf.n_participated,
#         on_cuda=conf.on_cuda,
#     )
#     conf.graph.rank = dist.get_rank()
#
#     # init related to randomness on cpu.
#     # if not conf.same_seed_process:
#     #     conf.manual_seed = 1000 * conf.manual_seed + conf.graph.rank
#     os.environ['PYTHONHASHSEED'] = str(conf.manual_seed)
#     random.seed(conf.manual_seed)
#     np.random.seed(conf.manual_seed)
#     conf.random_state = np.random.RandomState(conf.manual_seed)
#     torch.manual_seed(conf.manual_seed)
#     init_cuda(conf)
#
#     # init the model arch info.
#     conf.arch_info = (
#         param_parser.dict_parser(conf.complex_arch)
#         if conf.complex_arch is not None
#         else {"master": conf.arch, "worker": conf.arch}
#     )
#     conf.arch_info["worker"] = conf.arch_info["worker"].split(":")
#
#     # parse the fl_aggregate scheme.
#     conf.fl_aggregate = (
#         param_parser.dict_parser(conf.fl_aggregate)
#         if conf.fl_aggregate is not None
#         else conf.fl_aggregate
#     )
#     [setattr(conf, f"fl_aggregate_{k}", v) for k, v in conf.fl_aggregate.items()]
#
#     # define checkpoint for logging (for federated learning server).
#     checkpoint.init_checkpoint(conf, rank=str(conf.graph.rank))
#
#     # configure logger.
#     conf.logger = logging.Logger(conf.checkpoint_dir)
#
#     # display the arguments' info.
#     if conf.graph.rank == 0:
#         logging.display_args(conf)
#
#     # sync the processes.
#     dist.barrier()
#
# def init_cuda(conf):
#     torch.cuda.set_device(torch.device("cuda:" + str(conf.graph.rank % torch.cuda.device_count())))
#     torch.cuda.manual_seed(conf.manual_seed)
#     torch.cuda.manual_seed_all(conf.manual_seed)
#     torch.backends.cudnn.enabled = True
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
#
# import time
# if __name__ == "__main__":
#     conf = get_args()
#     conf.n_participated = int(conf.n_clients * conf.participation_ratio + 0.5)
#     conf.timestamp = str(int(time.time()))
#     size = conf.n_participated + 1
#     processes = []
#
#     mp.set_start_method("spawn")
#     for rank in range(size):
#         p = mp.Process(target=main, args=(rank, size, conf, conf.port))
#         p.start()
#         processes.append(p)
#
#     for p in processes:
#         p.join()
