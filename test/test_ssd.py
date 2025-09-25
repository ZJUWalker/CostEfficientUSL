# ###Createaio_handle
# from deepspeed.ops.op_builder import AsyncIOBuilder
# aio_handle=AsyncIOBuilder().load().aio_handle()


#非阻塞式文件写入
import os
path= 'log/test_1.pt'
os.path.isfile(path)

import torch
t=torch.empty(16 * 1024 * 1024,dtype=torch.uint8).cuda()
from deepspeed.ops.op_builder import AsyncIOBuilder
# dict={
#     "block_size": 2 * 1024 * 1024,   # 2MB
#     "queue_depth": 8,                # 其余保持默认
#     "single_submit": False,
#     "overlap_events": True,
#     "thread_count": 1,
# }
h = AsyncIOBuilder().load().aio_handle()

# h=AsyncIOBuilder().load().aio_handle()
h.async_pwrite(t,path)
h.wait()

os.path.isfile(path)

os.path.getsize(path)


# #并行文件写入
# import os
# os.path.isfile('/home/wyl/workspace/CostEfficientUSL/local_nvme/test_2KB.pt')

# import torch
# t=torch.empty(2048,dtype=torch.uint8).cuda()
# from deepspeed.ops.op_builder import AsyncIOBuilder
# h=AsyncIOBuilder().load().aio_handle(intra_op_parallelism=4)  # 4路并行文件写入
# h.async_pwrite(t,'/home/wyl/workspace/CostEfficientUSL/local_nvme/test_2KB.pt')
# h.wait()

# os.path.isfile('/home/wyl/workspace/CostEfficientUSL/local_nvme/test_2KB.pt')

# os.path.getsize('/home/wyl/workspace/CostEfficientUSL/local_nvme/test_2KB.pt')


# # 固定张量
# import os
# os.path.isfile('/home/wyl/workspace/CostEfficientUSL/local_nvme/test_4KB.pt')

# import torch
# t=torch.empty(4096, dtype=torch.uint8).pin_memory()
# from deepspeed.ops.op_builder import AsyncIOBuilder
# h = AsyncIOBuilder().load().aio_handle()
# h.async_pwrite(t,'/home/wyl/workspace/CostEfficientUSL/local_nvme/test_4KB.pt')
# h.wait()

# os.path.isfile('/home/wyl/workspace/CostEfficientUSL/local_nvme/test_4KB.pt')

# os.path.getsize('/home/wyl/workspace/CostEfficientUSL/local_nvme/test_4KB.pt')


# # 并行文件读取
# import os
# os.path.isfile('/home/wyl/workspace/CostEfficientUSL/local_nvme/test_2KB.pt')

# import torch
# t=torch.empty(2048,dtype=torch.uint8).cuda()
# from deepspeed.ops.op_builder import AsyncIOBuilder
# h=AsyncIOBuilder().load().aio_handle(intra_op_parallelism=4)  # 4路并行文件写入
# h.async_pread(t,'/home/wyl/workspace/CostEfficientUSL/local_nvme/test_2KB.pt')
# h.wait()

# os.path.isfile('/home/wyl/workspace/CostEfficientUSL/local_nvme/test_2KB.pt')

# os.path.getsize('/home/wyl/workspace/CostEfficientUSL/local_nvme/test_2KB.pt')
