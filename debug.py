from transfertools.models import LocIT
from transfertools.models import TCA, CORAL

import torch

TransferName = "locit"

N = 50
source_data = torch.randn([N, 1024 * 3])
target_data = torch.randn([N, 1024 * 3])

if TransferName == "coral":
      transfor = CORAL(scaling='standard') # [Transfer Model]
if TransferName == "locit":
      transfor = LocIT(psi=3,transfer_threshold=0, train_selection="edge")# [Transfer Model]
if TransferName == "tca":
      transfor = TCA(n_components = 10)# [Transfer Model]

print(source_data.shape, target_data.shape)
outputs = transfor.fit_transfer(source_data, target_data)
Xs_trans, Xt_trans = outputs
print(Xs_trans.shape, Xt_trans.shape)