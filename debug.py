from transfertools.models import LocIT
from transfertools.models import TCA, CORAL

import torch

TransferName = "tca"

N = 100
source_data = torch.randn([N, 500])
target_data = torch.randn([N, 500])

if TransferName == "coral":
      transfor = CORAL(scaling='standard') # [Transfer Model]
if TransferName == "locit":
      transfor = LocIT()# [Transfer Model]
if TransferName == "tca":
      transfor = TCA(n_components = 10)# [Transfer Model]

outputs = transfor.fit_transfer(source_data, target_data)
Xs_trans, Xt_trans = outputs
print(Xs_trans.shape, Xt_trans.shape)