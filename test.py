from ncps.wirings import AutoNCP
from ncps.torch import LTC

wiring = AutoNCP(16,10)

backbone = LTC(21, wiring, batch_first=True)