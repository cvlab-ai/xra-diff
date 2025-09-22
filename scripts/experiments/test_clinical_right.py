import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("WebAgg")

from reconsnet.data.dataset import ClinicalDataset

ds = ClinicalDataset("/home/roagen/media/docs/uck-clean/right")
