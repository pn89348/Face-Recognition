# Import Libraries
from datascience import *
import numpy as np
import matplotlib.pyplot as plots
plots.style.use('fivethirtyeight')

num_images = make_array(10, 50, 100, 200)
data_collection_time = make_array(3, 8, 15, 28)
training_time = make_array(0.33, 1.25, 2.39, 5.34)

analysis = Table().with_columns({"Number of Images": num_images, "Data Collection Time": data_collection_time, "Training Time": training_time})
print(analysis)

analysis.plot("Number of Images")
