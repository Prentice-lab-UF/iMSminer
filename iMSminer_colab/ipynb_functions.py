print("success0")
import os
import matplotlib.pyplot as plt
import IPython.display as Disp
from ipywidgets import widgets
import numpy as np
import pandas as pd
import cv2
os.chdir("/content/drive/My Drive/Colab Notebooks/iMSminer_colab")
from assorted_functions import make_image_1c
from google.colab import output
output.enable_custom_widget_manager()
print("success1")
class bbox_select():
    def __init__(self, im, i):
        self.exit = False
        self.im = im
        self.i = i
        self.selected_points = []
        self.fig, self.ax = plt.subplots()
        self.img = self.ax.imshow(self.im.copy())
        self.ka = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.disconnect_button = widgets.Button(description="End selection")
        Disp.display(self.disconnect_button)
        self.disconnect_button.on_click(self.disconnect_mpl)
        plt.show()

    def poly_img(self, img, pts):
        pts = np.array(pts, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)), 2)
        return img

    def onclick(self, event):
        self.selected_points.append([event.xdata, event.ydata])
        if len(self.selected_points) > 1:
            self.img.set_data(self.poly_img(self.im.copy(), self.selected_points))

    def disconnect_mpl(self, _):
        self.fig.canvas.mpl_disconnect(self.ka)
        self.selected_points.append([f"ROI_{self.i}", f"ROI_{self.i}"])
        
print("success2")
def interactive_ROI_selection():
  data_dir = input("Enter the directory path of preprocessed data: ")
  datasets = os.listdir(data_dir)
  datasets = np.asarray(datasets)[(np.char.find(
      datasets, "coords") == -1) & (np.char.find(datasets, "csv") != -1)]

  ROI_edge = pd.DataFrame()

  for rep, dataset in enumerate(datasets):
      print(f"Importing {dataset}.")
      df_build = pd.read_csv(f"{data_dir}/{dataset}")
      print(f"Finished importing {dataset}.")
      
      df_build.rename(columns={'Unnamed: 0': 'mz'}, inplace=True)
      df_build = df_build.T

      mz = df_build.loc['mz']
      df_build.drop('mz', inplace=True)

      df_coords = pd.read_csv(
          f"{data_dir}/{dataset[:-4]}_coords.csv")
      df_coords.rename(columns={'0': 'x', '1': 'y'}, inplace=True)
      df_coords['x'] = df_coords['x'] - np.min(df_coords['x']) + 1
      df_coords['y'] = df_coords['y'] - np.min(df_coords['y']) + 1

      df_build.reset_index(drop=True, inplace=True)
      df_coords.reset_index(drop=True, inplace=True)

      df_build = pd.concat([df_build, df_coords[['x', 'y']]], axis=1)

      img_array_1c = make_image_1c(data_2darray=pd.concat([pd.Series(np.sum(
          df_build.iloc[:, :-2], axis=1)), df_build.iloc[:, -2:]], axis=1).to_numpy())
      
      ROI_num = int(input("How many ROIs are you analyzing? "))
      ROIs = input(
          "How to name your ROIs, from left to right, top to bottom? Separate ROI names by one space.")
      ROIs = ROIs.split(" ")

      for ROI in ROIs:
          ROI_select = bbox_select(img_array_1c[0], rep)

  return ROI_select


            