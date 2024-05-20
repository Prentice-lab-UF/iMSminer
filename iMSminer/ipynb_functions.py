import matplotlib.pyplot as plt
import IPython.display as Disp
from ipywidgets import widgets
import numpy as np
import cv2

class bbox_select():
    """Interactive ROI selection for ipynb IPython Notebooks. Codes from https://gist.github.com/Pked01/83cdef1dfe49e4004f5af78708767850#file-draw_bbox-ipynb
    """
    def __init__(self, im):
        self.move = False
        self.im = im
        self.selected_points = []
        self.img = plt.imshow(self.im)
        self.ka = self.canvas.mpl_connect('button_press_event', self.onclick)
        plt.show(block=False)
        disconnect_button = widgets.Button(description="Disconnect mpl")
        Disp.display(disconnect_button)
        disconnect_button.on_click(self.disconnect_mpl)

    def poly_img(self, img, pts):
        pts = np.array(pts, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (np.random.randint(
            0, 255), np.random.randint(0, 255), np.random.randint(0, 255)), 7)
        return img

    def onclick(self, event):
        # display(str(event))
        self.selected_points.append([event.xdata, event.ydata])
        if len(self.selected_points) > 1:
            self.fig
            self.img.set_data(self.poly_img(
                self.im.copy(), self.selected_points))
            self.fig.canvas.draw()
            self.move = True

    def disconnect_mpl(self, _):
        self.fig.canvas.mpl_disconnect(self.ka)
        
        
        
def interactive_ROI_selection(self):
        """import preprocessed intensity matrix and coordinate arrays, perform ROI annotation and selection, and store information for further data analysis

        Parameters
        ----------
        ROI_num : int, user input
            user-specified number of ROIs for dataset
        ROIs : str, user input
            ROI annotations from left to right, top to bottom
        """

        datasets = os.listdir(self.data_dir)
        datasets = np.asarray(datasets)[(np.char.find(
            datasets, "coords") == -1) & (np.char.find(datasets, "csv") != -1)]

        ROI_edge = pd.DataFrame()

        for rep, dataset in enumerate(datasets):
            print(f"Importing {dataset}.")
            df_build = pd.read_csv(f"{self.data_dir}/{dataset}")
            print(f"Finished importing {dataset}.")

            df_build.rename(columns={'Unnamed: 0': 'mz'}, inplace=True)
            df_build = df_build.T

            self.mz = df_build.loc['mz']
            df_build.drop('mz', inplace=True)

            df_coords = pd.read_csv(
                f"{self.data_dir}/{dataset[:-4]}_coords.csv")
            df_coords.rename(columns={'0': 'x', '1': 'y'}, inplace=True)
            df_coords['x'] = df_coords['x'] - np.min(df_coords['x']) + 1
            df_coords['y'] = df_coords['y'] - np.min(df_coords['y']) + 1

            df_build.reset_index(drop=True, inplace=True)
            df_coords.reset_index(drop=True, inplace=True)

            df_build = pd.concat([df_build, df_coords[['x', 'y']]], axis=1)

            # img_array_1c = make_image_1c(data_2darray = pd.concat([pd.Series(np.ones(df_build.iloc[:,:].shape[0])),df_build.iloc[:,-2:]], axis=1).to_numpy())
            img_array_1c = make_image_1c(data_2darray=pd.concat([pd.Series(np.sum(
                df_build.iloc[:, :-2], axis=1)), df_build.iloc[:, -2:]], axis=1).to_numpy())
            self.img_array_1c = img_array_1c
            
            self.ROI_num = int(input("How many ROIs are you analyzing? "))
            ROIs = input(
                "How to name your ROIs, from left to right, top to bottom? Separate ROI names by one space.")
            ROIs = ROIs.split(" ")

            for ROI in ROIs:
                ROI_select = bbox_select(img_array_1c[0])


           