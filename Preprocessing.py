import numpy as np
import pydicom as pdicom
import os
import cv2
import matplotlib.pyplot as plt

'exec(%matplotlib inline)'

class PreProcessing:
    def __init__(self):
        pass

    def load_scan2(self, path, name):
        list1 = []
        for dirName, subdirList, fileList in os.walk(path):
            for filename in fileList:
                if name in filename.lower():
                    list1.append(os.path.join(dirName, filename))

        return list1

    def grayscale(self):
        f = self.load_scan2('/home/genomics/PycharmProjects/BreastCancerPredictor/CBIS-DDSM/','000000.dcm')
        d = []
        for i in range(0, len(f)):
            d.append(pdicom.read_file(f[i]))    # Read  dicom images


        for i in range(0, len(f)):
            img = d[i].pixel_array  # read pixels of image sfrom dicom image

            # Convert pixel_array (img) to -> gray image (img_2d_scaled)
            # Step 1. Convert to float to avoid overflow or underflow losses.
            img_2d = img.astype(float)
            # Step 2. Rescaling grey scale between 0-255
            img_2d_scaled = (np.maximum(img_2d, 0) / img_2d.max()) * 255
            ## Step 3. Convert to uint-16
            img_2d_scaled = np.uint16(img_2d_scaled)
            # converting integer image to byte to save it as a dicom image.
            d[i].PixelData = img_2d_scaled.tobytes()
            d[i].save_as("/home/genomics/PycharmProjects/BreastCancerPredictor/BIS-DDSM/Calc-Training_P_00635_RIGHT_MLO/08-07-2016-DDSM-68805/" + str(i) + ".dcm")


    def resize(self):
        f = self.load_scan2('/home/genomics/PycharmProjects/BreastCancerPredictor/Training-Benign/grayscaled/','.dcm')

        d = []  # list for reading images
        w = []  # stores width
        h = []  # stores height

        # read images and stores their width and height
        for i in range(0, len(f)):
            d.append(pdicom.read_file(f[i]))
            img = d[i].pixel_array
            print(img.shape)
            w.append(img.shape[0])
            h.append(img.shape[1])
        # print(d[1])

        # find max width and height
        maxw = np.amax(w)
        maxh = np.amax(h)
        print(maxw, maxh)

        dim = (maxh, maxw)
        resize = []
        for i in range(0, 5):
            img = d[i].pixel_array
            # resize to max width and height
            resize.append(cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR))
            print(resize[i].shape, i)


        for i in range(0, 5):
            d[i].PixelData = resize[i].tobytes()
            d[i].Rows = resize[i].shape[0]
            d[i].Columns = resize[i].shape[1]
            d[i].save_as("/home/genomics/PycharmProjects/BreastCancerPredictor/Training-Benign/resized/" + str(i + 5) + ".dcm")
            print(d[i].pixel_array.shape, i)

    def select_largest_obj(self, img_bin, lab_val=255, fill_holes=False, smooth_boundary=False, kernel_size=15):
        a="ff"
        # print(img_bin)
        n_labels, img_labeled, lab_stats , _ = cv2.connectedComponentsWithStats(img_bin, connectivity=8, ltype=cv2.CV_32S)

        largest_obj_lab = np.argmax(lab_stats[1:, 4]) + 1
        largest_mask = np.zeros(img_bin.shape, dtype=np.uint8)
        largest_mask[img_labeled == largest_obj_lab] = lab_val
        return largest_mask

    def max_pix_val(self, dtype):
        if dtype == np.dtype('uint8'):
            maxval = 2 ** 8 - 1
        elif dtype == np.dtype('uint16'):
            maxval = 2 ** 16 - 1
        else:
            raise Exception('Unknown dtype found in input image array')
        return maxval

    def suppress_artifacts(self, img, global_threshold=.05, fill_holes=False, smooth_boundary=True, kernel_size=15):


        maxval = self.max_pix_val(img.dtype)
        if global_threshold < 1.:
            low_th = int(img.max() * global_threshold)
        else:
            low_th = int(global_threshold)
        _, img_bin = cv2.threshold(img, low_th, maxval=maxval,
                                   type=cv2.THRESH_BINARY)

        breast_mask = self.select_largest_obj(img_bin, lab_val=maxval,
                                              fill_holes=True,
                                              smooth_boundary=True,
                                              kernel_size=kernel_size)
        plt.imshow(breast_mask, cmap='gray')
        plt.show()

        img_suppr = cv2.bitwise_and(img, breast_mask)
        return (img_suppr, breast_mask)



    def remove_pectoral(self, img, breast_mask, high_int_threshold=.8, morph_kn_size=3, n_morph_op=7, sm_kn_size=25):
        # Remove the pectoral muscle region from an input image
        # Args:
        #     img (2D array): input image as a numpy 2D array.
        #     breast_mask (2D array):
        #     high_int_threshold ([int]): a global threshold for high intensity
        #             regions such as the pectoral muscle. Default is 200.
        #     morph_kn_size ([int]): kernel size for morphological operations
        #             such as erosions and dilations. Default is 3.
        #     n_morph_op ([int]): number of morphological operations. Default is 7.
        #     sm_kn_size ([int]): kernel size for final smoothing (i.e. opening).
        #             Default is 25.
        # Returns:
        #     an output image with pectoral muscle region removed as a numpy
        #     2D array.
        # Notes: this has not been tested on .dcm files yet. It may not work!!!

        # Enhance contrast and then thresholding
        img_equ = cv2.equalizeHist(img)
        if high_int_threshold < 1.:
            high_th = int(img.max() * high_int_threshold)
        else:
            high_th = int(high_int_threshold)
        maxval = self.max_pix_val(img.dtype)
        _, img_bin = cv2.threshold(img_equ, high_th,
                                   maxval=maxval, type=cv2.THRESH_BINARY)
        pect_marker_img = np.zeros(img_bin.shape, dtype=np.int32)
        # Sure foreground (shall be pectoral).
        pect_mask_init = self.select_largest_obj(img_bin, lab_val=maxval,
                                                 fill_holes=True,
                                                 smooth_boundary=False)
        kernel_ = np.ones((morph_kn_size, morph_kn_size), dtype=np.uint8)
        pect_mask_eroded = cv2.erode(pect_mask_init, kernel_,
                                     iterations=n_morph_op)
        pect_marker_img[pect_mask_eroded > 0] = 255
        # Sure background - breast.
        pect_mask_dilated = cv2.dilate(pect_mask_init, kernel_,
                                       iterations=n_morph_op)
        pect_marker_img[pect_mask_dilated == 0] = 128
        # Sure background - pure background.
        pect_marker_img[breast_mask == 0] = 64
        # Watershed segmentation.
        img_equ_3c = cv2.cvtColor(img_equ, cv2.COLOR_GRAY2BGR)
        cv2.watershed(img_equ_3c, pect_marker_img)
        img_equ_3c[pect_marker_img == -1] = (0, 0, 255)
        # Extract only the breast and smooth.
        breast_only_mask = pect_marker_img.copy()
        breast_only_mask[breast_only_mask == -1] = 0
        breast_only_mask = breast_only_mask.astype(np.uint8)
        breast_only_mask[breast_only_mask != 128] = 0
        breast_only_mask[breast_only_mask == 128] = 255
        kernel_ = np.ones((sm_kn_size, sm_kn_size), dtype=np.uint8)
        breast_only_mask = cv2.morphologyEx(breast_only_mask, cv2.MORPH_OPEN,
                                            kernel_)
        img_breast_only = cv2.bitwise_and(img_equ, breast_only_mask)
        return (img_breast_only, img_equ_3c)

    def process(self, img_2d_scaled, blur_kn_size=3,  low_int_threshold=.05, high_int_threshold=.8):

        # perform multi-stage preprocessing on the input image
        # Args:
        #     blur_kn_size ([int]): kernel size for median blurring.
        #     low_int_threshold ([int]): cutoff used in artifacts suppression.
        #     high_int_threshold ([int]): cutoff used in pectoral muscle removal.
        # Returns:
        #     a tuple of (processed_image, color_image_with_boundary). If
        #     pectoral removal was not called, the color image is None.
        f = self.load_scan2('/home/genomics/PycharmProjects/BreastCancerPredictor/CBIS-DDSM/resized')
        d = []
        for i in range(0, len(f)):
            d.append(pdicom.read_file(f[i]))

        for i in range(0, len(f)):
            img = d[i].pixel_array
            # Step 3. Convert to uint8
            img_2d_scaled = np.uint8(img)
            img_proc = img_2d_scaled.copy()

            img_proc, mask_= self.suppress_artifacts(img_proc)
            img_proc, img_col = self.remove_pectoral(
            img_2d_scaled, mask_, high_int_threshold=high_int_threshold,)