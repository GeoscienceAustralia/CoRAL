"""
This Python module contains unittests for testing CoRAL algorithms and workflow
"""
import unittest, glob
import os.path
import numpy as np
import cv2 # requires opencv, e.g. use: pip install --user opencv-python-headless
from coral.corner_reflector import *
from coral.plot import *
from coral.plot2 import *


class TestPlotting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        file_in = "data/coral_serf.conf"
        cls.name = 'SERF'
        cls.geom = 'Descending'
        cr = [88, 111] # this is used as initial coordinate
        files = []
        for file in glob.glob("data/*.mli"):
            files.append(file)
        files.sort()
        cls.params = cf.get_config_params(file_in)
        cls.params[cf.OUT_DIR] = './tests/'
        cls.avgI, cls.rcs, cls.scr, cls.clt, cls.t, cls.cr_new, cls.cr_pos = loop(files, cls.params[cf.SUB_IM], cr, \
                                                                cls.params[cf.TARG_WIN_SZ], cls.params[cf.CLT_WIN_SZ])
        start_time = cls.t[0]
        end_time = cls.t[-1]
        margin = (end_time - start_time) / 50
        cls.start = start_time - margin
        cls.end = end_time + margin

    def test_plot_mean_intensity(self):
        '''test function to plot mean intensity (single geometry)'''
        plot_mean_intensity(self.avgI, self.cr_pos, self.name, self.params)
        ref_image = './tests/ref_images/mean_intensity_SERF.png'
        act_image = './tests/mean_intensity_SERF.png'
        imageA = cv2.imread(ref_image)
        imageB = cv2.imread(act_image)
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2) # squared sum of differences
        self.assertEqual(err, 0.0)
        os.remove(act_image)

    def test_plot_clutter(self):
        '''test function to plot clutter time series (single geometry)'''
        plot_clutter(self.t, self.clt, self.start, self.end, self.name, self.geom, self.params)
        ref_image = './tests/ref_images/clutter_SERF.png'
        act_image = './tests/clutter_SERF.png'
        imageA = cv2.imread(ref_image)
        imageB = cv2.imread(act_image)
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2) # squared sum of differences
        self.assertEqual(err, 0.0)
        os.remove(act_image)

    def test_plot_scr(self):
        '''test function to plot SCR (single geometry)'''
        plot_scr(self.t, self.scr, self.start, self.end, self.name, self.geom, self.params)
        ref_image = './tests/ref_images/scr_SERF.png'
        act_image = './tests/scr_SERF.png'
        imageA = cv2.imread(ref_image)
        imageB = cv2.imread(act_image)
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)  # squared sum of differences
        self.assertEqual(err, 0.0)
        os.remove(act_image)

    def test_plot_rcs(self):
        '''test function to plot RCS (single geometry)'''
        plot_rcs(self.t, self.rcs, self.start, self.end, self.name, self.geom, self.params)
        ref_image = './tests/ref_images/rcs_SERF.png'
        act_image = './tests/rcs_SERF.png'
        imageA = cv2.imread(ref_image)
        imageB = cv2.imread(act_image)
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)  # squared sum of differences
        self.assertEqual(err, 0.0)
        os.remove(act_image)


class TestPlotting2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        file_in = "data/coral_serf2.conf"
        cls.name = 'SERF'
        cr = [88, 111] # this is used as initial coordinate
        files_a = []
        for file in glob.glob("data/*.tif"):
            files_a.append(file)
        files_a.sort()
        files_d = []
        for file in glob.glob("data/*.mli"):
            files_d.append(file)
        files_d.sort()
        cls.params = cf.get_config_params(file_in)
        cls.params[cf.OUT_DIR] = './tests/'
        cls.avgI_a, cls.rcs_a, cls.scr_a, cls.clt_a, cls.t_a, cls.cr_new_a, cls.cr_pos_a = loop(files_a, \
                                                                cls.params[cf.SUB_IM], cr, \
                                                                cls.params[cf.TARG_WIN_SZ], cls.params[cf.CLT_WIN_SZ])
        cls.avgI_d, cls.rcs_d, cls.scr_d, cls.clt_d, cls.t_d, cls.cr_new_d, cls.cr_pos_d = loop(files_d, \
                                                                cls.params[cf.SUB_IM], cr, \
                                                                cls.params[cf.TARG_WIN_SZ], cls.params[cf.CLT_WIN_SZ])
        start_time = min(cls.t_a[0], cls.t_d[0])
        end_time = max(cls.t_a[-1], cls.t_d[-1])
        margin = (end_time - start_time) / 50
        cls.start = start_time - margin
        cls.end = end_time + margin

    def test_plot_mean_intensity2(self):
        '''test function to plot mean intensity (both geometries)'''
        plot_mean_intensity2(self.avgI_a, self.avgI_d, self.cr_pos_a, self.cr_pos_d, self.name, self.params)
        ref_image = './tests/ref_images/mean_intensity_SERF2.png'
        act_image = './tests/mean_intensity_SERF.png'
        imageA = cv2.imread(ref_image)
        imageB = cv2.imread(act_image)
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2) # squared sum of differences
        self.assertEqual(err, 0.0)
        os.remove(act_image)

    def test_plot_clutter2(self):
        '''test function to plot clutter time series (single geometry)'''
        plot_clutter2(self.t_a, self.t_d, self.clt_a, self.clt_d, self.start, self.end, self.name, self.params)
        ref_image = './tests/ref_images/clutter_SERF2.png'
        act_image = './tests/clutter_SERF.png'
        imageA = cv2.imread(ref_image)
        imageB = cv2.imread(act_image)
        print(imageA.shape)
        print(imageB.shape)
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2) # squared sum of differences
        self.assertEqual(err, 0.0)
        os.remove(act_image)

    def test_plot_scr2(self):
        '''test function to plot SCR (single geometry)'''
        plot_scr2(self.t_a, self.t_d, self.scr_a, self.scr_d, self.start, self.end, self.name, self.params)
        ref_image = './tests/ref_images/scr_SERF2.png'
        act_image = './tests/scr_SERF.png'
        imageA = cv2.imread(ref_image)
        imageB = cv2.imread(act_image)
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)  # squared sum of differences
        self.assertEqual(err, 0.0)
        os.remove(act_image)

    def test_plot_rcs2(self):
        '''test function to plot RCS (single geometry)'''
        plot_rcs2(self.t_a, self.t_d, self.rcs_a, self.rcs_d, self.start, self.end, self.name, self.params)
        ref_image = './tests/ref_images/rcs_SERF2.png'
        act_image = './tests/rcs_SERF.png'
        imageA = cv2.imread(ref_image)
        imageB = cv2.imread(act_image)
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)  # squared sum of differences
        self.assertEqual(err, 0.0)
        os.remove(act_image)


if __name__ == '__main__':
    unittest.main()
