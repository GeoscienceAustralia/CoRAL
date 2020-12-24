"""
This Python module contains unittests for testing CoRAL algorithms and workflow
"""
import unittest, glob
import os.path
import numpy as np
import cv2 # requires opencv, e.g. use: pip install --user opencv-python-headless
from coral.corner_reflector import *
from coral.dataio import readpar, readmli, read_radar_coords, write_radar_coords, read_input_files
from coral.plot import *
from coral.plot2 import *



class TestCoral(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        '''Set up params and open data for tests'''
        g = 'data/20180819_VV.mli'
        cr = [ 87, 110]
        sub_im = 51
        cls.targ_win_sz = 5
        cls.clt_win_sz = 9
        cls.cr_pos = np.array([sub_im, sub_im])

        # open files and read subset of image
        par = readpar(g + '.par')
        d1 = readmli(g, par, sub_im, cr)

        cls.rho_r = float(par['range_pixel_spacing'].split()[0])
        cls.rho_a = float(par['azimuth_pixel_spacing'].split()[0])
        cls.theta = float(par['incidence_angle'].split()[0])

        # force 3rd dimension of array
        cls.d = d1[np.newaxis, :, :,]


    def test_integrated_energy(self):
        '''Test the integrated energy calculations'''
        En, Ncr = calc_integrated_energy(self.d, self.cr_pos, self.targ_win_sz)
        E1, N1 = calc_integrated_energy(self.d, self.cr_pos, self.clt_win_sz)
 
        self.assertEqual(round(En[0]), 23) # 22.778708
        self.assertEqual(Ncr[0], 25) # 5*5=25
        self.assertEqual(round(E1[0]), 27) # 26.769154
        self.assertEqual(N1[0], 81) # 9*9=81

    def test_clutter_intensity(self):
        '''Test the average clutter calculations'''
        En = [22.778708]
        Ncr = [25]
        E1 = [26.769154]
        N1 = [81]
        Avg_clt, Eclt, Nclt = calc_clutter_intensity(En, E1, Ncr, N1)

        self.assertEqual(round(Avg_clt[0]), -11) # -11.47166578963688
        self.assertEqual(round(Eclt[0]), 4) # 3.9904459999999986
        self.assertEqual(Nclt[0], 56) # 81-24=56 

    def test_calc_total_energy(self):
        '''Test the total energy calculation'''
        Avg_clt = [-11.4716657896368]
        Eclt = [3.9904459999999986]
        Nclt = [56]
        Ncr = [25]
        En = [22.778708]
        Ecr = calc_total_energy(Ncr, Nclt, Eclt, En)

        self.assertEqual(round(Ecr[0]), 21) # 20.997258356639318


    def test_rcs_scr_calc(self):
        '''Test the RCS and SCR calculations'''
        Eclt = [3.9904459999999986]
        Nclt = [56]
        Ecr = [20.997258356639318]
        scr = calc_scr(Ecr, Eclt, Nclt)
        rcs = calc_rcs(Ecr, self.rho_r, self.rho_a, self.theta)

        self.assertEqual(round(scr[0]), 25) # 24.693291807917642
        self.assertEqual(round(rcs[0]), 37) # 36.968228116662544


class TestRCSTimeSeries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        '''Set up params and read data for tests'''
        cr = [ 87, 110]
        sub_im = 51
        cls.targ_win_sz = 5
        cls.clt_win_sz = 9
        cls.cr_pos = np.array([sub_im, sub_im])

        files = []
        for file in glob.glob("data/*.mli"):
            files.append(file)

        files.sort()
        # pre-allocate ndarray
        cls.d = np.empty((len(files),sub_im*2, sub_im*2))

        # open files and read subset of image
        for i, g in enumerate(files):
            par = readpar(g + '.par')
            cls.d[i] = readmli(g, par, sub_im, cr)

        cls.rho_r = float(par['range_pixel_spacing'].split()[0])
        cls.rho_a = float(par['azimuth_pixel_spacing'].split()[0])
        cls.theta = float(par['incidence_angle'].split()[0])


    def test_rcs_time_series(self):
        '''Test the RCS and SCR calculations'''
        En, Ncr = calc_integrated_energy(self.d, self.cr_pos, self.targ_win_sz)
        E1, N1 = calc_integrated_energy(self.d, self.cr_pos, self.clt_win_sz)
        Avg_clt, Eclt, Nclt = calc_clutter_intensity(En, E1, Ncr, N1)
        Ecr = calc_total_energy(Ncr, Nclt, Eclt, En)
        scr = calc_scr(Ecr, Eclt, Nclt)
        rcs = calc_rcs(Ecr, self.rho_r, self.rho_a, self.theta)

        self.assertEqual(round(np.nansum(rcs), 6), 215.210430)
        self.assertEqual(round(np.nansum(scr), 6), 120.524232)


class TestLoop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = []
        for file in glob.glob("data/*.mli"):
            cls.files.append(file)
        cls.files.sort()
        cls.cr = [ 87, 110]
        cls.sub_im = 51
        cls.targ_win_sz = 5
        cls.clt_win_sz = 9

    def test_loop(self):
        '''test the calculation loop function'''
        avgI, rcs, scr, Avg_clt, t, cr_new, cr_pos = loop(self.files, self.sub_im, self.cr, self.targ_win_sz, self.clt_win_sz)

        # test the mean value of the average intensity image
        self.assertEqual(round(np.mean(avgI), 6), -11.385613) # -11.385613198856245


class TestTiff(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = []
        for file in glob.glob("data/*.tif"):
            cls.files.append(file)

        cls.files.sort()
        cls.cr = [ 87, 110] # col, row
        cls.sub_im = 51
        cls.targ_win_sz = 5
        cls.clt_win_sz = 9

    def test_loop(self):
        '''test the calculation loop function'''
        avgI, rcs, scr, Avg_clt, t, cr_new, cr_pos = loop(self.files, self.sub_im, self.cr, self.targ_win_sz, self.clt_win_sz)

        # test the mean value of the average intensity image
        self.assertEqual(round(np.mean(avgI), 6), -11.385613) # -11.385613198856245
        
        
class TestShift(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = []
        for file in glob.glob("data/*.mli"):
            cls.files.append(file)

        cls.files.sort()
        cls.sub_im = 51
        cls.targ_win_sz = 5
        cls.clt_win_sz = 9
        cls.cr = [ 86, 111] # coordinates changed to check shift

    def test_loop(self):
        '''test the coordinate shift'''
        avgI, rcs, scr, Avg_clt, t, cr_new, cr_pos = loop(self.files, self.sub_im, self.cr, self.targ_win_sz, self.clt_win_sz)

        # test array containing values of shift to be applied to radar coordinates
        self.assertEqual(cr_pos[0], 52)  
        self.assertEqual(cr_pos[1], 50)         


class TestConfigFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.file_in = "data/coral_serf.conf"
        cls.sites = {'SERF': np.array([[-999, -999], [88, 111]])}

    def test_config_file(self):
        '''test function to read parameters from the config-file'''
        params = cf.get_config_params(self.file_in)
        self.assertEqual(params[cf.SUB_IM], 51)
        self.assertEqual(params[cf.ASC_LIST], None)
        self.assertEqual(params[cf.DESC_LIST], 'data/mli_desc.list')

    def test_read_input_files(self):
        '''test function to read input files from config-file'''
        params = cf.get_config_params(self.file_in)
        files_a, files_d, sites = read_input_files(params)
        self.assertEqual(files_a, None)
        self.assertEqual(files_d[0], './data/20180726_VV.mli')
        self.assertEqual(files_d[8], './data/20181030_VV.mli')
        self.assertEqual(self.sites.keys(), sites.keys())
        array1 = sites.get('SERF')
        array2 = self.sites.get('SERF')
        np.testing.assert_array_equal(array1, array2)


class TestCRfiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.file_in1 = "data/site_az_rg.txt"
        cls.file_in2 = "data/site_lon_lat_hgt_date1_date2_az_rg.txt"
        cls.file_out1 = "data/site_az_rg_new.txt"
        cls.file_out2 = "data/site_lon_lat_hgt_date1_date2_az_rg_new.txt"
        cls.sites = {'SERF' : np.array([[-999, -999],[ 87, 110]])}  
        cls.geom = "desc"
    
    def test_cr_file(self):           
        '''test function to read and write the radar coordinate files'''
        # open reduced CR coordinate file
        site, az, rg = read_radar_coords(self.file_in1)
        self.assertEqual(int(rg[0]), 87)
        self.assertEqual(int(az[0]), 110)
        # write to new file
        write_radar_coords(self.file_in1, self.file_out1, self.sites, self.geom)
        assert os.path.exists(self.file_out1) == 1
        os.remove(self.file_out1)
        
        # open full CR coordinate file
        site, az, rg = read_radar_coords(self.file_in2)
        self.assertEqual(int(rg[0]), 87)
        self.assertEqual(int(az[0]), 110)
        # write to new file
        write_radar_coords(self.file_in2, self.file_out2, self.sites, self.geom)
        assert os.path.exists(self.file_out2) == 1
        os.remove(self.file_out2)


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
