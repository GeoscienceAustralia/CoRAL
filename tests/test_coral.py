"""
This Python module contains unittests for testing CoRAL algorithms and workflow
"""
import unittest, glob
import os.path
import numpy as np
from coral.corner_reflector import *
from coral.dataio import readpar, readmli

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
        avgI, rcs, scr, Avg_clt, t = loop(self.files, self.sub_im, self.cr, self.targ_win_sz, self.clt_win_sz)

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
        avgI, rcs, scr, Avg_clt, t = loop(self.files, self.sub_im, self.cr, self.targ_win_sz, self.clt_win_sz)

        # test the mean value of the average intensity image
        self.assertEqual(round(np.mean(avgI), 6), -11.385613) # -11.385613198856245


if __name__ == '__main__':
    unittest.main()
