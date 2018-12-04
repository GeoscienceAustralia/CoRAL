import unittest, glob
import os.path
import numpy as np
from coral.corner_reflector import readpar, readmli, calc_total_energy, calc_scr, calc_rcs, calc_integrated_energy, calc_clutter_intensity


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
        cls.par = readpar(g + '.par')
        d1 = readmli(g, cls.par, sub_im, cr)
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

        print(Ecr[0])
        self.assertEqual(round(Ecr[0]), 21) # 20.997258356639318


    def test_rcs_scr_calc(self):
        '''Test the RCS and SCR calculations'''
        Eclt = [3.9904459999999986]
        Nclt = [56]
        Ecr = [20.997258356639318]
        scr = calc_scr(Ecr, Eclt, Nclt)
        rcs = calc_rcs(Ecr, self.par)

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
            cls.par = readpar(g + '.par')
            cls.d[i] = readmli(g, cls.par, sub_im, cr)


    def test_rcs_time_series(self):
        '''Test the RCS and SCR calculations'''
        En, Ncr = calc_integrated_energy(self.d, self.cr_pos, self.targ_win_sz)
        E1, N1 = calc_integrated_energy(self.d, self.cr_pos, self.clt_win_sz)
        Avg_clt, Eclt, Nclt = calc_clutter_intensity(En, E1, Ncr, N1)
        Ecr = calc_total_energy(Ncr, Nclt, Eclt, En)
        scr = calc_scr(Ecr, Eclt, Nclt)
        rcs = calc_rcs(Ecr, self.par)

        self.assertEqual(round(np.nansum(rcs), 6), 215.210430)
        self.assertEqual(round(np.nansum(scr), 6), 120.524232)


if __name__ == '__main__':
    unittest.main()
