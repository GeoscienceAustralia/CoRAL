import unittest, glob
import os.path
import numpy as np
from coral.corner_reflector import calc_target_energy, readpar, readmli, calc_total_energy, calc_scr, calc_rcs


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


    def test_calc_target_energy(self):
        '''Test the target energy calculations'''
        En, Ncr, Eclt, Nclt, Avg_clt = calc_target_energy(self.d, self.cr_pos, self.targ_win_sz, self.clt_win_sz)
 
        self.assertEqual(round(En[0]), 23)
        self.assertEqual(Ncr[0], 25)
        self.assertEqual(round(Eclt[0]), 4)
        self.assertEqual(Nclt[0], 56)
        self.assertEqual(round(Avg_clt[0]), -11)


    def test_calc_total_energy(self):
        '''Test the total energy calculation'''
        En, Ncr, Eclt, Nclt, Avg_clt = calc_target_energy(self.d, self.cr_pos, self.targ_win_sz, self.clt_win_sz)
        Ecr = calc_total_energy(Ncr, Nclt, Eclt, En)

        self.assertEqual(round(Ecr[0]), 21)


    def test_rcs_scr_calc(self):
        '''Test the RCS and SCR calculations'''
        En, Ncr, Eclt, Nclt, Avg_clt = calc_target_energy(self.d, self.cr_pos, self.targ_win_sz, self.clt_win_sz)
        Ecr = calc_total_energy(Ncr, Nclt, Eclt, En)
        scr = calc_scr(Ecr, Eclt, Nclt)
        rcs = calc_rcs(Ecr, self.par)

        self.assertEqual(round(scr[0]), 25)
        self.assertEqual(round(rcs[0]), 37)


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
        En, Ncr, Eclt, Nclt, Avg_clt = calc_target_energy(self.d, self.cr_pos, self.targ_win_sz, self.clt_win_sz)
        Ecr = calc_total_energy(Ncr, Nclt, Eclt, En)
        scr = calc_scr(Ecr, Eclt, Nclt)
        rcs = calc_rcs(Ecr, self.par)

        self.assertEqual(round(np.nansum(rcs), 6), 215.210430)
        self.assertEqual(round(np.nansum(scr), 6), 120.524232)


if __name__ == '__main__':
    unittest.main()
