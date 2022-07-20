from ROOT import TH1F, TH1
import ROOT
from numpy import sqrt, log, ceil, log2
from tqdm import tqdm
import numpy as np
from pdb import set_trace

def calc_sig(sig, bkg,s_err,b_err):

  ntot = sig + bkg

  #if(sig <= 0): return 0, 0
  #if(bkg <= 0): return 0, 0

  #Counting experiment
  significance = sqrt(2*((ntot * log(ntot/bkg)) - sig))
  #significance = sig / sqrt(bkg)

  #error on significance
  numer = sqrt((log(ntot/bkg)*s_err)**2 + ((log(1+(sig/bkg)) - (sig/bkg))*b_err)**2)
  uncert = (numer/significance)

  return significance, uncert

def sum_z(zs):
    sumu=0
    sumz=0
    for i in range(len(zs)):
        sumz+=zs[i][0]**2
        sumu+=(zs[i][0]*zs[i][1])**2
    sumz=sqrt(sumz)
    sumu=sqrt(sumu)/sumz if sumz != 0 else 0
    return sumz, sumu

def copy_hist(hname, h0, bl, br):

    nbin = br - bl + 1
    l_edge = h0.GetBinLowEdge(bl)
    r_edge = l_edge + h0.GetBinWidth(bl) * nbin
    h1 = TH1F(hname, hname, nbin, l_edge, r_edge)

    for i, j in enumerate(range(bl, br+1)):

        y = h0.GetBinContent(j)
        err = h0.GetBinError(j)

        h1.SetBinContent(i+1, y)
        h1.SetBinError(i+1, err)
        
    return h1

class pyTH1(object):
    """Customized TH1 class"""

    def __init__(self, hist):

        assert isinstance(hist, TH1), 'hist==ROOT.TH1F'

        self.BinContent = np.array([])
        self.BinError = np.array([])
        self.BinLowEdge = np.array([])
        self.BinWidth = np.array([])

        for i in range(1, hist.GetSize()-1):

            self.BinContent = np.append(self.BinContent, hist.GetBinContent(i))
            self.BinError = np.append(self.BinError, hist.GetBinError(i))
            self.BinLowEdge = np.append(self.BinLowEdge, hist.GetBinLowEdge(i))
            self.BinWidth = np.append(self.BinWidth, hist.GetBinWidth(i))

    def GetSize(self):

        return len(self.BinContent)

    def GetLowEdge(self):

        return self.BinLowEdge[0]

    def GetHighEdge(self):

        return self.BinLowEdge[-1] + self.BinWidth[-1]

    def GetBinContent(self, i=0):

        if i==0: return self.BinContent
        return self.BinContent[i-1]

    def GetBinError(self, i=0):

        if i==0: return self.BinError
        return self.BinError[i-1]

    def GetBinLowEdge(self, i=0):

        if i==0: return self.BinLowEdge
        return self.BinLowEdge[i-1]

    def GetBinWidth(self, i=0):

        if i==0: return self.BinWidth
        return self.BinWidth[i-1]

    def Integral(self, i=0, j=0):

        if i==0 and j==0:
            return self.BinContent.sum()

        return self.BinContent[i-1:j].sum()

    def IntegralAndError(self, i=0, j=0):

        if i==0 and j==0:
            return self.BinContent.sum(), sqrt((self.BinError**2).sum())

        return self.BinContent[i-1:j].sum(), sqrt((self.BinError[i-1:j]**2).sum())

    def to_TH1F(self, name, bl=-1, br=-1):

        if bl == -1 and br == -1:
            bl = 1
            br = self.GetSize()

        hist = TH1F(name, name, br - bl + 1, self.GetBinLowEdge(bl), self.GetBinLowEdge(br) + self.GetBinWidth(br))

        for i, j in enumerate(range(bl, br+1)):

            y = self.GetBinContent(j)
            err = self.GetBinError(j)

            hist.SetBinContent(i+1, y)
            hist.SetBinError(i+1, err)

        return hist


class categorizer(object):

    def __init__(self, h_sig, h_bkg, h_bkg_rw_num=None, h_bkg_rw_den=None):

        assert isinstance(h_sig, TH1), 'h_sig==ROOT.TH1'
        assert isinstance(h_bkg, TH1), 'h_bkg==ROOT.TH1'
        assert h_sig.GetSize() == h_bkg.GetSize(), 'h_sig and h_bkg have different sizes'
        assert h_sig.GetBinLowEdge(1) == h_bkg.GetBinLowEdge(1), 'h_sig and h_bkg have different edges'
        assert h_sig.GetBinWidth(1) == h_bkg.GetBinWidth(1), 'h_sig and h_bkg have different bin widths'
        self.reweight = False
        if h_bkg_rw_num and h_bkg_rw_den:
            self.reweight = True
            assert h_sig.GetSize() == h_bkg_rw_num.GetSize(), 'h_sig and h_bkg have different sizes'
            assert h_sig.GetBinLowEdge(1) == h_bkg_rw_num.GetBinLowEdge(1), 'h_sig and h_bkg have different edges'
            assert h_sig.GetBinWidth(1) == h_bkg_rw_num.GetBinWidth(1), 'h_sig and h_bkg have different bin widths'
            assert h_sig.GetSize() == h_bkg_rw_den.GetSize(), 'h_sig and h_bkg have different sizes'
            assert h_sig.GetBinLowEdge(1) == h_bkg_rw_den.GetBinLowEdge(1), 'h_sig and h_bkg have different edges'
            assert h_sig.GetBinWidth(1) == h_bkg_rw_den.GetBinWidth(1), 'h_sig and h_bkg have different bin widths'

        self.h_sig = pyTH1(h_sig)
        self.h_bkg = pyTH1(h_bkg)
        if self.reweight:
            self.h_bkg_rw_num = pyTH1(h_bkg_rw_num)
            self.h_bkg_rw_den = pyTH1(h_bkg_rw_den)

    def smooth(self, bl, br, SorB='B', function='Epoly2'):

        if SorB == 'S': hist = self.h_sig
        elif SorB == 'B': hist = self.h_bkg

        nbin = hist.GetSize()
        l_edge = hist.GetLowEdge()
        r_edge = hist.GetHighEdge()

        h_merge_list = ROOT.TList()

        if bl != 1:
            h_left = hist.to_TH1F('h_left', 1, bl-1)
            h_merge_list.Add(h_left)

        hist_to_smooth = hist.to_TH1F('hist_to_smooth', bl, br)
        smoothed_hist = fit_BDT('smoothed_hist', hist_to_smooth, function)
        h_merge_list.Add(smoothed_hist)

        if br != nbin:
            h_right = hist.to_TH1F('h_right', br+1, nbin)
            h_merge_list.Add(h_right)

        htemp = TH1F('htemp', 'htemp', nbin, l_edge, r_edge)
        htemp.Merge(h_merge_list)

        if SorB == 'S':
            self.h_sig = pyTH1(htemp)
        elif SorB == 'B':
            self.h_bkg = pyTH1(htemp)

        if bl != 1: h_left.Delete()
        hist_to_smooth.Delete()
        smoothed_hist.Delete()
        if br != nbin: h_right.Delete()
        htemp.Delete()

    def fit(self, bl, br, nbin, minN=5, floatB=False, earlystop=-1, pbar=False):

        if nbin == 1:

            if floatB: return [], 0

            nsig, dsig = self.h_sig.IntegralAndError(bl, br)
            nbkg, dbkg = self.h_bkg.IntegralAndError(bl, br)
            if self.reweight and nbkg != 0:
                if self.h_bkg_rw_den.Integral(bl, br) == 0: print("what!!!", bl, br)
                nbkg, dbkg = nbkg*self.h_bkg_rw_num.Integral(bl, br)/self.h_bkg_rw_den.Integral(bl, br), dbkg*self.h_bkg_rw_num.Integral(bl, br)/self.h_bkg_rw_den.Integral(bl, br)

            if nbkg < minN: return -1, -1

            z, u = calc_sig(nsig, nbkg, dsig, dbkg)

            return [bl], z

        elif nbin > 1:

            L = int(ceil(log2(nbin)))
            N2 = 2**(L - 1)
            N1 = nbin - N2

            bmax, zmax, stop = -1, -1, 0
 
            for b in (range(bl, br+1) if not pbar else tqdm(range(bl, br+1), ncols=70)):

                b1, z1 = self.fit(bl, b-1, N1, minN=minN, floatB=floatB, earlystop=earlystop)
                if b1 == -1: continue
                b2, z2 = self.fit(b, br, N2, minN=minN, earlystop=earlystop)
                if b2 == -1: break

                z = sqrt(z1**2 + z2**2)
                if z > zmax:
                    stop = 0
                    zmax = z
                    bmax = sorted(list(set(b1 + [b] + b2)))
                else:
                    stop += 1
                if stop == earlystop: break

            return bmax, zmax

def fit_BDT(hname, hist, function='Epoly2', printMessage=False):

    if not printMessage: ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)  #WARNING

    n_events = hist.Integral()

    nbin = hist.GetSize() - 2
    l_edge = hist.GetBinLowEdge(1)
    r_edge = l_edge + hist.GetBinWidth(1) * nbin

    bdt = ROOT.RooRealVar("bdt", "bdt", l_edge, r_edge)

    a = ROOT.RooRealVar("a", "a", -100, 100)
    b = ROOT.RooRealVar("b", "b", -100, 100)
    c = ROOT.RooRealVar("c", "c", -100, 100)
    d = ROOT.RooRealVar("d", "d", -100, 100)
    e = ROOT.RooRealVar("e", "e", -100, 100)
    f = ROOT.RooRealVar("f", "f", -100, 100)

    pdf = {}
    pdf['power'] = ROOT.RooGenericPdf("pdf","PDF for BDT distribution", "@0**@1", ROOT.RooArgList(bdt, a)) #power

    # EPoly Family
    pdf['Exp'] = ROOT.RooGenericPdf("pdf","PDF for BDT distribution", "exp(@1*@0)", ROOT.RooArgList(bdt, a))
    pdf['Epoly2'] = ROOT.RooGenericPdf("pdf","PDF for BDT distribution", "exp(@1*@0+@2*@0**2)", ROOT.RooArgList(bdt, a, b))
    pdf['Epoly3'] = ROOT.RooGenericPdf("pdf","PDF for BDT distribution", "exp(@1*@0+@2*@0**2+@3*@0**3)", ROOT.RooArgList(bdt, a, b, c))
    pdf['Epoly4'] = ROOT.RooGenericPdf("pdf","PDF for BDT distribution", "exp(@1*@0+@2*@0**2+@3*@0**3+@4*@0**4)", ROOT.RooArgList(bdt, a, b, c, d))
    pdf['Epoly5'] = ROOT.RooGenericPdf("pdf","PDF for BDT distribution", "exp(@1*@0+@2*@0**2+@3*@0**3+@4*@0**4+@5*@0**5)", ROOT.RooArgList(bdt, a, b, c, d, e))
    pdf['Epoly6'] = ROOT.RooGenericPdf("pdf","PDF for BDT distribution", "exp(@1*@0+@2*@0**2+@3*@0**3+@4*@0**4+@5*@0**5+@6*@0**6)", ROOT.RooArgList(bdt, a, b, c, d, e, f))

    data = ROOT.RooDataHist("dh", "dh", ROOT.RooArgList(bdt), hist)
    #data.Print("all")

    pdf[function].fitTo(data, ROOT.RooFit.Verbose(False), ROOT.RooFit.PrintLevel(-1))

    if printMessage:
        a.Print()
        b.Print()
        c.Print()
        d.Print()
        e.Print()
        f.Print()
    
        frame = bdt.frame()
        data.plotOn(frame)
        pdf[function].plotOn(frame)
    
        frame.Draw()
        #frame.Print('test.pdf')
    
        dof = {'power': 2, 'Exp': 2, 'Epoly2': 3, 'Epoly3': 4, 'Epoly4': 5, 'Epoly5': 6, 'Epoly6': 7}
        reduced_chi_square = frame.chiSquare(dof[function])
        probability = TMath.Prob(frame.chiSquare(dof[function]) * (nbin - dof[function]), nbin - dof[function])
        print('chi square:', reduced_chi_square)
        print('probability: ', probability)

    #raw_input('Press enter to continue')

    # fill the fitted pdf into a histogram
    hfit = TH1F(hname, hname, nbin, l_edge, r_edge)
    pdf[function].fillHistogram(hfit, ROOT.RooArgList(bdt), n_events)

    return hfit
