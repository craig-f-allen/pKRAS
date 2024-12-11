import numpy as np
import pandas as pd
import copy

class KRAS_Variant:
    def __init__(self,name,multipliers=[1 for i in range(17)],color = 'grey',kT=None):
        
        self.name = name
        self.multipliers = multipliers
        self.color = color
        self.type = None

        self.volscale = 250
        self.kint = (3.5e-4)*multipliers[0]
        self.kdissD = (1.1e-4)*multipliers[1]
        self.kdissT = (2.5e-4)*multipliers[2]
        self.kassD = (2.3e6)*multipliers[3]
        
        self.kcat = (5.4)*multipliers[5]
        self.Km = (.23e-6)/self.volscale*multipliers[6]
        self.kD = 3.9*multipliers[7]
        self.KmD = (3.86e-4)/self.volscale*multipliers[8]
        self.KmT = (3e-4)/self.volscale*multipliers[9]
        #if dep_var == 'kassT':
            
        # determines if kassT is dependent or kT depedent. WT should be depedent kT, mutants should be kassT dependent.
        if kT:
            self.type = 'dep_kassT'
            self.kT = copy.deepcopy(kT)*multipliers[10]
            self.kassT = self.kD*self.KmT*((self.kassD*self.kdissT)/(self.kdissD*self.kT))/self.KmD
            Haldaneint=(self.kassD*self.kdissT)/(self.kdissD*self.kassT)
        else:
            self.type = 'dep_kT'
            self.kassT = (2.2e6)*multipliers[4]
            
        self.Kd=(80e-9)*multipliers[11]
        self.kassEff=(4.5e7)*multipliers[12]

        # drug params
        self.k_on_SOSi = (1.1e7)/self.volscale
        self.k_off_SOSi = self.k_on_SOSi*(470e-9) # from BI-3407 Hofman Paper, K_D = 470 nmol/L
        self.k_on_tricomplex = (1e7)/self.volscale
        self.K_D_2 = (115e-9)
        self.k_on_panKRASi = 1.6e7*multipliers[15] # from https://doi.org/10.1038/s41586-023-06123-3
        self.k_off_panKRASi = 0.042*multipliers[16] # from https://doi.org/10.1038/s41586-023-06123-3
        self.k_on_panBRAFi = 1e7/self.volscale
        self.k_off_panBRAFi = self.k_on_panBRAFi*(6.13e-9) #K_D from https://www.sciencedirect.com/science/article/pii/S0021925823002764?via%3Dihub
        self.Kd_KRAS_OFF = 3.7e-6 #from https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b02052
        self.Kd_KRAS_OFF_ON_D = 4e-6
        self.Kd_KRAS_OFF_ON_T = 4e-6

        HaldaneGEF=self.kT*self.KmD/(self.kD*self.KmT)

        """
        if round(Haldaneint,ndigits=7) != round(HaldaneGEF,ndigits=7):
            print('Haldane coefficients not equal for {}'.format(self.name))
            print('Haldaneint: {}, HaldaneGEF: {}\n'.format(Haldaneint,HaldaneGEF))
        """

    @property
    def kT(self):
        Haldaneint=(self.kassD*self.kdissT)/(self.kdissD*self.kassT) # GTP, GDP not baked in.
        return self.kD*self.KmT*Haldaneint/self.KmD # GTP, GDP not baked in.
    
    @kT.setter
    def kT(self,kT):
        self.kassT = self.kD*self.KmT*((self.kassD*self.kdissT)/(self.kdissD*kT))/self.KmD

    @property
    def kdissEff(self):
        return self.kassEff*self.Kd
    
    @property
    def k_off_tricomplex(self):
        return self.k_on_tricomplex*self.K_D_2

WT = KRAS_Variant('WT',color='forestgreen')

kT = WT.kT
mutants_df = pd.read_excel('/home/ca784/Documents/RAS/RAS_ODE_model_kinetic_parameters_v2.xlsx',index_col=0).iloc[:, 0:-2]
def make_KRAS_Variant_from_index(index,color='grey',kT=None):
    return KRAS_Variant(index,list(mutants_df.loc[index])+[1]*3,color=color,kT=kT)

WT_Mut = KRAS_Variant('WT',color='forestgreen',kT=kT)
A146T = make_KRAS_Variant_from_index('A146T',kT=kT) #passing in WT kT to specify that kassT is dependent.
A146V = make_KRAS_Variant_from_index('A146V',kT=kT)
A59T = make_KRAS_Variant_from_index('A59T',kT=kT)
F28L = make_KRAS_Variant_from_index('F28L',kT=kT)
G12A = make_KRAS_Variant_from_index('G12A',kT=kT)
G12C = make_KRAS_Variant_from_index('G12C',color='royalblue',kT=kT)
G12D = make_KRAS_Variant_from_index('G12D',color='cornflowerblue',kT=kT)
G12E = make_KRAS_Variant_from_index('G12E',kT=kT)
G12P = make_KRAS_Variant_from_index('G12P',kT=kT)
G12R = make_KRAS_Variant_from_index('G12R',kT=kT)
G12S = make_KRAS_Variant_from_index('G12S',kT=kT)
G12V = make_KRAS_Variant_from_index('G12V',color='lightskyblue',kT=kT)
G13C = make_KRAS_Variant_from_index('G13C',kT=kT)
G13D = make_KRAS_Variant_from_index('G13D',color='orchid',kT=kT)
G13S = make_KRAS_Variant_from_index('G13S',kT=kT)
G13V = make_KRAS_Variant_from_index('G13V',kT=kT)
Q61H = make_KRAS_Variant_from_index('Q61H',kT=kT)
Q61K = make_KRAS_Variant_from_index('Q61K',kT=kT)
Q61L = make_KRAS_Variant_from_index('Q61L',kT=kT,color='crimson')
Q61P = make_KRAS_Variant_from_index('Q61P',kT=kT)
Q61R = make_KRAS_Variant_from_index('Q61R',kT=kT)
Q61W = make_KRAS_Variant_from_index('Q61W',kT=kT)
all_mutants = [WT_Mut,A146T,A146V,A59T,F28L,G12A,G12C,G12D,G12E,G12P,G12R,G12S,G12V,G13C,G13D,G13S,G13V,Q61H,Q61K,Q61L,Q61P,Q61R,Q61W]  #A59T #10gly11 or 10dupG

#OLD PARAMS
A59T_old = KRAS_Variant('A59T',[(2.4/10.2),(4.4/.79),(2.8/2.3),1,1,(2.4/10.2)*(3.15e-4)/5.4,1,1,1,1,1,1,1,1,1,1,1,1])
A146T_old = KRAS_Variant('A146T',[1,(244.9/0.51),(244.9/0.51),(244.9/0.51)*(1.1e-4)/(9.3e-9*2.3e6),(244.9/0.51)*(2.5e-4)/(6.9e-9*2.2e6),(708/833),1,(6232/62.5),1,1,1,(317/111),1,1,1,1,1,1]) #SWITCH OVER TO OTHER PARAMS #TODO
G12C_old = KRAS_Variant('G12C',[(49/68),1,1,1,1,(49/68)*(3.15e-4)/5.4,1,1,1,1,1,(67/56),1,1,1,1,1,1],color='royalblue')
G12D_old = KRAS_Variant('G12D',[(1.04/2.60),(2.0/4.2),5,(.7/.51),(4.8/1.4),(1.04/2.60)*(3.15e-4)/5.4,1,1,1,1,1,1,1,1,1,1,1,1],color='cornflowerblue')
G12V_old = KRAS_Variant('G12V',[(.39/2.6),(1.3/4.2),.8,(1.16/.51),(5.8/1.4),(.39/2.6)*(3.15e-4)/5.4,1,1,1,1,1,(1/2.25),1,1,1,1,1,1],color='lightskyblue')
F28L_old = KRAS_Variant('F28L',[1,25,25,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
G12P_old = KRAS_Variant('G12P',[(.043/.028),(1.4/2),(13/11),1,1,(.09/10),1,1,1,1,1,(57/18),1,1,1,1,1,1])
G12R_old = KRAS_Variant('G12R',[(1.4/10.2),(.37/.79),(.11/2.3),1,1,(1.4/10.2)*(3.15e-4)/5.4,1,1,1,1,1,1,1,1,1,1,1,1])
G13D_old = KRAS_Variant('G13D',[(9.6/68),(5.86/1.6),(5.86/1.6),1,1,(9.6/68)*(3.15e-4)/5.4,100,1,1,1,1,1,1,1,1,1,1,1],color='orchid')
G13S_old = KRAS_Variant('G13S',[(.032/.028),(.023/.0079),(.005/.005),1,1,(.032/.028)*(3.15e-4)/5.4,1,1,1,1,1,1,1,1,1,1,1,1])
G13V_old = KRAS_Variant('G13V',[(.013/.028),(.63/.0079),(.03/.005),1,1,(.013/.028)*(3.15e-4)/5.4,1,1,1,1,1,1,1,1,1,1,1,1])
Q61H_old = KRAS_Variant('Q61H',[(1.83/20.93),(9.6/6.5),(9.2/15),1,1,(1.83/20.93)*(3.15e-4)/5.4,(25/35),1,1,1,1,1,1,1,1,1,1,1])
Q61K_old = KRAS_Variant('Q61K',[(0.348/20.93),(9.1/6.5),(9/15),1,1,(0.348/20.93)*(3.15e-4)/5.4,(2000/35),1,1,1,1,1,1,1,1,1,1,1])
Q61L_old = KRAS_Variant('Q61L',[(.455/20.93),(16/6.5),(16/15),1,1,(.455/20.93)*(3.15e-4)/5.4,(2/35),1,1,1,1,(60/18),1,1,1,1,1,1],color='crimson')
Q61P_old = KRAS_Variant('Q61P',[(1.63/20.93),(6.8/6.5),(19/15),1,1,(1.63/20.93)*(3.15e-4)/5.4,(50/35),1,1,1,1,1,1,1,1,1,1,1])
Q61R_old = KRAS_Variant('Q61R',[(0.366/20.93),(7.4/6.5),(6.7/15),1,1,(0.366/20.93)*(3.15e-4)/5.4,(1500/35),1,1,1,1,1,1,1,1,1,1,1])
Q61W_old = KRAS_Variant('Q61W',[(1.52/20.93),(6.5/6.5),(12/15),1,1,(1.52/20.93)*(3.15e-4)/5.4,(100/35),1,1,1,1,1,1,1,1,1,1,1])
all_mutants_old = [WT,A146T_old,A59T_old,G12C_old,G12D_old,G12V_old,F28L_old,G12P_old,G12R_old,G13D_old,G13S_old,G13V_old,Q61H_old,Q61K_old,Q61L_old,Q61P_old,Q61R_old,Q61W_old]  #A59T #10gly11 or 10dupG

def RAS_model(t,y,k):

    GD=y[0]
    GT=y[1]
    G0=y[2]
    Eff=y[3]
    GTEff=y[4]
    GDV=y[5]
    GTV=y[6]
    G0V=y[7]
    GTEffV=y[8]

    GAP = k[26]
    GDP = k[27]
    GTP = k[28]
    GEF = k[29]

    VmaxD=k[0]*GEF
    VmaxT=k[1]*GEF*GDP/GTP
    KmD=k[2]
    KmT=k[3]
    Vmax=k[4]*GAP
    Km=k[5]
    kint=k[6]
    kdissD=k[7]
    kdissT=k[8]
    kassDGDP=k[9]*GDP
    kassTGTP=k[10]*GTP
    kassEff=k[11]
    kdissEff=k[12]

    VmaxDV=k[13]*GEF
    VmaxTV=k[14]*GEF*GDP/GTP
    KmDV=k[15]
    KmTV=k[16]
    VmaxV=k[17]*GAP
    KmV=k[18]
    kintV=k[19]
    kdissDV=k[20]
    kdissTV=k[21]
    kassDGDPV=k[22]*GDP
    kassTGTPV=k[23]*GTP
    kassEffV=k[24]
    kdissEffV=k[25]

    R1=(VmaxD*GD/KmD-VmaxT*GT/KmT)/(1+GD/KmD+GT/KmT+GDV/KmDV+GTV/KmTV)
    R2=Vmax*GT/(Km*(1+GTV/KmV)+GT)
    R3=kint*GT
    R4=kdissD*GD-kassDGDP*G0
    R5=kdissT*GT-kassTGTP*G0
    R6=kassEff*GT*Eff-kdissEff*GTEff
    R7=kint*GTEff

    R8=(VmaxDV*GDV/KmDV-VmaxTV*GTV/KmTV)/(1+GD/KmD+GT/KmT+GDV/KmDV+GTV/KmTV)
    R9=VmaxV*GTV/(KmV*(1+GT/Km)+GTV)
    R10=kintV*GTV
    R11=kdissDV*GDV-kassDGDPV*G0V
    R12=kdissTV*GTV-kassTGTPV*G0V
    R13=kassEffV*GTV*Eff-kdissEffV*GTEffV
    R14=kintV*GTEffV

    dydt=[-R1+R2+R3-R4+R7,
        R1-R2-R3-R5-R6,
        R4+R5,
        -R6+R7-R13+R14,
        (R6-R7),
        (-R8+R9+R10-R11+R14),
        (R8-R9-R10-R12-R13),
        (R11+R12),
        (R13-R14)]
    return dydt

def get_params_RAS(get_param_opts):

    mutant = get_param_opts['mutant']
    fract_mut = get_param_opts['fract_mut']
    GTot = get_param_opts['Total RAS']
    EffTot = get_param_opts['Effector']
    GAP = get_param_opts['GAP']
    GDP = get_param_opts['GDP']
    GTP = get_param_opts['GTP']
    GEF = get_param_opts['GEF']
    WT = KRAS_Variant('WT')
    
    WTRasTot=(1-fract_mut)*GTot
    MutRasTot=fract_mut*GTot
    
    y0 = np.zeros(9)
    y0[0]=WTRasTot #GD=y(1);
    y0[1]=0 #GT=y(2);
    y0[2]=0 #G0=y(3);
    y0[3]=EffTot #Eff=y(4);
    y0[4]=0 #GTEff=y(5);
    y0[5]=MutRasTot #GDV=y(6);
    y0[6]=0 #GTV=y(7);
    y0[7]=0 #G0V=y(8);
    y0[8]=0 #GTVEff=y(9);

    k = np.ones(33)
    k[0] = WT.kD
    k[1] = WT.kT
    k[2] = WT.KmD
    k[3] = WT.KmT
    k[4] = WT.kcat
    k[5] = WT.Km
    k[6] = WT.kint
    k[7] = WT.kdissD
    k[8] = WT.kdissT
    k[9] = WT.kassD
    k[10] = WT.kassT
    k[11] = WT.kassEff
    k[12] = WT.kdissEff

    k[13] = mutant.kD
    k[14] = mutant.kT
    k[15] = mutant.KmD
    k[16] = mutant.KmT
    k[17] = mutant.kcat
    k[18] = mutant.Km
    k[19] = mutant.kint
    k[20] = mutant.kdissD
    k[21] = mutant.kdissT
    k[22] = mutant.kassD
    k[23] = mutant.kassT
    k[24] = mutant.kassEff
    k[25] = mutant.kdissEff

    k[26] = GAP
    k[27] = GDP
    k[28] = GTP
    k[29] = GEF

    k[30] = y0[0]+y0[1]+y0[2]+y0[4]
    k[31] = y0[5]+y0[6]+y0[7]+y0[8]
    k[32] = y0[3]+y0[4]+y0[8]

    return np.array(k),np.array(y0)

# K_D_2s (for KRAS) based on https://www.nature.com/articles/s41586-024-07205-6 extended table 2
G12V.K_D_2 = 84.8e-9
G12C.K_D_2 = 40.3e-9
G12D.K_D_2 = 317e-9
G12R.K_D_2 = 271e-9
G12A.K_D_2 = 128e-9
G13D.K_D_2 = 342e-9
G13C.K_D_2 = 64.5e-9
Q61H.K_D_2 = 87.2e-9

# K_D_2s (for NRAS) based on https://www.nature.com/articles/s41586-024-07205-6 extended table 3
Q61K.K_D_2 = 72e-9
Q61L.K_D_2 = 238e-9
Q61R.K_D_2 = 237e-9

