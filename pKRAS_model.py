from KRAS_Variant import *

def pRAS_model(t,y,k):

    RAS_GDP=y[0]
    RAS_GTP=y[1]
    RAS_0=y[2]
    Eff=y[3]
    Eff_RAS_GTP=y[4]
    pRAS_GDP=y[5]
    pRAS_GTP=y[6]
    pRAS_0=y[7]
    Eff_pRAS_GTP=y[8]

    GAP = k[26]
    GDP = k[27]
    GTP = k[28]
    GEF = k[29]

    Src = k[30]
    SHP2 = k[31]

    k_Src = k[32]
    Km_Src = k[33]
    k_SHP2 = k[34]
    Km_SHP2 = k[35]

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

    VmaxDp=k[13]*GEF
    VmaxTp=k[14]*GEF*GDP/GTP
    KmDp=k[15]
    KmTp=k[16]
    Vmaxp=k[17]*GAP
    Kmp=k[18]
    kintp=k[19]
    kdissDp=k[20]
    kdissTp=k[21]
    kassDGDPp=k[22]*GDP
    kassTGTPp=k[23]*GTP
    kassEffp=k[24]
    kdissEffp=k[25]

    R1=(VmaxD*RAS_GDP/KmD-VmaxT*RAS_GTP/KmT)/(1+RAS_GDP/KmD+RAS_GTP/KmT+pRAS_GDP/KmDp+pRAS_GTP/KmTp)
    R2=Vmax*RAS_GTP/(Km*(1+pRAS_GTP/Kmp)+RAS_GTP)
    R3=kint*RAS_GTP
    R4=kdissD*RAS_GDP-kassDGDP*RAS_0
    R5=kdissT*RAS_GTP-kassTGTP*RAS_0
    R6=kassEff*RAS_GTP*Eff-kdissEff*Eff_RAS_GTP
    R7=kint*Eff_RAS_GTP

    R8=(VmaxDp*pRAS_GDP/KmDp-VmaxTp*pRAS_GTP/KmTp)/(1+RAS_GDP/KmD+RAS_GTP/KmT+pRAS_GDP/KmDp+pRAS_GTP/KmTp)
    R9=Vmaxp*pRAS_GTP/(Kmp*(1+RAS_GTP/Km)+pRAS_GTP)
    R10=kintp*pRAS_GTP
    R11=kdissDp*pRAS_GDP-kassDGDPp*pRAS_0
    R12=kdissTp*pRAS_GTP-kassTGTPp*pRAS_0
    R13=kassEffp*pRAS_GTP*Eff-kdissEffp*Eff_pRAS_GTP
    R14=kintp*Eff_pRAS_GTP

    R15 = (k_Src*Src*RAS_GTP)/(Km_Src+RAS_GTP)
    R16 = (k_SHP2*SHP2*pRAS_GTP)/(Km_SHP2+pRAS_GDP)

    dydt=[-R1+R2+R3-R4+R7-R15,
        R1-R2-R3-R5-R6+R16,
        R4+R5,
        -R6+R7-R13+R14,
        (R6-R7),
        (-R8+R9+R10-R11+R14-R16),
        (R8-R9-R10-R12-R13+R15),
        (R11+R12),
        (R13-R14)]
    
    print(R1,R2,R3,R4,R5,R6,R7)

    return dydt

def get_params_pRAS(get_param_opts):

    mutant = get_param_opts['mutant']
    fract_mut = get_param_opts['fract_mut']
    GTot = get_param_opts['Total RAS']
    EffTot = get_param_opts['Effector']
    GAP = get_param_opts['GAP']
    GDP = get_param_opts['GDP']
    GTP = get_param_opts['GTP']
    GEF = get_param_opts['GEF']
    Src = get_param_opts['Src']
    SHP2 = get_param_opts['SHP2']

    WT = KRAS_Variant('WT')
    
    WTRasTot=(1-fract_mut)*GTot
    MutRasTot=fract_mut*GTot
    
    y0 = np.zeros(9)

    y0[0]=WTRasTot #RAS:GDP=y(1);
    y0[1]=0 #RAS:GTP=y(2);
    y0[2]=0 #RAS=y(3);
    y0[3]=EffTot #Eff=y(4);
    y0[4]=0 #Eff:RAS:GTP=y(5);
    y0[5]=MutRasTot #pRAS:GDP=y(6);
    y0[6]=0 #pRAS:GTP=y(7);
    y0[7]=0 #pRAS=y(8);
    y0[8]=0 #Eff:pRAS:GTP=y(9);

    k = np.ones(36)

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

    k[30] = Src
    k[31] = SHP2

    k[32] = 1e-1 #k_Src
    k[33] = (1e-4)/250 #Km_Src
    k[34] = 1e-1 #k_SHP2
    k[35] = (1e-4)/250 #Km_SHP2

    return np.array(k),np.array(y0)