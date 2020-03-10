# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 15:19:33 2020

@author: User
"""

import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astroquery.gaia import Gaia
import matplotlib.pyplot as plt

def matchApogeeGaia(OC):
    hdul = fits.open('apogee.fits')
    apogee_data = hdul[1].data

    TEFF = apogee_data["TEFF"]
    TEFF_ERR = apogee_data["TEFF_ERR"]
    apogee_id = apogee_data['APOGEE_ID']
    TM_ID=[]
    Teff=[]
    Teff_err=[]
    for i,ID in enumerate(apogee_id):
        if '2M' in ID and TEFF[i]>0:
            TM_ID.append(ID.split('M')[1])
            Teff.append(TEFF[i])
            Teff_err.append(TEFF_ERR[i])
    apogee_df = pd.DataFrame({'tm_ID':TM_ID,'Teff':Teff,'Teff_err':Teff_err})

    if OC == 'NGC_2682':
        xmatch_data = Table.read('k2_dr2_1arcsec.fits', format='fits')
    else:
        xmatch_data = Table.read('kepler_dr2_1arcsec.fits', format='fits')
    xmatch_df = xmatch_data.to_pandas()
    if OC == 'NGC_2682':
        TWO_MASS_COL = 'tm_name'
    else:
        TWO_MASS_COL = 'tm_designation'
    xmatch_df = xmatch_df[['solution_id','source_id',TWO_MASS_COL]]
    xmatch_df.drop_duplicates(inplace=True)
    xmatch_df[TWO_MASS_COL] = xmatch_df[TWO_MASS_COL].str.decode("utf-8")
    xmatch_tm_ID = []
    for ID in xmatch_df[TWO_MASS_COL]:
        if 'J' in ID:
            xmatch_tm_ID.append(ID.split('J')[1])
        else:
            xmatch_tm_ID.append(None)
    xmatch_df['tm_ID'] = xmatch_tm_ID
    apogee_gaia = pd.merge(xmatch_df, apogee_df, how='left', on='tm_ID')
    print(f'Found {len(apogee_gaia.index)} matching APOGEE and GAIA stars.')

    apogee_Teff = apogee_gaia[['source_id','Teff','Teff_err']].copy()
    apogee_Teff.dropna(subset=['Teff'],inplace=True)
    apogee_Teff.rename(columns={'source_id':'ID'},inplace=True)
    print(f'{len(apogee_Teff.index)} of those stars have Teff data.')
    return apogee_Teff

def load_M67(num=True,array=True):
    data = open("M67 membership.txt", "r").read()
    data = data.split("\n")
    for i in range(len(data)):
        data[i] = data[i].split("  ")[:-1]
        if num==True:
            for j in range(len(data[i])):
                data[i][j] = float(data[i][j])
    if array == True:
        return np.array(data)
    else:
        return data

def load_cluster(OC):
    df = pd.read_csv('GAIA OC membership data.csv')
    OC_list = df['cluster'].unique()
    if OC not in OC_list:
        print("{} - is not a cluster in this data set".format(OC))
    else:
        return df[df['cluster']==OC]

def genSimbad(OC, simbad_file):
    simbad_template = open('SIMBAD_template.txt','r').read().split('\n')
    if OC=='NGC_2682':
        OC_mem = load_M67()
        gaia_IDs = OC_mem[:,0]
        member_df = pd.DataFrame({'ID':gaia_IDs,'ra':OC_mem[:,1],'dec':OC_mem[:,2],'mem_prob':OC_mem[:,3]})
    else:
        OC_mem = load_cluster(OC)
        gaia_IDs = list(OC_mem['ID'])
        member_df = OC_mem[['ID','RAdeg','DEdeg','membership probability']].copy()
        member_df.rename(columns={'RAdeg':'ra','DEdeg':'dec','membership probability':'mem_prob'},inplace=True)
    simbad_input = simbad_template[0]+'\n'
    for star in gaia_IDs:
        simbad_input+=('GAIA DR2 '+str(int(star))+'\n')
    simbad_input+=simbad_template[1]
    f = open(simbad_file, 'w')
    f.write(simbad_input)
    f.close()
    print(f'There is {len(gaia_IDs)} stars in the membership paper.')
    print('Simbad input txt file saved as "{}".'.format(simbad_file))
    return gaia_IDs, member_df

def GAIA_query(ids):
    print('Launching GAIA query...')
    if type(ids[0]) is not str:
        ids = [str(int(i)) for i in ids]
    query_ids = ",".join(ids)
    r= Gaia.launch_job("SELECT TOP 5000 gaia_source.source_id, gaia_source.phot_g_mean_mag, \
    gaia_source.phot_g_mean_flux, gaia_source.phot_g_mean_flux_error, gaia_source.a_g_val, \
    gaia_source.lum_val, gaia_source.lum_percentile_lower, gaia_source.lum_percentile_upper, \
    dist.source_id as dist_id, dist.r_est, dist.r_lo, dist.r_hi \
    FROM gaiadr2.gaia_source, external.gaiadr2_geometric_distance as dist \
    WHERE gaia_source.source_id=dist.source_id and gaia_source.source_id in ({});".format(query_ids))
    r = r.get_results()
    g_mag_err = 1.085736205*np.array(r['phot_g_mean_flux_error'])/np.array(r['phot_g_mean_flux'])
    lum_err = np.maximum(np.array(r['lum_val']-r['lum_percentile_lower']),np.array(r['lum_percentile_upper']-r['lum_val']))
    dist_err = np.maximum(np.array(r['r_est']-r['r_lo']),np.array(r['r_hi']-r['r_est']))
    df = pd.DataFrame({'ID':r['source_id'],'g_mag':r['phot_g_mean_mag'],'g_mag_err':g_mag_err,'A_G':r['a_g_val'],
                       'gaia_lum':r['lum_val'], 'gaia_lum_err':lum_err,'distance':r['r_est'],'distance_err':dist_err})
    print('Obtained {} out of {} stars in the cluster.'.format(len(df.index),len(ids)))
    return df

def mergeApogee(gaia_df, apogee_df):
    print('Matching APOGEE Teffs to queried GAIA stars')
    merged_df = pd.merge(gaia_df, apogee_df, how='left', on='ID')
    ref = []
    count = 0
    for star in merged_df['Teff']:
        if star>0:
            ref.append('apogee')
            count+=1
        else:
            ref.append(np.nan)
    merged_df['ref'] = ref
    print(f'Found {count} matching stars with APOGEE Teff data.')
    return merged_df

def readSimbad(file, df):
    print('Reading SIMBAD output and merging it to GAIA+APOGEE data...')
    collected_data = open(file,'r').read().split(':\n')[-1]
    first_lines = collected_data.split('\nGAIA')
    lines = []
    for line in first_lines:
        lines+=line.split('\nGaia')
    lines = lines[1:]
    print('found',len(lines),'matching IDs')

    all_data = []
    for i,line in enumerate(lines):
        if 'DR2' in line and 'Fe_H' in line:
            ID, feh_list = line.split('|Fe_H')
            ID = ID.split(' ')[2]
            data_dict={}
            data_dict['ID'] = int(ID)

            if '\n' in feh_list:
                feh_list=feh_list.split('\n')
            else:
                feh_list = [feh_list]
            feh_array = []
            for feh in feh_list:
                feh = feh.split('|')[1:-1]
                feh = [fehi.strip() for fehi in feh]
                try: a=len(feh[0])
                except IndexError: print(line,feh)
                if len(feh[0])>0:
                    if ' ' in feh[0]:
                        Teff = float(feh[0].split(' ')[0])
                        if Teff<1000:
                            continue
                    else:
                        if float(feh[0])>1000:
                            Teff = float(feh[0])
                        else: continue
                else: continue
                if len(feh[1])>0:
                    try: FeH = float(feh[1])
                    except ValueError: FeH = None
                else:
                    FeH = None

                ref = feh[-1]
                if float(ref[:4])<2000:
                    continue
                ref_list = []
                if len(feh_array)>0:
                    for f in feh_array:
                        ref_list.append(f[2])
                    if ref in ref_list:
                        continue
                feh_array.append([Teff, FeH, ref])

            if len(feh_array)==0:
                continue
            data_dict['feh_table'] = feh_array
            all_data.append(data_dict)

    print(len(all_data),'stars have Teff data')
    simbad_df = pd.DataFrame(all_data)
    length=[]
    for i in list(simbad_df['feh_table']):
        length.append(len(i))
    simbad_df['feh_count'] = length
    return_df = pd.merge(df, simbad_df, how='left', on='ID')

    indicator = []
    for i,star in return_df.iterrows():
        if np.isnan(star['Teff'])==True and np.isnan(star['feh_count'])==True:
            indicator.append(np.nan)
        else: indicator.append('keep')
    return_df['indicator'] = indicator
    return_df.dropna(subset=['indicator'],inplace=True)
    return_df.drop(columns=['indicator'],inplace=True)
    print('Found Teff for',len(return_df.index),'stars between SIMBAD and APOGEE.')
    return return_df

def membershipCut(df, member_df):
    print('Matching in membership data and applying membership cut...')
    return_df = pd.merge(df, member_df, how='left', on='ID')
    mem_prob = np.array(return_df['mem_prob'])
    gaia_ID = np.array(return_df['ID'])
    index = np.argsort(mem_prob)[::-1]
    mem_prob = mem_prob[index]
    gaia_ID = gaia_ID[index]
    selected_ID = []
    total_prob=0
    for i,prob in enumerate(mem_prob):
        current_prob = (total_prob+prob)/int(i+1)
        if current_prob>=0.99:
            total_prob+=prob
            selected_ID.append(gaia_ID[i])
        else: break
    print(f'{len(selected_ID)}/{len(gaia_ID)} stars are taken as cluster members')
    return return_df[return_df['ID'].isin(selected_ID)]

def countUniqueRefs(df, column='feh_table'):
    ref_list = []
    if column=='feh_table':
        for i in df['feh_table']:
            refs = []
            for j in i:
                refs.append(j[2])
            refs=list(set(refs))
            ref_list+=refs
    elif column=='ref':
        ref_list = df['ref']
    ref_set = set(ref_list)
    ref_count = np.zeros(len(ref_set))
    for i in ref_list:
        for j,r in enumerate(ref_set):
            if i==r:
                ref_count[j]+=1
    ref_df = pd.DataFrame({'ref':list(ref_set),'count':ref_count})
    ref_df.sort_values('count',inplace=True,ascending=False,ignore_index=True)
    return ref_df

def replaceByRef(df, target_ref):
    non_hits = []
    for i,star in df.iterrows():
        if star['Teff']>0:
            continue
        else:
            hit = False
            for paper in star['feh_table']:
                if target_ref == paper[2]:
                    df['Teff'][i] = paper[0]
                    df['ref'][i] = paper[2]
                    hit=True
                    break
            if hit == False:
                non_hits.append(star['ID'])
    return df, non_hits

def replaceManual(df, ranks):
    count=0
    for i,star in df.iterrows():
        count+=1
        print('Star {}/{}, GAIA ID: {}'.format(count,len(df.index),star['ID']))
        Teff = []
        ref = []
        for entry in star['feh_table']:
            Teff.append(entry[0])
            ref.append(entry[2])
        median = np.median(Teff)
        diff = np.array(Teff)-median
        for j,entry in enumerate(star['feh_table']):
            entry_no = int(ranks[ranks['ref']==ref[j]]['count'])
            if diff[j]>90:
                print('\033[1m'+str(j)+': '+str(Teff[j])+' '+ref[j]+' '+str(entry_no)+' entries'+'\033[0m')
            else:
                print(str(j)+': '+str(Teff[j])+' '+ref[j]+' '+str(entry_no)+' entries')
        decision_index = int(input('Teff choice?'))
        df['Teff'][i] = Teff[decision_index]
        df['ref'][i] = ref[decision_index]
        print()
    return df

def assignTeff(df, max_diff_from_median=90):
    #seperate apogee stars from rest
    apogee_stars = df[df['ref']=='apogee']
    other_stars = df[df['ref']!='apogee']

    #assign Teff and ref for stars that only have one source of Teff
    singulars = other_stars[other_stars['feh_count']==1].copy()
    multiples = other_stars[other_stars['feh_count']>1].copy()
    Teff = []
    ref = []
    for i,star in singulars.iterrows():
        Teff.append(star['feh_table'][0][0])
        ref.append(star['feh_table'][0][2])
    singulars['Teff'] = Teff
    singulars['ref'] = ref
    print(f'Using APOGEE Teff for {len(apogee_stars.index)} stars.')
    print(f'There are {len(singulars.index)} stars without APOGEE data and have Teff provided by only one paper.')
    
    #assign Teff and ref for stars that have multiple sources but sources agree with each other
    #criteria = max(source_Teff-median_Teff)<90
    #paper is chosen by number of stars it supply. The paper that supplies most stars gets used first
    problematic_IDs=[]
    regular_IDs=[]
    for i,star in multiples.iterrows():
        Teffs = []
        for paper in star['feh_table']:
            Teffs.append(paper[0])
        median = np.median(Teffs)
        diff = np.array(Teffs)-median
        if max(diff)>max_diff_from_median:
            problematic_IDs.append(star['ID'])
        else: regular_IDs.append(star['ID'])
    print(f'Picking Teff automatically for {len(regular_IDs)} stars, {len(problematic_IDs)} stars require manual picking.')
    problematic_df = multiples[multiples['ID'].isin(problematic_IDs)].copy()
    regular_df = multiples[multiples['ID'].isin(regular_IDs)].copy()

    rank = countUniqueRefs(regular_df)
    ref_length=np.sum(rank['count'])
    ranks=[]
    while ref_length>0:
        index = np.where(rank['count']==rank['count'][0])[0]
        if len(index)>1:
            papers = np.array(rank['ref'])[index]
            years = []
            for p in papers:
                years.append(int(p[:4]))
            target_paper = papers[np.argmax(years)]
        else:
            target_paper = rank['ref'][0]

        regular_df,other_ID = replaceByRef(regular_df,target_paper)
        rank = countUniqueRefs(regular_df[regular_df['ID'].isin(other_ID)])
        ranks.append(rank)
        ref_length = np.sum(rank['count'])
    
    final_ranks = countUniqueRefs(pd.concat([regular_df,singulars]),column='ref')
    return pd.concat([apogee_stars, singulars,regular_df]), problematic_df, final_ranks

def extinction(OC,dist_mod_x,GAIA_g_band_extinction_coeff = 2.294):
    data = open("{}/{}_dustmap.txt".format(OC,OC), "r").read()
    data = data.split("\n")
    min_rel = float(data[9].split(": ")[-1])
    max_rel = float(data[10].split(": ")[-1])
    dist_mod_list = data[26].split("|")[-1].strip().split()
    redenning = data[28].split("|")[-1].strip().split()
    for i in range(len(redenning)-1):
        dist_mod_list[i] = float(dist_mod_list[i])
        redenning[i] = float(redenning[i])

    if dist_mod_x < min_rel or dist_mod_x > max_rel:
        print("WARNING: distance modulus {} exceeds reliable redenning range.".format(dist_mod_x))
    for i in range(len(redenning)):
        if dist_mod_x >= dist_mod_list[i] and dist_mod_x < dist_mod_list[i+1]:
            dist_mod_slope = (redenning[i+1]-redenning[i])/(dist_mod_list[i+1]-dist_mod_list[i])
            return (dist_mod_slope*(dist_mod_x-dist_mod_list[i])+redenning[i])*GAIA_g_band_extinction_coeff
    print("bad distance modulus: {}".format(dist_mod_x))
          
def BC_G(Teff,Teff_err):
    Teff_Sun = 5778
    a0_48 = [6.000E-02, 2.634E-02]
    a1_48 = [6.731E-05, 2.438E-05]
    a2_48 = [-6.647E-08, -1.129E-09]
    a3_48 = [2.859E-11, -6.722E-12]
    a4_48 = [-7.197E-15, 1.635E-15]
    a0_34 = [1.749, -2.487]
    a1_34 = [1.977E-03, -1.876E-03]
    a2_34 = [3.737E-07, 2.128E-07]
    a3_34 = [-8.966E-11, 3.807E-10]
    a4_34 = [-4.183E-14, 6.570E-14]

    BC = []
    BC_err = []
    for i in range(len(Teff)):
        #Teff should be an array or list of floats, which are stellar effective tempeartures in kelvin
        #Between 3300k-8000k

        #returns np array of bolometric corrections
        Teff_val = Teff[i] - Teff_Sun
        if Teff[i] >= 3300 and Teff[i] < 4000:
            BC.append(a0_34[0]+(a1_34[0]*Teff_val)+(a2_34[0]*Teff_val**2)+(a3_34[0]*Teff_val**3)+(a4_34[0]*Teff_val**4))
            BC_err.append(((a0_34[1]**2)+(Teff_err[i]**2)*((a1_34[0]+Teff_val*(2*a2_34[0]+Teff_val*(3*a3_34[0]+4*a4_34[0]*Teff_val)))**2)+((a1_34[1]**2)*(Teff_val**2))+((a2_34[1]**2)*(Teff_val**4))+((a3_34[1]**2)*(Teff_val**6))+((a4_34[1]**2)*(Teff_val**8)))**0.5)     
        elif Teff[i] >= 4000 and Teff[i] <= 8000:
            BC.append(a0_48[0]+(a1_48[0]*Teff_val)+(a2_48[0]*Teff_val**2)+(a3_48[0]*Teff_val**3)+(a4_48[0]*Teff_val**4))
            BC_err.append(((a0_48[1]**2)+(Teff_err[i]**2)*((a1_48[0]+Teff_val*(2*a2_48[0]+Teff_val*(3*a3_48[0]+4*a4_48[0]*Teff_val)))**2)+((a1_48[1]**2)*(Teff_val**2))+((a2_48[1]**2)*(Teff_val**4))+((a3_48[1]**2)*(Teff_val**6))+((a4_48[1]**2)*(Teff_val**8)))**0.5)
        else:
            print("Invalid temperature {}. BC has been set to Nan".format(Teff[i]))
            BC.append(np.nan)
            BC_err.append(np.nan)
    return np.array(BC), np.array(BC_err)
          
def lum_calc(OC,g,g_err,r,r_err,Teff,Teff_err,Mbol_Sun = 4.75):
    #g = array of apparent GAIA G-band magnitude
    #r = median distance of cluster in pc.
    #Teff = array of effective temperatures
    dist_mod = -1*(5 - 5*np.log10(r))
    dist_mod_err = 5*(r_err/r)/np.log(10)
    A_G = extinction(OC, dist_mod_x=dist_mod)
    #we assume the error on A_G is 0
    BC,BC_err = BC_G(Teff,Teff_err)
    M_G = g - dist_mod-A_G #array of absolute GAIA G-band magnitudes
    M_G_err = (g_err**2+dist_mod_err**2)**0.5
    L = (10**(-0.4*(M_G+BC-Mbol_Sun))) #array of stellar luminosities in units of solar luminosities
    L_err = 0.4*np.log(10)*L*((M_G_err**2)+(BC_err)**2)**0.5
    return L,L_err

def matchTeff_err(df, paper, file, ra_col, dec_col, Teff_col, Teff_err_col):
    if paper not in df['ref'].unique():
        raise NameError("No such paper found!")
    paper_data = open(file,'r').read().split('\n')[1:]
    ra = []
    dec = []
    Teff = []
    Teff_err = []
    for line in paper_data:
        line_data = line.split()
        if ra_col is not None:
            ra.append(float(line_data[ra_col]))
        else: ra.append(np.nan)
        if dec_col is not None:
            dec.append(float(line_data[dec_col]))
        else: dec.append(np.nan)
        Teff.append(float(line_data[Teff_col]))
        Teff_err.append(float(line_data[Teff_err_col]))
    paper_df = pd.DataFrame({'ra':ra,'dec':dec,'Teff':Teff,'Teff_err':Teff_err})
    
    for i,star in df.iterrows():
        if star['ref']==paper:
            count = Teff.count(star['Teff'])
            if count==1:
                for j,t in enumerate(Teff):
                    if star['Teff']==t:
                        df['Teff_err'][i]=Teff_err[j]
                        break
            elif count>1:
                if ra_col is not None and dec_col is not None:
                    suspect_df = paper_df[paper_df['Teff']==star['Teff']]
                    angular_dist = np.sqrt((suspect_df['ra']-star['ra'])**2+(suspect_df['dec']-star['dec'])**2)
                    df['Teff_err'][i] = list(suspect_df['Teff_err'])[np.argmin(angular_dist)]
                else:
                    print("Warning: Found {} matching Teff for {} of Teff {} and there is no ra/dec columns".format(count,star['ID'],star['Teff']))
            else:
                print("Warning: Found 0 matching Teff for {} of Teff {}, using up and down 1K seach".format(star['ID'],star['Teff']))
                suspect_df = paper_df[paper_df['Teff'].between(star['Teff']-1,star['Teff']+1)]
                if len(suspect_df.index)==1:
                    df['Teff_err'][i] = suspect_df['Teff_err']
                elif len(suspect_df.index)>1:
                    angular_dist = np.sqrt((suspect_df['ra']-star['ra'])**2+(suspect_df['dec']-star['dec'])**2)
                    df['Teff_err'][i] = list(suspect_df['Teff_err'])[np.argmin(angular_dist)]
                else:
                    print("Warning: Up and down 1K search failed, no data for this star.")
    return df

def fixedTeff_err(df, paper, Teff_err):
    for i,star in df.iterrows():
        if star['ref']==paper:
            df['Teff_err'][i]=Teff_err
    return(df)
    
def plotCluster(df, plot_gaia=False, plot_train=False, grid_path='grid2_early.csv'):
    fig, ax=plt.subplots(1,1, figsize=(10,10))
    log_cal_lum_err = df['cal_lum_err']/(df['cal_lum']*np.log(10))
    log_Teff_err = df['Teff_err']/(df['Teff']*np.log(10))
    if plot_gaia == True:
        log_gaia_lum_err = df['gaia_lum_err']/(df['gaia_lum']*np.log(10))
        ax.errorbar(np.log10(df['Teff']), np.log10(df['gaia_lum']), xerr=log_Teff_err, yerr=log_gaia_lum_err, fmt='.', zorder=1, c='grey', label='gaia luminosity')
    if plot_train == True:
        train_df = pd.read_csv('grid2_early.csv')
        points = train_df[:100000]
        ax.scatter(np.log10(points['effective_T']),np.log10(points['luminosity']), s=5, zorder=0, c='lightgrey',label='training data')
    ax.errorbar(np.log10(df['Teff']), np.log10(df['cal_lum']), xerr=log_Teff_err, yerr=log_cal_lum_err, fmt='.', zorder=2, c='black')
    ax.scatter(np.log10(df['Teff']), np.log10(df['cal_lum']), s=15, zorder=3, c='blue', label='computed luminosity')
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_xlabel(r'$\log10 T_{eff}$')
    ax.set_ylabel(r'$\log10(L/L_{\odot})$')
    ax.legend()
    ax.grid()
    plt.show()
    
def removeStars(df, logTeff, logL, N):
    logTeffs = np.array(np.log10(df['Teff']))
    logLs = np.array(np.log10(df['cal_lum']))
    IDs = np.array(df['ID'])
    diff = []
    for i,logTeffi in enumerate(logTeffs):
        diff.append(np.sqrt((2.5*(logTeffi-logTeff))**2+(logLs[i]-logL)**2))
    sort_index = np.argsort(diff)
    sorted_ID = IDs[sort_index]
    drop_list = sorted_ID[:N]
    df.drop(index=df[df['ID'].isin(drop_list)].index, inplace=True)
    return df

def ra_dec_conv(s,r):
    s = s.split(" ")
    for i in range(len(s)):
              s[i] = float(s[i])
              if i == 0:
                        s[i] = s[i]*15
              elif i == 1:
                        s[i] = (s[i]*15)/60
              elif i == 2:
                        s[i] = (s[i]*15)/3600
    
    r = "+11 48 00"
    r = r.split(" ")
    for i in range(len(r)):
              r[i] = float(r[i])
              if i == 1:
                        r[i] = r[i]/60
              elif i == 2:
                        r[i] = r[i]/3600
    print(sum(s),sum(r))
    