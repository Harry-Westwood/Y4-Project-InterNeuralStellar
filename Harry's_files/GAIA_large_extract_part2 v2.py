import numpy as np
import json
import csv
import copy

# 'NGC_1027', 'NGC_103', 'NGC_1039', 'NGC_1193', 'NGC_1220', 'NGC_1245', 'NGC_129', 'NGC_1333', 'NGC_1342', 'NGC_1348', 'NGC_136', 'NGC_1444', 'NGC_146', 'NGC_1496', 'NGC_1502', 'NGC_1513', 'NGC_1528', 'NGC_1545', 'NGC_1579', 'NGC_1582', 'NGC_1605', 'NGC_1624', 'NGC_1647', 'NGC_1662', 'NGC_1664', 'NGC_1708', 'NGC_1724', 'NGC_1750', 'NGC_1758', 'NGC_1778', 'NGC_1798', 'NGC_1817', 'NGC_1857', 'NGC_188', 'NGC_1883', 'NGC_189', 'NGC_1893', 'NGC_1901', 'NGC_1907', 'NGC_1912', 'NGC_1960', 'NGC_2099', 'NGC_2112', 'NGC_2129', 'NGC_2141', 'NGC_2158', 'NGC_2168', 'NGC_2183', 'NGC_2184', 'NGC_2186', 'NGC_2192', 'NGC_2194', 'NGC_2204', 'NGC_2215', 'NGC_2225', 'NGC_2232', 'NGC_2236', 'NGC_2243', 'NGC_2244', 'NGC_225', 'NGC_2251', 'NGC_2254', 'NGC_2259', 'NGC_2262', 'NGC_2264', 'NGC_2266', 'NGC_2269', 'NGC_2281', 'NGC_2286', 'NGC_2287', 'NGC_2301', 'NGC_2302', 'NGC_2304', 'NGC_2309', 'NGC_2311', 'NGC_2318', 'NGC_2323', 'NGC_2324', 'NGC_2335', 'NGC_2343', 'NGC_2345', 'NGC_2353', 'NGC_2354', 'NGC_2355', 'NGC_2358', 'NGC_2360', 'NGC_2362', 'NGC_2367', 'NGC_2374', 'NGC_2383', 'NGC_2396', 'NGC_2401', 'NGC_2414', 'NGC_2420', 'NGC_2421', 'NGC_2422', 'NGC_2423', 'NGC_2425', 'NGC_2428', 'NGC_2432', 'NGC_2437', 'NGC_2439', 'NGC_2447', 'NGC_2448', 'NGC_2451A', 'NGC_2451B', 'NGC_2453', 'NGC_2455', 'NGC_2477', 'NGC_2482', 'NGC_2489', 'NGC_2506', 'NGC_2509', 'NGC_2516', 'NGC_2527', 'NGC_2533', 'NGC_2539', 'NGC_2546', 'NGC_2547', 'NGC_2548', 'NGC_2567', 'NGC_2571', 'NGC_2580', 'NGC_2587', 'NGC_2588', 'NGC_2627', 'NGC_2632', 'NGC_2635', 'NGC_2645', 'NGC_2658', 'NGC_2659', 'NGC_2660', 'NGC_2669', 'NGC_2670', 'NGC_2671', 'NGC_2682', 'NGC_2818', 'NGC_2849', 'NGC_2866', 'NGC_2910', 'NGC_2925', 'NGC_2972', 'NGC_3033', 'NGC_3105', 'NGC_3114', 'NGC_3228', 'NGC_3255', 'NGC_3293', 'NGC_3324', 'NGC_3330', 'NGC_3496', 'NGC_3532', 'NGC_3572', 'NGC_3590', 'NGC_3603', 'NGC_366', 'NGC_3680', 'NGC_3766', 'NGC_381', 'NGC_3960', 'NGC_4052', 'NGC_4103', 'NGC_433', 'NGC_4337', 'NGC_4349', 'NGC_436', 'NGC_4439', 'NGC_4463', 'NGC_457', 'NGC_4609', 'NGC_4755', 'NGC_4815', 'NGC_4852', 'NGC_5138', 'NGC_5168', 'NGC_5269', 'NGC_5281', 'NGC_5288', 'NGC_5316', 'NGC_5381', 'NGC_5460', 'NGC_559', 'NGC_5593', 'NGC_5606', 'NGC_5617', 'NGC_5662', 'NGC_5715', 'NGC_5749', 'NGC_5764', 'NGC_581', 'NGC_5822', 'NGC_5823', 'NGC_5925', 'NGC_5999', 'NGC_6005', 'NGC_6025', 'NGC_6031', 'NGC_6067', 'NGC_6087', 'NGC_609', 'NGC_6124', 'NGC_6134', 'NGC_6152', 'NGC_6167', 'NGC_6178', 'NGC_6192', 'NGC_6193', 'NGC_6204', 'NGC_6208', 'NGC_6216', 'NGC_6231', 'NGC_6242', 'NGC_6249', 'NGC_6250', 'NGC_6253', 'NGC_6259', 'NGC_6268', 'NGC_6281', 'NGC_6318', 'NGC_6322', 'NGC_6357', 'NGC_637', 'NGC_6383', 'NGC_6396', 'NGC_6400', 'NGC_6404', 'NGC_6405', 'NGC_6416', 'NGC_6425', 'NGC_6451', 'NGC_6469', 'NGC_6475', 'NGC_6494', 'NGC_6520', 'NGC_6531', 'NGC_654', 'NGC_6561', 'NGC_6568', 'NGC_6583', 'NGC_659', 'NGC_6603', 'NGC_6611', 'NGC_6613', 'NGC_663', 'NGC_6631', 'NGC_6633', 'NGC_6645', 'NGC_6649', 'NGC_6664', 'NGC_6694', 'NGC_6704', 'NGC_6705', 'NGC_6709', 'NGC_6716', 'NGC_6728', 'NGC_6735', 'NGC_6755', 'NGC_6756', 'NGC_6791', 'NGC_6793', 'NGC_6800', 'NGC_6802', 'NGC_6811', 'NGC_6819', 'NGC_6823', 'NGC_6827', 'NGC_6830', 'NGC_6834', 'NGC_6846', 'NGC_6866', 'NGC_6871', 'NGC_6910', 'NGC_6913', 'NGC_6939', 'NGC_6940', 'NGC_6991', 'NGC_6997', 'NGC_7024', 'NGC_7031', 'NGC_7039', 'NGC_7044', 'NGC_7058', 'NGC_7062', 'NGC_7063', 'NGC_7067', 'NGC_7082', 'NGC_7086', 'NGC_7092', 'NGC_7128', 'NGC_7129', 'NGC_7142', 'NGC_7160', 'NGC_7209', 'NGC_7226', 'NGC_7235', 'NGC_7243', 'NGC_7245', 'NGC_7261', 'NGC_7281', 'NGC_7296', 'NGC_7380', 'NGC_7419', 'NGC_7423', 'NGC_743', 'NGC_744', 'NGC_7510', 'NGC_752', 'NGC_7654', 'NGC_7762', 'NGC_7788', 'NGC_7789', 'NGC_7790', 'NGC_869', 'NGC_884', 'NGC_957', 'Negueruela_1', 'Patchick_3', 'Patchick_75', 'Patchick_90', 'Patchick_94', 'Pfleiderer_3', 'Pismis_1', 'Pismis_11', 'Pismis_12', 'Pismis_15', 'Pismis_16', 'Pismis_17', 'Pismis_18', 'Pismis_19', 'Pismis_2', 'Pismis_20', 'Pismis_21', 'Pismis_22', 'Pismis_23', 'Pismis_27', 'Pismis_3', 'Pismis_4', 'Pismis_5', 'Pismis_7', 'Pismis_8', 'Pismis_Moreno_1', 'Platais_10', 'Platais_3', 'Platais_8', 'Platais_9', 'Pozzo_1', 'RSG_1', 'RSG_5', 'RSG_7', 'RSG_8', 'Riddle_4', 'Roslund_2', 'Roslund_3', 'Roslund_4', 'Roslund_5', 'Roslund_6', 'Roslund_7', 'Ruprecht_1', 'Ruprecht_10', 'Ruprecht_100', 'Ruprecht_101', 'Ruprecht_102', 'Ruprecht_105', 'Ruprecht_107', 'Ruprecht_108', 'Ruprecht_111', 'Ruprecht_112', 'Ruprecht_115', 'Ruprecht_117', 'Ruprecht_119', 'Ruprecht_121', 'Ruprecht_126', 'Ruprecht_127', 'Ruprecht_128', 'Ruprecht_130', 'Ruprecht_134', 'Ruprecht_135', 'Ruprecht_138', 'Ruprecht_143', 'Ruprecht_144', 'Ruprecht_145', 'Ruprecht_147', 'Ruprecht_148', 'Ruprecht_151', 'Ruprecht_16', 'Ruprecht_161', 'Ruprecht_164', 'Ruprecht_167', 'Ruprecht_170', 'Ruprecht_171', 'Ruprecht_172', 'Ruprecht_174', 'Ruprecht_176', 'Ruprecht_18', 'Ruprecht_19', 'Ruprecht_23', 'Ruprecht_24', 'Ruprecht_25', 'Ruprecht_26', 'Ruprecht_27', 'Ruprecht_28', 'Ruprecht_29', 'Ruprecht_32', 'Ruprecht_33', 'Ruprecht_34', 'Ruprecht_35', 'Ruprecht_36', 'Ruprecht_37', 'Ruprecht_4', 'Ruprecht_41', 'Ruprecht_42', 'Ruprecht_43', 'Ruprecht_44', 'Ruprecht_45', 'Ruprecht_47', 'Ruprecht_48', 'Ruprecht_50', 'Ruprecht_54', 'Ruprecht_58', 'Ruprecht_60', 'Ruprecht_61', 'Ruprecht_63', 'Ruprecht_66', 'Ruprecht_67', 'Ruprecht_68', 'Ruprecht_71', 'Ruprecht_75', 'Ruprecht_76', 'Ruprecht_77', 'Ruprecht_78', 'Ruprecht_79', 'Ruprecht_82', 'Ruprecht_83', 'Ruprecht_84', 'Ruprecht_85', 'Ruprecht_91', 'Ruprecht_93', 'Ruprecht_94', 'Ruprecht_96', 'Ruprecht_97', 'Ruprecht_98'
#OC = "NGC_6819"
OC = "NGC_2682"
des_error = 0.01
def load_csv(file):
          data = []
          with open(file, 'r') as f:
                    reader = csv.reader(f)
                    for i in reader:
                              data.append(i)
          return copy.deepcopy(data[0]), data[1:]

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

def GAIA_search(ra,dec,x_width,y_width):
          import astropy.units as u
          from astropy.coordinates import SkyCoord
          from astroquery.gaia import Gaia
          coord = SkyCoord(ra,dec, unit=(u.degree, u.degree), frame='icrs')
          width = u.Quantity(x_width, u.deg)
          height = u.Quantity(y_width, u.deg)
          r = Gaia.query_object_async(coordinate=coord, width=width, height=height)
          ddict = {}
          for i in ["designation","source_id",'phot_g_mean_flux','phot_g_mean_flux_error','rv_template_teff','teff_val','radius_val','lum_val','rv_template_fe_h']:
                    try:
                              ddict[i] = r[i]
                    except:
                              print("{} did bad".format(i))
          return ddict

def bol_mag(mv, Teff):
          d = 857 #pc, though estimated between 800-900pc
          MV = mv - 2.5*np.log10((d/10)**2)
          BC = -8.499*(np.log10(Teff) - 4)**4 + 13.421*(np.log10(Teff) - 4)**3 - 8.131*(np.log10(Teff) - 4)**2 - 3.901*(np.log10(Teff) - 4) - 0.438
          Mbol = MV + BC
          return Mbol

def lum_calc(mv,Teff):
          bolmag = bol_mag(mv, Teff)
          bolmag_sun = 4.75
          lum = 10**((bolmag_sun-bolmag)/2.5)
          return lum

def step1_create_files():
          headers = ["ID","RAdeg","DEdeg","parallax","parallax error","membership probability","cluster"]
          data = []
          clusters = []
          OC_data = []
          cluster_ids = []
          data = data[1:]
          clusters = clusters[1:]
          clusters = sorted(list(set(clusters)))
          with open("GAIA OC membership data.csv", 'r') as f:
             reader = csv.reader(f)
             for i in reader:
                       data.append(i)
                       clusters.append(i[-1])

          if OC in clusters:
                    for i in range(len(data)):
                              if data[i][-1] == OC:
                                        #data[i][0] = "GAIA DR2 "+data[i][0]
                                        OC_data.append(data[i])
                                        cluster_ids.append("GAIA DR2 "+data[i][0])
                    with open("GAIA step1 data {}.csv".format(OC), "w", newline="") as f:
                              writer = csv.writer(f)
                              writer.writerows([headers]+OC_data)
                    f = open("{} members.txt".format(OC), 'w')
                    for i in cluster_ids:
                              f.write(i+"\n")
                    f.close()
                    #with open("{} members.txt".format(OC), 'w') as f:
                    #              json.dump(cluster_ids, f)
                    
          else:
                    print("{} - is not a cluster in this data set".format(OC))

def step2_membership():
          if OC == "NGC_2682":
                    GAIA = load_M67()
          else:
                    headers_GAIA, GAIA = load_csv(file="GAIA step1 data {}.csv".format(OC))
          headers_SIM, SIM = load_csv(file="{} SIMBAD.csv".format(OC))
          headers_SIM = headers_SIM+["computed luminosity","membership probability"]
          for i in range(len(GAIA)):
                    for j in range(len(SIM)):
                              if str(int(GAIA[i][0])) == SIM[j][0]:
                                        if OC == "NGC_2682":
                                                  SIM[j] = SIM[j]+[GAIA[i][-1]]
                                        else:
                                                  SIM[j] = SIM[j]+[GAIA[i][-2]] #appends membership probability
                                        #############add more columns here if you wanna
                                        ##don't forget to add any columns to the headers_SIM variable
          SIM.sort(key=lambda x: x[-1])
          SIM.reverse()
          pm = []
          mv = []
          Teff = []
          for i in range(len(SIM)):
                    mv.append(SIM[i][3])
                    Teff.append(SIM[i][7])
                    pm.append(1-float(SIM[i][-1]))
          star_slice = None
          for i in range(len(pm)-1):
                    error_perc = np.sum(pm[:i+1])/len(pm[:i+1])
                    if error_perc > des_error:
                              star_slice = i
                              break
          if star_slice != None:
                    SIM = SIM[:star_slice]              
          lum = list(lum_calc(mv=np.array(mv,dtype=float),Teff=np.array(Teff,dtype=float)))
          for i in range(len(SIM)):
                    SIM[i] = SIM[i][:-1]+[lum[i]]+[SIM[i][-1]]

          with open("GAIA OC membership {}.csv".format(OC), "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows([headers_SIM]+SIM)
          
                                        
       
#step1_create_files()
step2_membership()
#ddict = GAIA_search(ra=132.84977065044, dec=11.58065417105,x_width=0.1,y_width=0.1)        

#RA = 0-360
#DEC = -90 - 90

