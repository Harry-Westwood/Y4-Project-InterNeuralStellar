import numpy as np
import json
import csv


file = "M67 membership.txt"
#ID: Gaia-DR2 identifier,
#R.A.(°)
#Decl. (°)
#PRF = RF membership probability.

def load_M67(file,num=True,array=True):
          data = open(file, "r").read()
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
                              
data = load_M67(file=file)
des_error = 0.01
mship = 1-data[:,-1]
for i in range(len(mship)-1):
          error_perc = np.sum(mship[:i+1])
          if error_perc >= des_error:
                    star_slice = i+1
                    break
          #print(error_perc)

strdata = load_M67(file=file,num=False,array=False)

#for i in strdata[:star_slice]:
#          print("query id GAIA DR2 "+i[0])
          
#with open('conf_M67_IDs.txt', 'w') as f:
#          json.dump(data, f)
count3 = 0
for i in range(len(strdata)):
          if float(strdata[i][-1]) == 1.0:
                   count3+=1
          strdata[i][-1] = 1-float(strdata[i][-1])
#vals = np.genfromtxt("SIMBAD_output.csv",skip_header=1)
vals = []
with open("SIMBAD_output_recent.csv", 'r') as f:
   reader = csv.reader(f)
   for i in reader:
             vals.append(i)
headers = vals[0]+["membership probability","ra","dec"]
vals2 = vals[1:]

prfs = 0
for i in range(len(strdata)):
          for j in range(len(vals2)):
                    if strdata[i][0] in vals2[j][0]:
                              prfs+=  strdata[i][-1]
                              vals2[j] = vals2[j]+[(1-strdata[i][-1]),strdata[i][1],strdata[i][2]]
                    
expected_error = prfs*len(vals2)
#print(expected_error,prfs,len(vals2))

vals2 = [headers]+vals2

count = 0
count2 = 0

for i in range(len(strdata)):
          if strdata[i][-1] == 0.0:
                    count2+=1
          for j in range(1,len(vals)):
                    if strdata[i][0] in vals[j][0]:
                              #print(strdata[i][-1])
                              count+=1

ra_centre = (max(data[:,1])+min(data[:,1]))/2
ra_rad = max(data[:,1])-ra_centre
dec_centre = (max(data[:,2])+min(data[:,2]))/2
dec_rad = max(data[:,2])-dec_centre
print(ra_centre, ra_rad)
print(dec_centre, dec_rad)



with open("SIMBAD_output3.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(vals2)


          


