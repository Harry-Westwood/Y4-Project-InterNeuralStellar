a = "http://simbad.u-strasbg.fr/simbad/sim-id?Ident=%401128235&Name=NGC%20%202682%20%20%20128&submit=submit"
#http://simbad.u-strasbg.fr/simbad/sim-id?Ident=%401128235&Name=NGC++2682+++128&submit=display+all+measurements#lab_meas

#a="http://simbad.u-strasbg.fr/simbad/sim-id?Ident=%401130581&Name=Cl*%20NGC%202682%20%20%20SAND%20%20%20%20%20859&submit=submit"
#http://simbad.u-strasbg.fr/simbad/sim-id?Ident=%401130581&Name=Cl*+NGC+2682+++SAND+++++859&submit=display+all+measurements#lab_meas
a = a.split("Name=")
url_start = a[0]+"Name="
url_end = "&submit=display+all+measurements#lab_meas"
b = a[1].split("&submit=submit")[0]
b = list(filter(None, b.split("%20")))
b = '+'.join(b)
#print(url_start)
#print(b)
sim_url = url_start+b+url_end
print(sim_url)

'''
import urllib.request
from bs4 import BeautifulSoup


with urllib.request.urlopen(sim_url) as url:
          html = url.read()
soup = BeautifulSoup(html,features="lxml")

# kill all script and style elements
for script in soup(["script", "style"]):
    script.extract()    # rip it out

# get text
text = soup.get_text()

# break into lines and remove leading and trailing space on each
lines = (line.strip() for line in text.splitlines())
# break multi-headlines into a line each
chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
# drop blank lines
text = '\n'.join(chunk for chunk in chunks if chunk)
for i in range(len(text)-4):
          if text[i:i+5] == "|Teff":
                    text = text[i:]
                    break
for i in range(len(text)-4):
          if text[i:i+3] == "plx":
                    text = text[:i]
                    #print("true")
                    break
#print(text)
text = text.split("\n")
print(text)
'''
