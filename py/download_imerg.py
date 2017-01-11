import ftplib
import os

# connect to FTP server:
ftp = ftplib.FTP('jsimpson.pps.eosdis.nasa.gov')
ftp.login('david.brochart@gmail.com', 'david.brochart@gmail.com')
ftp.cwd('NRTPUB/imerg/gis')

dest = '../big_data/imerg/gis/'
os.makedirs(dest, exist_ok=True)

months_or_years = ['2015', '2016', '2017', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

# copy files from every FTP directory to destination directory:
def get_data():
    lines = []
    ftp.dir(lines.append)
    names = [line.split()[-1] for line in lines]
    for name in names:
        if name in months_or_years:
            ftp.cwd(name)
            get_data()
            ftp.cwd('..')
        else:
            if name.endswith('.V03E.30min.tif.gz'):
                destname = dest + name
                if not os.path.exists(destname):
                    print('Downloading to ' + destname)
                    destfile = open(destname, 'wb')
                    ftp.retrbinary("RETR " + name, destfile.write)
                    destfile.close()
                else:
                    print('File ' + destname + ' already downloaded')

get_data()
ftp.quit()
