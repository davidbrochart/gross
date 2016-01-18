import ftplib
import os

# connect to FTP server:
ftp = ftplib.FTP('ftp.nodc.noaa.gov')
ftp.login('anonymous', 'david.brochart@gmail.com')
ftp.cwd('pub/data.nodc/jason2/gdr/gdr')

dest = '../big_data/jason2/gdr'

# get directories names (e.g. cycle000):
lines = []
ftp.dir(lines.append)
dirnames = [line.split()[-1] for line in lines]
dirnames = [dirname for dirname in dirnames if dirname.startswith('cycle') and int(dirname[5:]) >= 246]

done = False
while not done:
    try:
        # copy files from every FTP directory to destination directory:
        for dirname in dirnames:
            ftp.cwd(dirname)
            lines = []
            ftp.dir(lines.append)
            filenames = [line.split()[-1] for line in lines]
            for filename in filenames:
                if filename.endswith('.nc'):
                    destname = dest + '/' + filename
                    if not os.path.exists(destname):
                        print('Downloading to ' + destname)
                        destfile = open(destname, 'wb')
                        ftp.retrbinary("RETR " + filename, destfile.write)
                        destfile.close()
                    else:
                        print('File ' + destname + ' already downloaded')
            ftp.cwd('..')
        done = True
    except:
        os.system('rm -f ' + destname)

ftp.quit()
