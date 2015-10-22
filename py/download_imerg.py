import ftplib
import os

# connect to FTP server:
ftp = ftplib.FTP('jsimpson.pps.eosdis.nasa.gov')
ftp.login('david.brochart@gmail.com', 'david.brochart@gmail.com')
ftp.cwd('NRTPUB/imerg')

for imerg_type in ['early', 'late']:
    ftp.cwd(imerg_type)
    dest = '../big_data/imerg/' + imerg_type
    
    
    # get directories names (one directory per month, e.g. '201503'):
    lines = []
    ftp.dir(lines.append)
    dirnames = [line.split()[-1] for line in lines]
    
    # copy files from every FTP directory to destination directory:
    for dirname in dirnames:
        ftp.cwd(dirname)
        lines = []
        ftp.dir(lines.append)
        filenames = [line.split()[-1] for line in lines]
        for filename in filenames:
            if not os.path.exists(dest + '/' + dirname):
                os.makedirs(dest + '/' + dirname)
            destname = dest + '/' + dirname + '/' + filename
            if not os.path.exists(destname):
                print('Downloading to ' + destname)
                destfile = open(destname, 'wb')
                ftp.retrbinary("RETR " + filename, destfile.write)
                destfile.close()
            else:
                print('File ' + destname + ' already downloaded')
        ftp.cwd('..')

    ftp.cwd('..')

ftp.quit()
