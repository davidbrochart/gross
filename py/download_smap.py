import ftplib
import os

# connect to FTP server:
ftp = ftplib.FTP('n5eil01u.ecs.nsidc.org')
ftp.login('anonymous', 'anonymous')
ftp.cwd('SAN/SMAP')

for smap_type in ['SPL3SMP.002']:
    ftp.cwd(smap_type)
    dest = '../big_data/smap/' + smap_type

    # get directories names (one directory per day, e.g. '2015.03.31'):
    lines = []
    ftp.dir(lines.append)
    dirnames = [line.split()[-1] for line in lines]
    dirnames = [dirname for dirname in dirnames if dirname.startswith('20') and dirname[4] == '.' and dirname[7] == '.']

    # copy files from every FTP directory to destination directory:
    for dirname in dirnames:
        ftp.cwd(dirname)
        lines = []
        ftp.dir(lines.append)
        filenames = [line.split()[-1] for line in lines]
        for filename in filenames:
            if filename.endswith('.h5'):
                destname = dest + '/' + filename
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
