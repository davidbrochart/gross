import sys
import os
from zipfile import ZipFile
import shutil
from numba import jit
import requests
import numpy as np
import rasterio
from tqdm import tqdm

def delineate(lat, lon, sub_latlon=[], accDelta=10000):
    getSubBass = True
    sample_i = 0
    samples = np.empty((1024, 2), dtype=np.float32)
    labels= np.empty((1024, 3), dtype=np.int32)
    sub_latlon = np.empty((1, 2), dtype=np.float32)
    dirNeighbors = np.empty(1024, dtype=np.uint8)
    accNeighbors = np.empty(1024, dtype=np.uint32)
    ws_latlon = np.empty(2, dtype=np.float32)
    # output mask ->
    mxw = 3000 # bytes
    myw = mxw * 8 # bits
    mm = np.empty((myw, mxw), dtype = np.uint8)
    mm_back = np.empty((myw, mxw), dtype = np.uint8)
    mx0_deg = 0
    my0_deg = 0
    # <- output mask

    if len(sub_latlon) == 0:
        sub_latlon[0] = [lat, lon]
    else:
        sub_latlon = np.empty((len(sub_latlon), 2), dtype=np.float32)
        sub_latlon[:] = sub_latlon
    dir_tile, acc_tile = getTile(lat, lon)
    _, _, _, _, lat0, lon0, pix_deg = getTileInfo(lat, lon)
    print('Getting bassin partition...')
    samples, labels, sample_size, mx0_deg, my0_deg, ws_mask, ws_latlon, dirNeighbors, accNeighbors = do_delineate(lat, lon, lat0, lon0, dir_tile, acc_tile, getSubBass, sample_i, samples, labels, pix_deg, accDelta, sub_latlon, mm, mm_back, mx0_deg, my0_deg, dirNeighbors, accNeighbors)
    print('Delineating sub-bassins...')
    mask, latlon = [], []
    getSubBass = False
    for sample_i in tqdm(range(sample_size)):
        _, _, _, mx0_deg, my0_deg, ws_mask, ws_latlon, dirNeighbors, accNeighbors = do_delineate(lat, lon, lat0, lon0, dir_tile, acc_tile, getSubBass, sample_i, samples, labels, pix_deg, accDelta, sub_latlon, mm, mm_back, mx0_deg, my0_deg, dirNeighbors, accNeighbors)
        mask.append(ws_mask)
        latlon.append(ws_latlon)
    ws = {}
    ws['outlet'] = samples[sample_size - 1::-1]
    ws['mask'] = mask[::-1]
    ws['latlon'] = np.empty((sample_size, 2), dtype=np.float32)
    ws['latlon'][:, :] = latlon[::-1]
    # label reconstruction:
    ws['label'] = []
    for sample_i in range(sample_size - 1, -1, -1):
        if sample_i == sample_size - 1: # outlet subbassin
            ws['label'].append('0')
        else:
            i = labels[sample_i][0]
            ws['label'].append(ws['label'][i] + ',' + str(labels[sample_i][2]))
    return ws

#def do_stream(self, lat, lon, lat0, lon0, olat, olon, dir_tile, pix_deg):
#    x, y, x_deg, y_deg = getXY(lat, lon, lat0, lon0, pix_deg)
#    if olon - pix_deg / 2 <= lon < olon + pix_deg / 2 and olat - pix_deg <= lat < olat + pix_deg / 2:
#        return 0
#    stream = [[x_deg + pix_deg / 2, y_deg - pix_deg / 2]]
#    done = False
#    while not done:
#        _, x, y, _, _, x_deg, y_deg = go_get_dir(dir_tile[y, x], dir_tile, x, y, 0, 0, x_deg, y_deg, pix_deg)
#        stream.append([x_deg + pix_deg / 2, y_deg - pix_deg / 2])
#        if olon - pix_deg / 2 <= x_deg < olon + pix_deg / 2 and olat - pix_deg <= y_deg < olat + pix_deg / 2:
#            done = True
#    return LineString(stream).length

@jit(nopython=True)
def do_delineate(lat, lon, lat0, lon0, dir_tile, acc_tile, getSubBass, sample_i, samples, labels, pix_deg, accDelta, sub_latlon, mm, mm_back, mx0_deg, my0_deg, dirNeighbors, accNeighbors):
    if getSubBass:
        x, y, x_deg, y_deg = getXY(lat, lon, lat0, lon0, pix_deg)
        acc = int(acc_tile[y,  x])
        samples[0, :] = [y_deg - pix_deg / 2, x_deg + pix_deg / 2]
        rm_latlon(samples[0], sub_latlon, pix_deg)
        sample_i = 0
        labels[0, :] = [-1, 1, 0] # iprev, size, new_label
        label_i = 0
        new_label = 0
    else:
        lat, lon = samples[sample_i]
        x, y, x_deg, y_deg = getXY(lat, lon, lat0, lon0, pix_deg)
        if sample_i == 0:
            mm_back[:] = 0
            mx = int(mm.shape[0] / 2 - 1)
            my = int(mm.shape[0] / 2 - 1)
            mx0_deg = x_deg - pix_deg * mx
            my0_deg = y_deg + pix_deg * my
            mm[:] = 0
        else:
            mm_back[:] |= mm[:]
            mx = int(round((x_deg - mx0_deg) / pix_deg))
            my = int(round((my0_deg - y_deg) / pix_deg))
    neighbors_i = 0
    dirNeighbors[0] = 255 # 255 is for uninitialized
    accNeighbors[0] = 0
    done = False
    skip = False
    while not done:
        reached_upper_ws = False
        if not skip:
            if getSubBass:
                this_acc = int(acc_tile[y, x])
                this_accDelta = acc - this_acc
                append_sample = False
                if this_accDelta >= accDelta and this_acc >= accDelta:
                    append_sample = True
                if in_latlon([y_deg - pix_deg / 2, x_deg + pix_deg / 2], sub_latlon, pix_deg):
                    append_sample = True
                if append_sample:
                    acc = this_acc
                    sample_i += 1
                    if sample_i == samples.shape[0]:
                        samples_new = np.empty((samples.shape[0] * 2, 2), dtype=np.float32)
                        samples_new[:samples.shape[0], :] = samples
                        samples = samples_new
                        labels_new = np.empty((labels.shape[0] * 2, 3), dtype=np.int32)
                        labels_new[:labels.shape[0], :] = labels
                        labels = labels_new
                    samples[sample_i, :] = [y_deg - pix_deg / 2, x_deg + pix_deg / 2]
                    rm_latlon(samples[sample_i], sub_latlon, pix_deg)
                    labels[sample_i, :] = [label_i, labels[label_i, 1] + 1, new_label]
                    new_label = 0
                    label_i = sample_i
            else:
                if (mm_back[my, int(np.floor(mx / 8))] >> (mx % 8)) & 1 == 1: # we reached the upper sub-watershed
                    reached_upper_ws = True
                else:
                    mm[my, int(np.floor(mx / 8))] |= 1 << (mx % 8)
        nb = dirNeighbors[neighbors_i]
        if not reached_upper_ws and nb == 255:
            # find which pixels flow into this pixel
            nb = 0
            for i in range(8):
                if i < 4:
                    dir_back = 1 << (i + 4)
                else:
                    dir_back = 1 << (i - 4)
                dir_next, _, _, _, _, _, _ = go_get_dir(1 << i, dir_tile, x, y, mx, my, x_deg, y_deg, pix_deg)
                if dir_next == dir_back:
                    nb = nb | (1 << i)
            dirNeighbors[neighbors_i] = nb
            if getSubBass:
                accNeighbors[neighbors_i] = acc
        if reached_upper_ws or nb == 0: # no pixel flows into this pixel (this is a source), so we cannot go upper
            if neighbors_i == 0: # we are at the outlet and we processed every neighbor pixels, so we are done
                done = True
            else:
                passed_ws = False
                go_down = True
                while go_down:
                    _, x, y, mx, my, x_deg, y_deg = go_get_dir(dir_tile[y, x], dir_tile, x, y, mx, my, x_deg, y_deg, pix_deg)
                    if getSubBass:
                        if passed_ws: # we just passed a sub-basin
                            this_label = labels[label_i]
                            new_label = this_label[2] + 1
                            this_length = this_label[1]
                            while labels[label_i, 1] >= this_length:
                                label_i -= 1
                            passed_ws = False
                        # check if we are at a sub-basin outlet that we already passed
                        y_down, x_down = samples[label_i]
                        if (y_down - pix_deg / 4 < y_deg - pix_deg / 2 < y_down + pix_deg / 4) and (x_down - pix_deg / 4 < x_deg + pix_deg / 2 < x_down + pix_deg / 4):
                            passed_ws = True
                    neighbors_i -= 1
                    nb = dirNeighbors[neighbors_i]
                    i = find_first1(nb)
                    nb = nb & (255 - (1 << i))
                    if nb == 0:
                        if neighbors_i == 0:
                            go_down = False
                            done = True
                    else:
                        go_down = False
                        skip = True
                    dirNeighbors[neighbors_i] = nb
                acc = accNeighbors[neighbors_i]
        else: # go up
            skip = False
            neighbors_i += 1
            if neighbors_i == dirNeighbors.shape[0]:
                dirNeighbors_new = np.empty(dirNeighbors.shape[0] * 2, dtype = np.uint8)
                dirNeighbors_new[:dirNeighbors.shape[0]] = dirNeighbors
                dirNeighbors = dirNeighbors_new
                accNeighbors_new = np.empty(accNeighbors.shape[0] * 2, dtype = np.uint32)
                accNeighbors_new[:accNeighbors.shape[0]] = accNeighbors
                accNeighbors = accNeighbors_new
            dirNeighbors[neighbors_i] = 255
            accNeighbors[neighbors_i] = 0
            i = find_first1(nb)
            _, x, y, mx, my, x_deg, y_deg = go_get_dir(1 << i, dir_tile, x, y, mx, my, x_deg, y_deg, pix_deg)
        if done:
            ws_latlon = np.empty(2, dtype=np.float32)
            if getSubBass:
                sample_size = sample_i + 1
                # we need to reverse the samples (incremental delineation must go downstream)
                samples[:sample_size, :] = samples[sample_size-1::-1, :].copy()
                labels[:sample_size, :] = labels[sample_size-1::-1, :].copy()
                sample_i = 0
                ws_mask = np.empty((1, 1), dtype=np.uint8)
            else:
                mm[:] &= ~mm_back[:]
                ws_mask, ws_latlon[0], ws_latlon[1] = get_bbox(mm, pix_deg, mx0_deg, my0_deg)
    return samples, labels, sample_size, mx0_deg, my0_deg, ws_mask, ws_latlon, dirNeighbors, accNeighbors

@jit(nopython=True)
def getXY(lat, lon, lat0, lon0, pix_deg):
    lat = round(lat, 5)
    lon = round(lon, 5)
    x = int(np.floor((lon - lon0) / pix_deg))
    y = int(np.floor((lat0 - lat) / pix_deg))
    x_deg = lon0 + x * pix_deg
    y_deg = lat0 - y * pix_deg
    return x, y, x_deg, y_deg

def getTileInfo(lat, lon):
    if (-56 <= lat <= 15) and (-93 <= lon <= -32): # South America
        lat0, lon0 = 15, -93
        pix_deg = 0.004166666666667
        dir_url = 'http://earlywarning.usgs.gov/hydrodata/sa_15s_zip_grid/sa_dir_15s_grid.zip'
        acc_url = 'http://earlywarning.usgs.gov/hydrodata/sa_15s_zip_grid/sa_acc_15s_grid.zip'
        tile_width = 14640
        tile_height = 17040
    else:
        print('Position not covered for now: lat, lon = ' + str(lat) + ', ' + str(lon))
        sys.exit()
    return tile_width, tile_height, dir_url, acc_url, lat0, lon0, pix_deg

def getTile(lat, lon):
    _, _, dir_url, acc_url, lat0, lon0, pix_deg = getTileInfo(lat, lon)
    x, y, x_deg, y_deg = getXY(lat, lon, lat0, lon0, pix_deg)

    url = {'dir': dir_url, 'acc': acc_url}
    tiles = []
    for typ in ['dir', 'acc']:
        tmpDir = 'tmp/'
        this_url = url[typ]
        this_dir = this_url[this_url.rfind('/') + 1:this_url.rfind('_grid')]
        adf_file = tmpDir + this_dir + '/' + this_dir + '/w001001.adf'
        if not os.path.exists(adf_file):
            if not os.path.exists(tmpDir):
                os.mkdir(tmpDir)
            fpath = tmpDir + os.path.basename(this_url)
            if os.path.exists(tmpDir + os.path.basename(this_url)):
                print('Already downloaded ' + this_url)
            else:
                print('Downlading ' + this_url)
                r = requests.get(this_url, stream=True)
                with open(fpath, 'wb') as f:
                    for data in r.iter_content():
                        f.write(data)
            if os.path.exists(tmpDir + this_dir):
                shutil.rmtree(tmpDir + this_dir)
            print('Unzipping ' + fpath)
            with ZipFile(fpath, 'r') as z:
                z.extractall(tmpDir)
        print('Opening ' + adf_file)
        with rasterio.open(adf_file) as src:
            data = src.read()
        tiles.append(data[0].astype({'dir': np.uint8, 'acc': np.uint32}[typ]))
    return tiles[0], tiles[1]

@jit(nopython=True)
def in_latlon(ll, ll_list, prec):
    for i in range(ll_list.shape[0]):
        if ll_list[i, 0] > -900:
            if (ll[0] - prec / 2 <= ll_list[i, 0] < ll[0] + prec / 2) and (ll[1] - prec / 2 <= ll_list[i, 1] < ll[1] + prec / 2):
                return True
    return False

@jit(nopython=True)
def rm_latlon(ll, ll_list, prec):
    for i in range(ll_list.shape[0]):
        if ll_list[i, 0] > -900:
            if (ll[0] - prec / 2 <= ll_list[i, 0] < ll[0] + prec / 2) and (ll[1] - prec / 2 <= ll_list[i, 1] < ll[1] + prec / 2):
                ll_list[i] = [-999, -999]

@jit(nopython=True)
def go_get_dir(dire, dir_tile, x, y, mx, my, x_deg, y_deg, pix_deg):
    for i in range(8):
        if (dire >> i) & 1 == 1:
            break
    dx = np.array([1, 1, 0, -1, -1, -1, 0, 1])[i]
    dy = np.array([0, 1, 1, 1, 0, -1, -1, -1])[i]
    return dir_tile[y + dy, x + dx], x + dx, y + dy, mx + dx, my + dy, x_deg + dx * pix_deg, y_deg - dy * pix_deg

@jit(nopython=True)
def find_first1(x):
    i = 0
    while (x & 1) == 0:
        x = x >> 1
        i += 1
    return i

@jit(nopython=True)
def get_bbox(mm, pix_deg, mx0_deg, my0_deg):
    going_down = True
    i = mm.shape[0] >> 1
    i0 = i
    i1 = i - 1
    done = False
    while not done:
        for j in range(mm.shape[1]):
            if mm[i, j] != 0:
                done = True
        if not done:
            if going_down:
                i0 += 1
                i = i1
            else:
                i1 -= 1
                i = i0
            going_down = not going_down
    if i > 0:
        i -= 1
    done = False
    while not done:
        done = True
        for j in range(mm.shape[1]):
            if mm[i, j] != 0:
                done = False
        if not done:
            i -= 1
            if i < 0:
                done = True
    i += 1

    x0 = mm.shape[1] * 8
    x1 = -1
    y0 = -1
    y1 = -1
    found_y = False
    done = False
    while not done:
        found_x = False
        for j in range(mm.shape[1]):
            if mm[i, j] != 0:
                found_x = True
                for k in range(8):
                    if (mm[i, j] >> k) & 1 == 1:
                        l = j * 8 + k
                        if x0 > l:
                            x0 = l
                        if x1 < l:
                            x1 = l
        if found_x:
            found_y = True
            y0 = i
            if y1 < 0:
                y1 = i
        if not found_x and found_y:
            done = True
        else:
            i += 1
            if i == mm.shape[0]:
                done = True
    y0 += 1
    x1 += 1
    mask = np.empty((y0 - y1, x1 - x0), dtype=np.uint8)
    for i in range(y1, y0):
        for j in range(x0, x1):
            mask[(i - y1), j - x0] = (mm[i, int(np.floor(j / 8))] >> (j % 8)) & 1
    return mask, my0_deg - pix_deg * y1, mx0_deg + pix_deg * x0
