import hashlib
import os
import tarfile

from os.path import join
from pkg_resources import get_distribution,resource_exists
from tempfile import TemporaryDirectory
from timeit import default_timer
from urllib.request import urlretrieve

MODULE_PATH = join(get_distribution('campvideo').module_path,'campvideo')

class Model:
    def __init__(self,name,filename,url,archive=None,sha=None):
        # model name
        self.name = name
        # uncompressed model file
        self.filename = filename
        # download url
        self.url = url
        # archive name (optional)
        self.archive = archive
        # file SHA for verification
        self.sha = sha
        
    def get(self):
        # check if file already downloaded
        if resource_exists('campvideo',join('models',self.filename)):
            print('Model `%s` found -- skipping' % self.name)
            return
        
        # download file
        with TemporaryDirectory() as temp:
            # save download in temporary directory
            if self.archive is not None:
                fname = join(temp,self.archive) 
            else:
                fname = join(temp,self.filename)
            out,msg = urlretrieve(self.url,fname,self._reporthook)
            
            # extract if archive downloaded
            if self.archive is not None:
                with tarfile.open(out) as tf:
                    tf.extract(self.filename,join(MODULE_PATH,'models'))
            # otherwise copy file into models directory        
            else:
                os.rename(out,join(MODULE_PATH,'models',self.filename))
                
        # verify file was downloaded correctly by comparing SHA
        fname = join(MODULE_PATH,'models',self.filename)
        sha = hashlib.sha1()
        with open(fname,'rb') as fh:
            while True:
                # read in 10 MB chunks
                buff = fh.read(10 * 1024 ** 2)
                if not buff: break
                sha.update(buff)
                
        if self.sha != sha.hexdigest():
            raise Exception('Error downloading model %s' % self.name)
    
    # function for displaying download info for a urlretrieve request    
    def _reporthook(self,count,block_size,total_size):
        global start
        if count == 0:
            start = default_timer()
            return
        # time since first call
        dur = default_timer() - start
        # amount downloaded
        progress = int(count * block_size)
        # MB/s
        speed = progress / (1024 ** 2 * dur)
        percent = min(int(count * block_size * 100 / total_size), 100)
		# print out progress and download speed
        print('\rDownloading model `%s`... %d%%, %6.2f MB/s' % 
              (self.name,percent,speed),end='',flush=True)
        if percent == 100: print() # adds newline

# list of models to download
MODELS = [
    Model(
        name='EAST',
        filename='frozen_east_text_detection.pb',
        url='https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1',
        archive='frozen_east_text_detection.tar.gz',
        sha='fffabf5ac36f37bddf68e34e84b45f5c4247ed06'
        )
]

# script for downloading, extracting, and saving models into campvideo package
def main():
    # check if models folder exists
    if not resource_exists('campvideo','models'):
        os.mkdir(join(MODULE_PATH,'models'))
    
    for model in MODELS:
        model.get()
    print("All models downloaded!")