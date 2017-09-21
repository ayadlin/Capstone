# use this code on desktop capstone or on directory where all pdf and quosa
# search results are

import os
from glob import glob
import os, errno
from os import path
import requests
from bs4 import BeautifulSoup as bs
#import regex as re
import copy
import pandas as pd
import data_frame_creator

files = []
start_dir = os.getcwd()
pattern   = "*.pdf"

for dir,_,_ in os.walk(start_dir):
    files.extend(glob(os.path.join(dir,pattern)))


def find_source(doc):
    doc = doc+'.html'
    #print(doc)
    file = bs(open(doc),'lxml')
    div =  file.find_all("div", {"style": "padding: 2px; font-size: 12px; padding-left: 5px; font-style: italic;"})
    text = str(div[1])
    start = text.find('Source:') + 15
    end = text.find('</div>')-1
    source = text[start:end]
    return source




source_dict = {}
for file in files:
    key = file[-48:]
    if os.path.isfile(file+'.html'):
        #print(key)
        source_dict[key]=find_source(file);

data_frame_creator.write_pickle('source_dict.pickle', source_dict)
