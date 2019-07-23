import pandas as pd
import numpy as np
import os,re,jieba,joblib
from cal_tfidf import *

class aso_tfidf():
    def __init__(self,df,dict_intro,dict_ver):
        '''

        :param df: dataframe  contains five columns [appleid，rdtimestamp，timestamp，filename, category]
        :param dict_intro:
        :param dict_ver:
        '''
        self.df = df
        self.intro = dict_intro
        self.ver = dict_ver

        self.gentime_dict()

    def gentime_dict(self):
        self.dict_time = {
            '20170131':1485878400,
            '20170228':1488297600,
            '20170331':1490976000,
            '20170430':1493568000,
            '20170531':1496246400,
            '20170630':1498838400,
            '20170731':1501516800,
            '20170831':1504195200,
            '20170930':1506787200,
            '20171031':1509465600,
            '20171130':1512057600,
            '20171231':1514736000}

    def genappleid_text(self,timestamp):
        '''
        这个函数是用于生成一个在本时间点下appleid和文本的对应关系
        :param timestamp:
        :return:dict _appleid_text
        '''
        dict_appleid_text = {}
        tempdf = self.df[self.df['rdtimestamp']<timestamp & self.df['timestamp'] < timestamp]
        l_appleid = list(set(tempdf['appleid'].values.tolist()))
        for appleid in l_appleid:
            tempdf2 = tempdf[tempdf['appleid'] == appleid]
            tempfilename_l = tempdf2['filename'].values.tolist()
            tempvertext = []
            for i in tempfilename_l:
                try:
                    tempvertext += self.ver[i]
                except:
                    continue
            try:
                temptext = self.intro[appleid] + tempvertext
                dict_appleid_text[appleid] = temptext
            except:
                continue
        return dict_appleid_text
