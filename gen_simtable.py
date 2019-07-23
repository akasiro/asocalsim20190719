import pandas as pd
import joblib,sqlite3,os
from cal_tfidf import *

class aso_calsim():
    def __init__(self,df,df_type_id,dict_intro,dict_ver,exemplarlist = None):
        '''

        :param df: dataframe  contains five columns [appleid，rdtimestamp，timestamp，filename]
        :param df_type_id: contains two columns [appleid, type_code]
        :param dict_intro:
        :param dict_ver:
        :param exemplarlist Default None, if wants to cal with exemplar model , input a list of exemplar appleid
        '''
        self.df = df
        self.df_type_id = df_type_id
        self.intro = dict_intro
        self.ver = dict_ver
        self.exemplarlist = exemplarlist

        self.dfpreclean(df,df_type_id)
        self.gentime_dict()
        self.dict_type_code_id = self.gendict_type_code_id(self.df_type_id)
    def dfpreclean(self,df1,df2):
        temp = pd.merge(df1,df2, on = 'appleid')
        temp = temp[['appleid','rdtimestamp', 'timestamp', 'filename']].drop_duplicates()
        self.df = temp
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
    def gendict_type_code_id(self,df_type_id):
        '''
        用于将df_type_id转化成字典
        :param df_type_id:
        :return:
        '''
        type_code = list(set(df_type_id['type_code'].values.tolist()))
        dict_type_code_id = {}
        for i in type_code:
            temp = df_type_id[df_type_id['type_code'].isin([i])]
            tempappleidlist = list(set(temp['appleid'].values.tolist()))
            dict_type_code_id[i] = tempappleidlist
        return dict_type_code_id

    def genappleid_text(self,timestamp):
        '''
        这个函数是用于生成一个在本时间点下appleid和文本的对应关系
        :param timestamp:
        :return:dict _appleid_text
        '''
        dict_appleid_text = {}
        tempdf = self.df[(self.df['rdtimestamp']<timestamp) & (self.df['timestamp'] < timestamp)]
        l_appleid = list(set(tempdf['appleid'].values.tolist()))
        for appleid in l_appleid:
            try:
                temptext = self.intro[str(appleid)]
            except:
                continue
            tempvertext = []
            try:
                tempdf2 = tempdf[tempdf['appleid'] == appleid]
                tempfilename_l = tempdf2['filename'].values.tolist()
                for i in tempfilename_l:
                    try:
                        tempvertext += self.ver[i]
                    except:
                        continue
            except:
                pass
            temptext += tempvertext
            dict_appleid_text[appleid] = temptext
        return dict_appleid_text

    def cal_tfidf1(self,exemplar = False):
        '''
        将所有category的所有的文本视为corpus，比较该游戏与对应的catogory的全体文本的相似度
        :return:
        '''
        df = pd.DataFrame()
        for date in self.dict_time:
            # 构造在这个时间点上appleid与text对应的字典
            dict_appleid_text = self.genappleid_text(self.dict_time[date])
            #  构造baselistoftextlist
            baselistoftextlist = []
            dict_idinsim_typecode = {}
            i = 0
            for type_code  in self.dict_type_code_id:
                temptextlist = []
                templist = self.dict_type_code_id[type_code]
                for appleid in templist:
                    if exemplar:
                        if appleid not in self.exemplarlist:
                            continue
                    try:
                        temptextlist += dict_appleid_text[appleid]
                    except:
                        continue
                baselistoftextlist.append(temptextlist)
                dict_idinsim_typecode[type_code] = i
                i+=1
            # 使用构造出的baselistofte xtlist训练模型

            tfidf_model = tfidfmodel_cal(baselistoftextlist)


            # 计算每个id的战略差异
            dict_id_diff = {}
            for type_code in self.dict_type_code_id:
                for appleid in self.dict_type_code_id[type_code]:
                    if exemplar:
                        if appleid in self.exemplarlist:
                            continue
                    try:
                        textlist = dict_appleid_text[appleid]
                        tempsim = tfidf_model.cal_tfidf(textlist,dict_idinsim_typecode[type_code])
                        if tempsim == 0:
                            continue
                        dict_id_diff[appleid] = 1 - tempsim
                    except:
                        continue
            # 将结果转化为dataframe
            if exemplar:
                column_name = 'diff_e'
            else:
                column_name = 'diff_p_f'
            tempdf = pd.DataFrame({'appleid':[k for k in dict_id_diff], column_name:[dict_id_diff[k] for k in dict_id_diff]})
            tempdf['date'] = date
            tempdf['timestamp'] = self.dict_time[date]
            df = df.append(tempdf,ignore_index=True)
        return df





    def cal_tfidf2(self):
        '''
        将一个category的所有的文本视为corpus，将该游戏与category中的每个游戏的文本分别比较，将最大相似度视为相似度
        :return:
        '''
        df = pd.DataFrame()
        for date in self.dict_time:
            # 构造在这个时间点上appleid与text对应的字典
            dict_appleid_text = self.genappleid_text(self.dict_time[date])
            # 计算sim
            dict_id_diff = {}
            for type_code in self.dict_type_code_id:
                templist = self.dict_type_code_id[type_code]
                # 构造baselistoftextlist
                baselistoftextlist = []
                for appleid in templist:
                    try:
                        baselistoftextlist.append(dict_appleid_text[appleid])
                    except:
                        continue
                #训练模型
                try:
                    tfidf_model = tfidfmodel_cal(baselistoftextlist)
                except:
                    continue
                # 计算每个id的战略差异
                for appleid in templist:
                    try:
                        textlist = dict_appleid_text[appleid]
                        tempsim = tfidf_model.cal_tfidf(textlist)
                        if tempsim == 0:
                            continue
                        dict_id_diff[appleid] = 1 - tempsim
                    except:
                        continue
            # 将结果转化为dataframe
            tempdf = pd.DataFrame({'appleid': [k for k in dict_id_diff], 'diff_p_max': [dict_id_diff[k] for k in dict_id_diff]})
            tempdf['date'] = date
            tempdf['timestamp'] = self.dict_time[date]
            df = df.append(tempdf, ignore_index=True)
        return df

    def gentfidftable(self):
        df1 = self.cal_tfidf1()
        df2 = self.cal_tfidf2()
        tempdf = df1.merge(df2,on = ['appleid','date','timestamp'], how = 'outer')
        if self.exemplarlist != None:
            df3 = self.cal_tfidf1(exemplar=True)
            tempdf = tempdf.merge(df3, on = ['appleid','date','timestamp'], how = 'outer')
        return tempdf

if __name__ == '__main__':
    def load_joblib(name):
        with open(name + '.pkl', 'rb') as f:
            return joblib.load(f)
    #  读取数据
    dict_intro = load_joblib(os.path.join('..','..','..','..','data','aso.niaoge','20190723','dict_intro'))
    dict_ver = load_joblib(os.path.join('..','..','..','..','data','aso.niaoge','20190723','dict_ver'))
    conn = sqlite3.connect(os.path.join('..','..','..','..','data','aso.niaoge','20190723','aso20190723.db'))
    data1 = pd.read_sql('select appleid, rdtimestamp from baseinfo',conn)
    data2 = pd.read_sql('select appleid, timestamp,filename from version',conn)
    data = data1.merge(data2, on = 'appleid')
    df_type_id = pd.read_sql('select type_code,appleid from category_appleid_fromTDorigin',conn)
    df_type_id = df_type_id[~df_type_id['type_code'].isin(['T200700', 'T201000', 'T201100', 'T201500'])]
    with open(os.path.join('..','..','..','..','data','aso.niaoge','20190723','exemplar.txt'), 'r') as f:
        exemplarlist = f.read().replace('\n','').split(',')
    exemplarlist = [int(i) for i in exemplarlist]
    # 计算
    a = aso_calsim(data,df_type_id,dict_intro,dict_ver,exemplarlist)
    result = a.gentfidftable()
    # 保存结果
    result.to_csv(os.path.join('..','..','data','aso_diff20190724.csv'))
    print('Done')
