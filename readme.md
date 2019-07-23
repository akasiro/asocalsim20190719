数据处理说明
# 数据预处理

## 1.删除掉文本数据中的非中文部分，只保留中文
```
re.sub(r'[^\u4e00-\u9fa5]','',text)
```
## 2. jieba分词
```
pip install jieba
```
```
temp = jieba.lcut(temp)
```

## 3. 去掉stop word
```
temp = [i for i in temp if i not in stopwords]
```
## 4. 处理dataframe中的文本数据，处理为dict，并保存到pickle保证重复使用

## 5. 写一个函数用于根据时间戳提取对应的文本的key
### 关于时间戳与2017年每一年月末时间戳的对应关系存储在dict_time中
```
time_dict = {
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
    '20171231':1514736000
}


```
 # 计算战略差异：tfidf方法
 使用到的数据源：
 * dict_intro:键为appleid，值为introduction的文本分词后的列表的字典
 * dict_ver:键为filename（由appleid和版本号组成的字符串），值为本次更新内容的文本分词后的列表的字典
 * appleid和category的对应关系
 * 提取自数据库的dataframe：包含appleid，rdtimestamp，timestamp，filename四列，含义分别为
 appleid：游戏id，来自database中的baseinfo_td表和version表
 rdtimestamp：游戏发布时间戳，来自baseinfo_td表
 timestamp：游戏本次版本更新的时间点，来自version表
 filename：保存本次游戏更新内容的文本文件名，由游戏id和版本号组成，来自version表
 ## 1 明确在该时间点上有哪些游戏，分别属于哪些类别
 通过游戏的发布时间来确定在该时间点上游戏是否存在
 代码实现：
 ```
 # 判断方式
 if rdtimestamp < the timestamp of this month
 ```
 ## 2 明确在该时间点上这些游戏已经有哪些更新内容
 ```
 if timestamp < the timestamp of this month
 ```
 ## 3 将游戏的intro和已有的更新的文本视为在这个时间点上每个游戏的文本
 ## 4 计算文本相似度
 计算文本相似度的函数写在cal_tfidf.py
 计算相似度需要三个输入：
 * listoftextlist-作为训练模型的corpus文本集：形式为list of word list
 * textlist-需要计算相似的文本列表：形式为word list
 * typecode-相似度的index
 基于不同的理论模型和比较相似度的方式这三者会有所不同
 ### 4.1 基于prototype model的计算
 #### 4.1.1 将所有category的所有的文本视为corpus，比较该游戏与对应的catogory的全体文本的相似度
 listoftextlist 形式为
 ```
 [word list of category1, word list of category2, ...]
 word list of category = word list of game1 + word list of game2 + ... 
 ```
 textlist 形式为
 ```
 word list of game = word list of game intro + word list of game version
 ```
 typecode 为game所对应的categoy在模型中的index
 #### 4.1.2 将一个category的所有的文本视为corpus，将该游戏与category中的每个游戏的文本分别比较，将最大相似度视为相似度
 listoftextlist 形式为
 ```
 [word list of game1, wordlist of game2,...]
 ```
 textlist与上面相同
 typecode 为False，或者直接选择不输入这个参数
 ### 4.2.基于exemplar model的计算
  listoftextlist 形式为
 ```
 [word list of category1, word list of category2, ...]
 word list of category = word list of exemplar1 + word list of exemplar2 + ... 
 ```
 textlist 同上
 typecode 为game所对应的categoy在模型中的index

 