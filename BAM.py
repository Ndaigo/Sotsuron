#argumentClass、relatedID判定
#確定版#

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Optional
import os
import re
from sudachipy import tokenizer
from sudachipy import dictionary
import gensim
import numpy as np
from scipy import spatial
from gensim import corpora
from gensim import models
from tqdm import tqdm
import math

import glob
#from sklearn.manifold import TSNE
#from sklearn.decomposition import PCA

import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer, BertModel, BertForSequenceClassification
#from actrainmodel2 import BertForSequenceClassificationMultiLabel

tokenizer_obj = dictionary.Dictionary().create()
mode = tokenizer.Tokenizer.SplitMode.C

MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'
# トークナイザとモデルのロード
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)
model = model.cuda()

ac_bert_sc = BertForSequenceClassification.from_pretrained('BAM/ac_model_transformers')
#ac_bert_sc = BertForSequenceClassificationMultiLabel.from_pretrained('BAM/ac_model_transformers2')
ac_bert_sc = ac_bert_sc.cuda()

#ri_bert_sc = BertForSequenceClassification.from_pretrained('BAM/ri_model_transformers')
#ri_bert_sc = ri_bert_sc.cuda()

#chive_vectors = models.KeyedVectors.load("/home/daigo/Sotsuron/chive-1.2-mc5_gensim/chive-1.2-mc5.kv")

# --------------- データ定義ここから ---------------
ARGUMENT_CLASSES = [
    "Premise : 過去・決定事項",
    "Premise : 未来（現在以降）・見積",
    "Premise : その他（例示・訂正事項など）",
    "Claim : 意見・提案・質問",
    "Claim : その他",
    "金額表現ではない",
    "その他",
]


@dataclasses.dataclass
class MoneyExpression:
    """
    金額表現，関連する予算のID，議論ラベルを保持する．
    """

    moneyExpression: str
    relatedID: Optional[list[str]]
    argumentClass: Optional[str]


@dataclasses.dataclass(frozen=True)
class LProceedingItem:
    """
    地方議会会議録の一つ分の発言を保持する．
    発言には複数の金額表現が含まれる場合がある．
    """

    speakerPosition: str
    speaker: str
    utterance: str
    moneyExpressions: list[MoneyExpression]

    @staticmethod
    def from_dict(d: dict[str, any]):
        return LProceedingItem(
            speakerPosition=d["speakerPosition"],
            speaker=d["speaker"],
            utterance=d["utterance"],
            moneyExpressions=[MoneyExpression(**m) for m in d["moneyExpressions"]],
        )


@dataclasses.dataclass(frozen=True)
class LProceedingObject:
    """
    地方議会会議録の一つ分の会議を保持する．
    一つ分の会議は発言オブジェクトのリストを持つ．
    """

    date: str
    localGovernmentCode: str
    localGovernmentName: str
    proceedingTitle: str
    url: str
    proceeding: list[LProceedingItem]

    @staticmethod
    def from_dict(d: dict[str, any]):
        return LProceedingObject(
            date=d["date"],
            localGovernmentCode=d["localGovernmentCode"],
            localGovernmentName=d["localGovernmentName"],
            proceedingTitle=d["proceedingTitle"],
            url=d["url"],
            proceeding=[LProceedingItem.from_dict(x) for x in d["proceeding"]],
        )


@dataclasses.dataclass(frozen=True)
class DSpeechRecord:
    """
    国会会議録の一つ分の発言を保持する．
    発言には複数の金額表現が含まれる場合がある．
    """

    speechID: str
    speechOrder: int
    speaker: str
    speakerYomi: Optional[str]
    speakerGroup: Optional[str]
    speakerPosition: Optional[str]
    speakerRole: Optional[str]
    speech: str
    startPage: int
    createTime: str
    updateTime: str
    speechURL: str
    moneyExpressions: list[MoneyExpression]

    @staticmethod
    def from_dict(d: dict[str, any]):
        return DSpeechRecord(
            speechID=d["speechID"],
            speechOrder=d["speechOrder"],
            speaker=d["speaker"],
            speakerYomi=d["speakerYomi"],
            speakerGroup=d["speakerGroup"],
            speakerPosition=d["speakerPosition"],
            speakerRole=d["speakerRole"],
            speech=d["speech"],
            startPage=d["startPage"],
            createTime=d["createTime"],
            updateTime=d["updateTime"],
            speechURL=d["speechURL"],
            moneyExpressions=[MoneyExpression(**m) for m in d["moneyExpressions"]],
        )


@dataclasses.dataclass(frozen=True)
class DProceedingObject:
    """
    国会会議録の一つ分の会議を保持する．
    一つ分の会議は発言オブジェクトのリストを持つ．
    """

    issueID: str
    imageKind: str
    searchObject: int
    session: int
    nameOfHouse: str
    nameOfMeeting: str
    issue: str
    date: str
    closing: Optional[str]
    speechRecord: list[DSpeechRecord]
    meetingURL: str
    pdfURL: str

    @staticmethod
    def from_dict(d: dict[str, any]):
        return DProceedingObject(
            issueID=d["issueID"],
            imageKind=d["imageKind"],
            searchObject=d["searchObject"],
            session=d["session"],
            nameOfHouse=d["nameOfHouse"],
            nameOfMeeting=d["nameOfMeeting"],
            issue=d["issue"],
            date=d["date"],
            closing=d["closing"],
            speechRecord=[DSpeechRecord.from_dict(x) for x in d["speechRecord"]],
            meetingURL=d["meetingURL"],
            pdfURL=d["pdfURL"],
        )


@dataclasses.dataclass(frozen=True)
class MinutesObject:
    """
    BAMタスクにおける会議録データのフォーマット．
    地方議会と国会の両方を持つ．
    """

    local: list[LProceedingObject]
    diet: list[DProceedingObject]

    @staticmethod
    def from_dict(d: dict[str, any]):
        return MinutesObject(
            local=[LProceedingObject.from_dict(x) for x in d["local"]],
            diet=[DProceedingObject.from_dict(x) for x in d["diet"]],
        )


@dataclasses.dataclass(frozen=True)
class BudgetItem:
    """
    予算項目一つ分を保持する．

    budgetIdの命名規則は以下の通り．
    ID-[year]-[localGovernmentCode]-00-[index]
    例：ID-2020-401307-00-000001
    """

    budgetId: str
    budgetTitle: str
    url: Optional[str]
    budgetItem: str
    budget: str
    categories: list[str]
    typesOfAccount: Optional[str]
    department: str
    budgetLastYear: Optional[str]
    description: str
    budgetDifference: Optional[str]


@dataclasses.dataclass(frozen=True)
class BudgetObject:
    """
    BAMタスクにおける予算リストの配布用フォーマット．
    地方議会と国会の両方を持つ．

    地方議会（local）は，キーが自治体コード（localGovernmentCode），値がその自治体の予算項目リストとなる辞書型である．
    国会（diet）は，予算項目リストである．
    """

    local: dict[str, list[BudgetItem]]
    diet: list[BudgetItem]

    @staticmethod
    def from_dict(d: dict[str, any]):
        return BudgetObject(
            local={k: [BudgetItem(**x) for x in v] for k, v in d["local"].items()},
            diet=[BudgetItem(**x) for x in d["diet"]],
        )


# --------------- データ定義ここまで ---------------


def get_args():
    """
    コマンドライン引数を処理する関数．
    [-m]オプションで会議録データを指定し，
    [-b]オプションで予算項目リストデータを指定する．
    """
    parser = argparse.ArgumentParser(
        description="""Budget Argument Miningタスクの推論スクリプトサンプル．
ランダムにargumentClassとrelatedIDを設定する．"""
    )

    parser.add_argument("-m", "--minute", required=True, help="会議録データを指定します")
    parser.add_argument("-b", "--budget", required=True, help="予算項目リストデータを指定します")
    return parser.parse_args()


def load_minute(minute_path: str) -> MinutesObject:
    """
    指定したパスから会議録データを読み込み，MinutesObjectインスタンスとして返す．
    """
    p = Path(minute_path)
    return MinutesObject.from_dict(json.loads(p.read_text()))


def load_budget(budget_path: str) -> BudgetObject:
    """
    指定したパスから予算項目データを読み込み，BudgetObjectインスタンスとして返す．
    """
    p = Path(budget_path)
    return BudgetObject.from_dict(json.loads(p.read_text()))



#-----argumentClassをつける関数
def ac_inference(sentences,mex: MoneyExpression,giin):
    # データの符号化
    max_length = 430
    encoding = tokenizer(
        sentences, 
        max_length=max_length, 
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    #encoding["giin_flags_tensor"] = torch.tensor([[giin]])
    encoding = { k: v.cuda() for k, v in encoding.items() }

    # 推論
    with torch.no_grad():
        output = ac_bert_sc.forward(**encoding)
    scores = output.logits # 分類スコア
    labels_predicted = scores.argmax(-1) # スコアが最も高いラベル

    #print(ARGUMENT_CLASSES[labels_predicted])

    mex.argumentClass = ARGUMENT_CLASSES[labels_predicted]

def BAM_aC(sentences,proc: MoneyExpression,giin):

    sentence_idx = 0
    bsentence_idx = 0
    mex_idx=-1
    bmex_idx=0
    # 各金額表現の繰り返し処理
    for procidx,mex in enumerate(proc):
                
        while (True):
            if sentence_idx==bsentence_idx and procidx!=0:
                mex_idx = sentences[sentence_idx][bmex_idx:].find(mex.moneyExpression)
            else:
                mex_idx = sentences[sentence_idx].find(mex.moneyExpression)

            if mex_idx >= 0:
                if sentence_idx==bsentence_idx:
                    bmex_idx += (mex_idx+len(mex.moneyExpression))
                else:
                    bmex_idx = mex_idx
                bsentence_idx = sentence_idx
                mex_idx=-1
                break
            else:    
                sentence_idx += 1

        if mex.moneyExpression.find("円") < 0 and mex.moneyExpression.find("無料") < 0:
            if mex.moneyExpression.find("ドル") < 0:
                mex.argumentClass = ARGUMENT_CLASSES[5]
            else:
                mex.argumentClass = ARGUMENT_CLASSES[2]
                
        if mex.argumentClass == None:
            ac_inference(sentences[sentence_idx],mex,giin)

#-----ここまで



#-----relatedIDを付けるための関数
def new_idf(docfreq, totaldocs, log_base=2.0, add=5.0):
    return add + math.log(1.0 * totaldocs / docfreq, log_base)

def TFIDF(sentences):

    wakatilist=[]
    for si in sentences:
        for sj in si:
            w = [m.normalized_form() for m in tokenizer_obj.tokenize(sj, mode) 
            if m.part_of_speech()[0] == "名詞" and  m.part_of_speech()[1] != "数詞" and len(m.normalized_form())>2]
            wakatilist.append(w)

    dictionary = corpora.Dictionary(wakatilist)
    corpus = list(map(dictionary.doc2bow, wakatilist))
    test_model = models.TfidfModel(corpus,wglobal=new_idf)
    corpus_tfidf = test_model[corpus]
    texts_tfidf = []
    for doc in corpus_tfidf:
        text_tfidf = []
        for word in doc:
            text_tfidf.append([dictionary[word[0]], word[1]])
        texts_tfidf.append(text_tfidf)
    #print(texts_tfidf[0])
    
    s_tfidf = [[[] for i in range(len(sentences[j]))] for j in range(len(sentences))]
    l = 0
    for i in range(len(sentences)):
        for j in range(len(sentences[i])):
            s_tfidf[i][j] = texts_tfidf[l+j]
        l += j+1
    
    #print(s_tfidf)
    return s_tfidf

def budMA(text):
    tokens = tokenizer_obj.tokenize(text,mode)
    budtext = "".join([t.normalized_form() for t in tokens])
    return budtext

"""
def tokutyousimi(tokutyou):
    t = []
    for w in tokutyou:
        t.append(w)
    for w in tokutyou:
        if w[0] in chive_vectors:
            c = chive_vectors.most_similar(w[0],topn=3)
            for i in c:
                t.append([i[0],i[1]])          
    return t
"""

def BAM_rID(budgets,proc: MoneyExpression,sentences,s_tfidf):

    sentence_idx = 0
    bsentence_idx = 0
    mex_idx=-1
    bmex_idx=0
    # 各金額表現の繰り返し処理
    for procidx,mex in enumerate(proc):
                
        while (True):
            if sentence_idx==bsentence_idx and procidx!=0:
                mex_idx = sentences[sentence_idx][bmex_idx:].find(mex.moneyExpression)
            else:
                mex_idx = sentences[sentence_idx].find(mex.moneyExpression)

            if mex_idx >= 0:
                if sentence_idx==bsentence_idx:
                    bmex_idx += (mex_idx+len(mex.moneyExpression))
                else:
                    bmex_idx = mex_idx
                bsentence_idx = sentence_idx
                mex_idx=-1
                break
            else:    
                sentence_idx += 1


        #if ri_inference(sentences[sentence_idx]):
        tfidf = sorted(s_tfidf[sentence_idx], reverse=True, key=lambda x: x[1])
        s_tokutyogo=[]
        for idx,tfidf in enumerate(tfidf):
            #if idx == 0:
                #s_tokutyogo.append(tfidf)
            #elif tfidf[1] >= 0.6:
                #s_tokutyogo.append(tfidf)
            s_tokutyogo.append(tfidf)
            
        #s_tokutyogo = tokutyousimi(s_tokutyogo)
        #print(s_tokutyogo)
        #print(mex)
        #print(sentences[sentence_idx])
        #print()
        mexmemo=[]
        for s in s_tokutyogo:
            for budget in budgets:
                if mexmemo == []:
                    if budget["description"] != None:
                        if budget["description"].find(s[0])>=0:
                            if not (budget["budgetId"] in mexmemo):
                                mexmemo.append(budget["budgetId"])
                    if budget["budgetItem"].find(s[0])>=0:
                        if not (budget["budgetId"] in mexmemo):
                            mexmemo.append(budget["budgetId"])
            
        if mexmemo!=[]:
            mex.relatedID = mexmemo
                        
#-----ここまで


#文を区切る関数
def Textsplit(text):
    smemo = re.split(r'[、]',text)
    i = 0
    while(i < len(smemo)):
        s = smemo[i]
        if i != 0 and len(s)<=8:
            smemo[i-1] += ("、" + s)
            del smemo[i]
            i-=1
        i+=1

    sen = []   
    for s1 in smemo:
        a = re.split(r'[。\n\r　]',s1)    
        for s2 in a:
            if s2 != "":
                sen.append(s2)
    #print(sen)
    return sen

def Textsplit2(text):

    sen = []   
    a = re.split(r'[。\n\r　]',text)    
    for s2 in a:
        if s2 != "":
            sen.append(s2)
    #print(sen)
    return sen  
#----------


def estimate_local(minutesObj: MinutesObject, budgetObj: BudgetObject):
    """
    地方議会を対象としたargumentClassとrelatedIDの推論を行う．
    """

    budgetlocal = budgetObj.local
    bud = {}
    for b1k,b1v in budgetlocal.items():
        blist=[]
        for b2 in tqdm(b1v):
            bdict={}
            bdict["budgetId"] = b2.budgetId
            bdict["budgetItem"] = budMA(b2.budgetItem)
            bdict["description"] = budMA(b2.description)

            blist.append(bdict)
        
        bud[b1k] = blist


    # 地方議会会議録の各会議で繰り返し処理
    for proc_obj in tqdm(minutesObj.local):
        # その自治体の予算項目リストを取得
        budgets = bud[proc_obj.localGovernmentCode]

        # 会議の年
        year = proc_obj.date.split("-")[0]

        # 会議の年と対応する予算項目リストを抽出
        budgets_filtered = [x for x in budgets if x["budgetId"].split("-")[1] == year]

        seall=[]
        for proc in proc_obj.proceeding:
            #seall.append(Textsplit2(proc.utterance))
            sen = Textsplit2(proc.utterance)
            i = 0
            while(i < len(sen)):
                s = sen[i]
                if i < len(sen)-1 and len(s)<=200:
                    sen[i] += ("。" + sen[i+1])
                    del sen[i+1]  
                i+=1
            seall.append(sen) 
        #print(seall)

        seall_tfidf = TFIDF(seall)

        # 各発言の繰り返し処理
        for procidx,proc in enumerate(proc_obj.proceeding):

            if proc.moneyExpressions== []:
                continue
            
            sentences = Textsplit(proc.utterance)
            sentences2 = Textsplit2(proc.utterance)
            i = 0
            while(i < len(sentences2)):
                s = sentences2[i]
                if i < len(sentences2)-1 and len(s)<=200:
                    sentences2[i] += ("。" + sentences2[i+1])
                    del sentences2[i+1]  
                i+=1
            #print(sentences2)
            
            if proc.speakerPosition=="議員":
                giin = 1
            else:
                giin = 0


            #argumentClassを付ける
            BAM_aC(sentences,proc.moneyExpressions,giin)

            #relatedIDを設定する
            BAM_rID(budgets_filtered,proc.moneyExpressions,sentences2,seall_tfidf[procidx])
            


def estimate_diet(minutesObj: MinutesObject, budgetObj: BudgetObject):
    """
    国会を対象としたargumentClassとrelatedIDの推論を行う．
    """

    # 予算項目リストを取得
    budgets = budgetObj.diet
    bud = []
    for bf in budgets:
        b={}
        b["budgetId"] = bf.budgetId
        b["budgetItem"] = budMA(bf.budgetItem)
        b["description"] = budMA(bf.description)
        bud.append(b)

    # 国会会議録の各会議で繰り返し処理
    for proc_obj in tqdm(minutesObj.diet):
        
        year = proc_obj.date.split("-")[0]

        seall=[]
        for sp in proc_obj.speechRecord:
            #seall.append(Textsplit2(sp.speech))
            sen = Textsplit2(sp.speech)
            i = 0
            while(i < len(sen)):
                s = sen[i]
                if i < len(sen)-1 and len(s)<=200:
                    sen[i] += ("。" + sen[i+1])
                    del sen[i+1]  
                i+=1
            seall.append(sen) 

        seall_tfidf = TFIDF(seall)

        # 各発言の繰り返し処理
        for spidx,sp in enumerate(proc_obj.speechRecord):

            if sp.moneyExpressions== []:
                continue
            
            sentences = Textsplit2(sp.speech)
            sentences2 = Textsplit2(sp.speech)
            i = 0
            while(i < len(sentences2)):
                s = sentences2[i]
                if i < len(sentences2)-1 and len(s)<=200:
                    sentences2[i] += ("。" + sentences2[i+1])
                    del sentences2[i+1]  
                i+=1
            #svector = bunsyobekutoru(sentences)
            if sp.speakerPosition==None and sp.speech[:10].find("委員長")<0:
                giin = 1
            else:
                giin = 0

            #argumentClassを付ける
            BAM_aC(sentences,sp.moneyExpressions,giin)

            #relatedIDを設定する
            BAM_rID(bud,sp.moneyExpressions,sentences2,seall_tfidf[spidx])


# --------------- main ---------------
if __name__ == "__main__":
    # コマンドライン引数の解析
    #args = get_args()

    # 会議録の読み込み
    minutefile = os.path.join(os.path.dirname(__file__), 'BAMData/BAMData2.json')
    minutesObj = load_minute(str(minutefile))

    # 予算項目リストの読み込み
    budgetfile = os.path.join(os.path.dirname(__file__), 'BAMData/PoliInfo3_BAM-budget.json')
    budgetObj = load_budget(str(budgetfile))

    # 地方議会会議録の金額表現に対して推論
    estimate_local(minutesObj, budgetObj)

    # 国会会議録の金額表現に対して推論
    estimate_diet(minutesObj, budgetObj)

    # 標準出力に結果を出力
    resultfile = os.path.join(os.path.dirname(__file__), 'BAMData/testresult2.json')
    with open(resultfile, mode='wt') as f:
        print(json.dumps(dataclasses.asdict(minutesObj), ensure_ascii=False, indent=4),file=f)