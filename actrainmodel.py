#議員フラグなしのモデル作成
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

#from sklearn.manifold import TSNE
#from sklearn.decomposition import PCA

import random
import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer, BertModel,BertForSequenceClassification
import pytorch_lightning as pl

tokenizer_obj = dictionary.Dictionary().create()
mode = tokenizer.Tokenizer.SplitMode.C

MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'

# トークナイザとモデルのロード
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)

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


#---学習のクラス
class BertForSequenceClassification_pl(pl.LightningModule):
        
    def __init__(self, model_name, num_labels, lr):
        # model_name: Transformersのモデルの名前
        # num_labels: ラベルの数
        # lr: 学習率

        super().__init__()
        
        # 引数のnum_labelsとlrを保存。
        # 例えば、self.hparams.lrでlrにアクセスできる。
        # チェックポイント作成時にも自動で保存される。
        self.save_hyperparameters() 

        # BERTのロード
        self.bert_sc = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        
    # 学習データのミニバッチ(`batch`)が与えられた時に損失を出力する関数を書く。
    # batch_idxはミニバッチの番号であるが今回は使わない。
    def training_step(self, batch, batch_idx):
        output = self.bert_sc(**batch)
        loss = output.loss
        self.log('train_loss', loss) # 損失を'train_loss'の名前でログをとる。
        return loss

    # テストデータのミニバッチが与えられた時に、
    # テストデータを評価する指標を計算する関数を書く。
    def test_step(self, batch, batch_idx):
        labels = batch.pop('labels') # バッチからラベルを取得
        output = self.bert_sc(**batch)
        labels_predicted = output.logits.argmax(-1)
        num_correct = ( labels_predicted == labels ).sum().item()
        accuracy = num_correct/labels.size(0) #精度
        self.log('accuracy', accuracy) # 精度を'accuracy'の名前でログをとる。

    # 検証データのミニバッチが与えられた時に、
    # 検証データを評価する指標を計算する関数を書く。
    def validation_step(self, batch, batch_idx):
        output = self.bert_sc(**batch)
        val_loss = output.loss
        self.log('val_loss', val_loss) # 損失を'val_loss'の名前でログをとる。

    # 学習に用いるオプティマイザを返す関数を書く。
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    
checkpoint = pl.callbacks.ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_top_k=1,
    save_weights_only=True,
    dirpath='model/',
)

# 学習の方法を指定
trainer = pl.Trainer(
    gpus=1, 
    max_epochs=10,
    callbacks = [checkpoint]
)
#--------------


#-----argumentClassをつける関数
def BAM_aC(sentences,proc: MoneyExpression):

    dataset = []
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
        
        max_length = 430
        encoding = tokenizer(
            sentences[sentence_idx],
            max_length=max_length, 
            padding='max_length',
            truncation=True
        )
        encoding['labels'] = ARGUMENT_CLASSES.index(mex.argumentClass)
        encoding = { k: torch.tensor(v) for k, v in encoding.items() }
        dataset.append(encoding)
    
    return dataset
        
#-----ここまで


#----------

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


def estimate_local(minutesObj: MinutesObject):
    """
    地方議会を対象としたargumentClassとrelatedIDの推論を行う．
    """
    dataset = []
    dmemo=[]
    # 地方議会会議録の各会議で繰り返し処理
    for proc_obj in tqdm(minutesObj.local):

        dmemo2=[]
        # 各発言の繰り返し処理
        for procidx,proc in enumerate(proc_obj.proceeding):

            if proc.moneyExpressions== []:
                continue
            #sentences = re.split(r'[、。\n]',proc.utterance)
            #sentences2 = re.split(r'[\n]',proc.utterance)
            sentences = Textsplit(proc.utterance)

            #argumentClass
            for d in BAM_aC(sentences,proc.moneyExpressions):
                dmemo2.append(d)

        for d in dmemo2:
            dmemo.append(d)
    
    for d in dmemo:
        dataset.append(d)
    #print(dataset_for_loader[1])

    return dataset
    
    

def estimate_diet(minutesObj: MinutesObject):

    dataset = []
    dmemo=[]
    # 地方議会会議録の各会議で繰り返し処理
    for proc_obj in tqdm(minutesObj.diet):
        
        dmemo2=[]
        # 各発言の繰り返し処理
        for spidx,sp in enumerate(proc_obj.speechRecord):

            if sp.moneyExpressions== []:
                continue
            
            sentences = Textsplit2(sp.speech)

            #argumentClassを付ける
            for d in BAM_aC(sentences,sp.moneyExpressions):
                dmemo2.append(d)

        for d in dmemo2:
            dmemo.append(d)
    
    for d in dmemo:
        dataset.append(d)
    
    return dataset


# --------------- main ---------------
if __name__ == "__main__":
    # コマンドライン引数の解析
    #args = get_args()

    # 会議録の読み込み
    minutefile = os.path.join(os.path.dirname(__file__), 'BAMData/PoliInfo3_BAM-minutes-training.json')
    minutesObj = load_minute(str(minutefile))

    # 予算項目リストの読み込み
    #budgetfile = os.path.join(os.path.dirname(__file__), 'BAMData/PoliInfo3_BAM-budget.json')
    #budgetObj = load_budget(str(budgetfile))


    dataset_for_loader=[]
    # 地方議会会議録の金額表現に対して推論
    for d in estimate_local(minutesObj):
        dataset_for_loader.append(d)

    # 国会会議録の金額表現に対して推論
    for d in estimate_diet(minutesObj):
        dataset_for_loader.append(d)
    #print(dataset_for_loader[1])

    random.shuffle(dataset_for_loader) # ランダムにシャッフル
    n = len(dataset_for_loader)
    n_train = int(0.85*n)
    #n_val = int(0.1*n)
    dataset_train = dataset_for_loader[:n_train] # 学習データ
    dataset_val = dataset_for_loader[n_train:] # 検証データ
    #dataset_val = dataset_for_loader[n_train:n_train+n_val] # 検証データ
    #dataset_test = dataset_for_loader[n_train+n_val:] # テストデータ

    dataloader_train = DataLoader(
        dataset_train, batch_size=20, shuffle=True
    )
    dataloader_val = DataLoader(dataset_val, batch_size=256)
    #dataloader_test = DataLoader(dataset_test, batch_size=256)

    model = BertForSequenceClassification_pl(
        MODEL_NAME, num_labels=7, lr=1e-5
    )
    trainer.fit(model, dataloader_train, dataloader_val) 

    best_model_path = checkpoint.best_model_path

    #test = trainer.test(test_dataloaders=dataloader_test)
    #print(f'Accuracy: {test[0]["accuracy"]:.2f}')
    #print("--------------\n")

    model = BertForSequenceClassification_pl.load_from_checkpoint(
        best_model_path
    ) 
    model.bert_sc.save_pretrained('BAM/ac_model_transformers') 