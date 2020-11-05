import os
import pandas as pd
import numpy as np
def download_datas():
    if not os.path.isfile("./datas/ratings_train.txt"):
        logging.info("Download Datas")
        os.mkdir("./datas")
        urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="./datas/ratings_train.txt")
        urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="./datas/ratings_test.txt")

def get_datas():
    train_set = pd.read_table("./datas/ratings_train.txt")
    test_set = pd.read_table("./datas/ratings_test.txt")

    train_set['document'] = train_set['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
    train_set['document'].replace('', np.nan, inplace=True)
    train_set = train_set.dropna(how = 'any')
    train_x, train_y = train_set['document'], train_set['label']


    test_set.drop_duplicates(subset = ['document'], inplace=True) # document 열에서 중복인 내용이 있다면 중복 제거
    test_set['document'] = test_set['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","") # 정규 표현식 수행
    test_set['document'].replace('', np.nan, inplace=True) # 공백은 Null 값으로 변경
    test_set = test_set.dropna(how='any') # Null 값 제거

    test_x, test_y = test_set['document'], test_set['label']
    return train_x, train_y, test_x, test_y
