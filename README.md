# qa_match
Deep learning model for match medical QA 



# IDEA 1 :
give more match count example (but not relevant answer) more weight to loss

step 1:
  抽樣10000筆 從這些資料中觀察大概重疊字的分布
  結巴斷詞   2.8276
  2 gram and 3 gram(不含1gram) 2.4723
  因為沒有包含1gram所以反而比斷詞重疊還要少
  
  tfidf的hit rate(邱老師)
  hit rate @1 = 0.632
  hit rate @2 = 0.763
  hit rate @3 = 0.816
  hit rate @4 = 0.895
  hit rate @5 = 1.000
  
  台灣e院
  hit rate @1 = 0.767
  hit rate @2 = 0.874
  hit rate @3 = 0.920
  hit rate @4 = 0.946
  hit rate @5 = 0.970
  
  尋醫問藥
  hit rate @1 = 0.535
  hit rate @2 = 0.638
  hit rate @3 = 0.690
  hit rate @4 = 0.723
  hit rate @5 = 0.747
  
step 2:
  從step1的結果中 設定超參數(給比較多match的負樣本的權重)
