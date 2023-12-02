# PositivityBoundsByNeuralNetwork

## 現状
lib/attempt4.pyというファイルが現行のコード

損失関数がうまく収束せず、結果がうまく出力できていない状況

損失関数の中に拘束条件項があり、その重みづけのパラメータの値を色々変えてみるなど試したが、大抵の場合で損失関数の値が0かinfとなってしまう


## フォルダ構成
- lib/ : 本番ファイル
- practice/ : 練習、試行用

## 計画

1. [ExtremalEFT論文](https://arxiv.org/abs/2011.02957)の(g3,g4) plotをNNに描かせる: 
教師なし PINN
null constraintは損失関数に加えることで考慮

2. High Spin Supressionを考えてみる: 
spectral densityに対するconstraintを扱うことができるはず


## 期待されるメリット

SDPBとの比較

- メッシュの細かさを自動的に最適化してくれる

- constraintの扱いが簡単 損失関数に加えるだけ