## はじめに

2017年度の京都大学医学部附属病院と野間研＠立命で行う予定の、keras勉強会の演習資料を共有するためのプロジェクトです。

[](
## 日程

日付や日時は変更する可能性があります，

|Trial|Date|Description|
|-------|-----|----|
|第一回|2016/05/13|昨年度夏季合宿における深層学習レクチャの復習。学習環境の構築と盲目的な実行|
|第二回|2016/05/20|MLP(多層パーセプトロン)のソースコード解説。課題発表|
|第三回|2016/05/27|SdA（Stacked denoising Autoencoder）のソースコード解説。課題発表|
|休み|2016/06/03|杉山出張のためお休み|
|休み|2016/06/10|杉山出張のためお休み|
|第四回|2016/06/17|CNN（畳み込みニューラルネットワーク）のソースコード解説。課題発表|
|第五回|2016/06/24|データセットの作り方解説。画像学習用ベンチマークCifarでデータセットを作ってみよう|
|第六回|2016/07/01|学習結果や、学習過程を可視化方法の解説。深層学習の中で何が動いているのか見てみよう|
)

## 概要

立命と京大で行うKeras勉強会のサンプルコードとかもろもろを含んだリポジトリです。わしはchainerで
深層学習やるねん！って人はchainerリポジトリにサンプルコードがあるのでそっちを見てください。

## ルール

### 連絡先

大槻への連絡はrotsuki@mxdlab.netまで。

### 演習資料
誠意作成中です。
[](
演習資料を以下の命名規則にしたがって作成し、当日までにWikiにアップロードします。なお、２回目以降はパワーポイントファイルやpdfではなく、Wikiページとしてアップしますので、リンクを辿って下さい。

```
DeepLearning-<Trial Number>-<Date>.(pptx|pdf)
```

* [第一回演習資料](https://github.com/medinfo2/deeplearning/wiki/files/DeepLearning-1-20160513.pptx)
)

### 課題

課題で作成したプログラムは、gitでbranchを作ってお互いに参照できるようにすると良いと思います。以下、ブランチの作り方とリポジトリへの反映方法です。

```
$ git checkout -b deeplearning-<username>-<Trial Number>
...<実際の作業>...
$ git commit -a -m "<comment>"
$ git push origin deeplearning-<username>-<Trial Number>
$ git checkout master
```

### 環境のセットアップ

Homebrewとpyenvを利用すると比較的すぐ作業環境を構築することができます。

```
$ /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
$ brew update
$ brew install pyenv
To use Homebrew's directories rather than ~/.pyenv add to your profile:
  export PYENV_ROOT=/usr/local/var/pyenv

To enable shims and autocompletion add to your profile:
  if which pyenv > /dev/null; then eval "$(pyenv init -)"; fi
$ echo "export PYENV_ROOT=/usr/local/var/pyenv; " >> ~/.bash_profile
$ echo "if which pyenv > /dev/null; then eval $(pyenv init -); fi" >> ~/.bash_profile
$ source ~/.bash_profile
$ pyenv install anaconda-2.4.0
$ git clone https://github.com/medinfo2/deeplearning.git
$ cd deeplearning
$ pyenv local anaconda-2.4.0
$ pip install keras
```

既にbrewをインストールしていたり、pyenvを使わず自分の環境を使いたい人は適宜、手順を省いて下さい。

### 参考文献

* [元にしたソースコードmlp](https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py)
* [Homebrew](http://brew.sh/index_ja.html)
