# ICMR 2024 - A Multi-Stage Deep Learning Approach for Cheapfake Detection Incorporating Text-Image and Image-Image Comparisons

This is the source code for the Cheapfake challenge. </br>
The title of the paper is 'A Multi-Stage Deep Learning Approach for Cheapfake Detection Incorporating Text-Image and Image-Image Comparisons'

## Getting Started

The environment settings are as follows.


```
conda create -n icmr python=3.8
```

```
conda activate icmr
```

pip install the packages
```
pip install -r requirements.txt
```

### Structure of directory
```
├── datasets
│   ├── public_test_acm.json
│   ├── test (Image folder)

```
### Task 1

The path parameter must be entered.

```
python task1.py --path /your_dataset_path/
```

### Task 2

The path parameter must be entered.

```
python task2.py --path /your_dataset_path/
```
## Overall Process

<img src="/src/overall.png" />

## Performance
### Task 1 
|Model|Accuracy	|Precision	|F1|
| :--: | :--: | :--: |:--: |
|NLI + Detection RoI|	0.672|	0.608	|0.717|
|<span style="color:#FF87CEEB">Our Methods</span>|	0.719|	0.650|	0.743|

### Task 2

|Model|Accuracy	|Precision	|F1|
| :--: | :--: | :--: |:--: |
|<span style="color:#FF87CEEB">Our Methods</span>|	0.557|	0.531|	0.637|


## Authors

* **Jangwon Seo** - *School of Electrical Engineering, Korea University* - jwein307@korea.ac.kr
* **Hyo-Seok Hwang** - *School of Electrical Engineering, Korea University* - shdhkj960@korea.ac.kr
* **Jiyoung Lee** - *Safe AI* - jiyoung.lee@safeai.kr
* **Minhyeok Lee** - *School of Electrical and Electronics Engineering, Chung-Ang University* - mlee@cau.ac.kr
* **Wonsuk Kim** - *Safe AI* - wonderit@safeai.kr
* **Junhee Seok** - *School of Electrical Engineering, Korea University* - jseok14@korea.ac.kr
