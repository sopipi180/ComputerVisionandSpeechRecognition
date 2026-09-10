*Computer Vision and Speech Recognition* 
This is my submission for the individual project for Computer Vision and Speech Recognition class at EADA. This repository currently contains the Session 1 project setup and exploratory data analysis (EDA) of the MILK10k dataset.

Skin cancer is one of the most common cancers, and outcomes depend a lot on catching malignant lesions early. In practice a dermatologist looks at a suspicious lesion, and decides whether it needs to be biopsied or can be left alone. The goal of this project is to build a computer vision system that supports that decision... given one or both images of a lesion, predict its diagnosis.


The dataset — MILK10k
Source: https://api.isic-archive.com/doi/milk10k/ (described in detail in J Invest Dermatol, doi 10.1016/j.jid.2025.06.1594)

*How to Run*
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

python Session1/session1.py       # first run downloads the dataset (~345 MB)
python Session1/practice_ex1.py   # homework exercises
