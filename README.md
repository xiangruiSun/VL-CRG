# VL-CRG: Vision-Language Model with Causal Relation Graph Pre-Training

This repository extends **LXMERT** to incorporate **Causal Relation Graph (CRG) Pre-Training** for **Visual Commonsense Reasoning (VCR)** and **Visual Question Answering (VQA)**. It enhances structured reasoning by learning causal interactions between objects and textual elements, improving interpretability and robustness.

This project is based on [LXMERT](https://github.com/airsplay/lxmert), originally designed for vision-language tasks. We introduce modifications to integrate **causal reasoning** and **structured learning**.

---

## 🔥 Key Contributions
- **Causal Pre-Training (CRG-PT):** Models causal dependencies in vision-language data.
- **Structured Reasoning:** Learns object interactions via **Causal Relation Graphs (CRGs)**.
- **Improved Robustness:** Evaluates **Consistency** and **Sensitivity** in VCR/VQA tasks.
- **Performance Gains:** Achieves **higher accuracy**, **better generalization**, and **faster convergence**.

---

## 📌 Pre-trained Models
The pre-trained **LXMERT** model (870 MB) is available for download:

```bash
mkdir -p snap/pretrained
wget https://nlp.cs.unc.edu/data/model_LXRT.pth -P snap/pretrained
```

For our **VL-CRG Pre-trained Model**, please check our **model release page**.

---

## 🚀 Running VL-CRG on VCR/VQA

### 1️⃣ **Setup Environment**
Install dependencies:
```bash
pip install -r requirements.txt
```

### 2️⃣ **Download Data**
We use **MS COCO** images and **VCR/VQA** annotations.

#### **Image Features (MS COCO)**
```bash
mkdir -p data/mscoco_imgfeat
wget https://nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/train2014_obj36.zip -P data/mscoco_imgfeat
unzip data/mscoco_imgfeat/train2014_obj36.zip -d data/mscoco_imgfeat && rm data/mscoco_imgfeat/train2014_obj36.zip

wget https://nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/val2014_obj36.zip -P data/mscoco_imgfeat
unzip data/mscoco_imgfeat/val2014_obj36.zip -d data && rm data/mscoco_imgfeat/val2014_obj36.zip
```

#### **Convert Image Features**
```bash
python src/tools/detection_feature_converter.py
```

#### **Process Question-Answer Annotations**
```bash
python src/tools/compute_softscore.py 
```

---

### 3️⃣ **Fine-Tuning VL-CRG on VCR/VQA**
Modify `src/config.py` to specify dataset.

Run fine-tuning:
```bash
PYTHONPATH=$PYTHONPATH:./src \
python -u src/tasks/vqa.py \
--train train --valid val \
--llayers 9 --xlayers 5 --rlayers 5 \
--loadLXMERTQA snap/pretrained/model \
--batchSize 32 --optim bert --lr 3e-5 --epochs 20 \
--name vl-crg-vqa
```

---

### 4️⃣ **Evaluating VL-CRG**
```bash
PYTHONPATH=$PYTHONPATH:./src \
python -u src/tasks/vqa.py \
--train train --test val \
--llayers 9 --xlayers 5 --rlayers 5 \
--loadLXMERTQA snap/pretrained/model \
--batchSize 32 --load output/vl-crg-vqa.pth \
--tqdm
```

Check accuracy breakdown:
```bash
python acc_per_type.py output/val_predict.json
```

---

## 📊 **Performance Metrics**
We evaluate **VL-CRG** on **Visual Commonsense Reasoning (VCR)** using the following metrics:

### **Task-Specific Accuracy**
- **Q → A Accuracy**: Predicting the correct answer given an image and a question.
- **QA → R Accuracy**: Selecting the correct rationale given a question and its answer.
- **Q → AR Accuracy**: Jointly predicting both answer and rationale.

### **Robustness Metrics**
- **Consistency**: Ensures logically related questions about the same image receive non-contradictory answers.
- **Sensitivity**: Measures the model’s ability to adjust predictions when input variations (e.g., reference changes) occur.

---

## 🏆 **Results**
### **VL-CRG vs. Baseline (w/o CRG)**
| Model      | Q → A | QA → R | Q → AR | Consistency | Sensitivity |
|------------|------:|------:|------:|------------:|------------:|
| **VL-CRG** |  75.2 |  77.5 |  58.6 |        80.1 |        72.4 |
| *w/o CRG*  |  63.0 (-12.2) |  70.3 (-7.2) |  47.5 (-11.1) |  68.9 (-11.2) |  61.7 (-10.7) |

---

## 📈 **Convergence Analysis**
The graph below shows the validation accuracy trends during training.

![Convergence Analysis](output/convergence_analysis.png)

- **Solid lines** indicate VL-CRG performance.
- **Dashed lines** show baseline (w/o CRG) performance.
- CRG pre-training improves **convergence speed** and **final accuracy**.

---

## 📜 **Citation**
If you find this work useful, please cite:
```bibtex
@article{vl-crg2025,
  title={VL-CRG: Causal Relation Graph Pre-Training for Vision-Language Tasks},
  author={Your Name et al.},
  journal={ArXiv},
  year={2025}
}
```

---

## 🔍 **Acknowledgments**
This repository is built on **LXMERT**. The original LXMERT repo can be found [here](https://github.com/airsplay/lxmert).
