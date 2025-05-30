# ───────────────────────────────────────────
#  model_best.py  –  one-shot runnable version
# ───────────────────────────────────────────
import os, sys, logging, numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader
from glob import glob
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
from monai.transforms import (Compose, EnsureChannelFirstD, LoadImaged,
                               Resized, ToTensord, Spacingd,
                               ScaleIntensityRanged)
from monai.utils import set_determinism
from monai.data  import Dataset

# ─── backbone 路徑 ───
sys.path.append("/home/u3861345/data_50/MedicalNet/models")
from resnet import resnet50            # MedicalNet-ResNet50

# ─── 基本路徑 ───
DATA_DIR   = "/home/u3861345/data_50/data_train_test_2"
LABEL_CSV  = "/home/u3861345/data_50/data_train_test_2/labels.csv"
PRETRAIN_W = "/home/u3861345/MedicalNet-master/pretrain/resnet_200.pth"

# ─── logging ───
logging.basicConfig(filename='/home/u3861345/data_50/log_data/model_best_1.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ───────── 损失 ─────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=.25, gamma=2):
        super().__init__(); self.a=alpha; self.g=gamma
    def forward(self,x,y):
        bce = F.binary_cross_entropy_with_logits(x,y,reduction='none')
        pt  = torch.exp(-bce)
        at  = self.a*y + (1-self.a)*(1-y)
        return (at*(1-pt)**self.g*bce).mean()

def mixed_loss(o,y):                   # BCE + focal
    return .7*nn.BCEWithLogitsLoss()(o,y) + .3*FocalLoss()(o,y)

# ───────── Dataset ─────────
def prepare(root, csv,
            pixdim=(1,1,1), a_min=-200, a_max=200,
            size=(128,128,128)):
    set_determinism(0)
    df = pd.read_csv(csv, header=None,
                     names=['pid','tumor','smoke','betel','drink','label'])

    def collect(split):
        out=[]
        for f in sorted(glob(os.path.join(root,f"{split}Volumes","*.nii.gz"))):
            pid = os.path.basename(f)[:-7]
            r   = df[df.pid==pid]
            if r.empty: continue
            r=r.iloc[0]
            out.append(dict(
                vol=f,
                label=r.label,
                tumor=r.tumor,
                smoke=r.smoke,
                betel=r.betel,
                drink=r.drink
            ))
        return out

    keys=["vol","label","tumor","smoke","betel","drink"]
    tfm=Compose([
        LoadImaged(keys=["vol"]),
        EnsureChannelFirstD(keys=["vol"]),
        Spacingd(keys=["vol"],pixdim=pixdim,mode="bilinear"),
        ScaleIntensityRanged(keys=["vol"],a_min=a_min,a_max=a_max,
                             b_min=0,b_max=1,clip=True),
        Resized(keys=["vol"],spatial_size=size),
        ToTensord(keys=keys)
    ])
    return Dataset(collect("Train"),tfm), Dataset(collect("Test"),tfm)

# ───────── Model ─────────
def GN(c): return nn.GroupNorm(1,c)    # insta-style GN

class Net(nn.Module):
    def __init__(self, pth):
        super().__init__()
        self.backbone = resnet50(sample_input_D=128,
                                 sample_input_H=128,
                                 sample_input_W=128,
                                 num_seg_classes=512,
                                 shortcut_type='B', no_cuda=False)

        # ---- clinical branch ----
        self.cli = nn.Sequential(
            nn.Linear(4,64),  GN(64), nn.ReLU6(True), nn.Dropout(.2),
            nn.Linear(64,64), GN(64), nn.ReLU6(True), nn.Dropout(.2),
            nn.Linear(64,32), GN(32), nn.ReLU6(True)
        )

        # ---- classification head ----
        self.cls = nn.Sequential(
            nn.Linear(512+32,256), GN(256), nn.ReLU6(True), nn.Dropout(.3),
            nn.Linear(256,128),    GN(128), nn.ReLU6(True), nn.Dropout(.25),
            nn.Linear(128,64),     GN(64),  nn.ReLU6(True), nn.Dropout(.2),
            nn.Linear(64,32),      GN(32),  nn.ReLU6(True),
            nn.Linear(32,1)
        )

        if os.path.exists(pth):
            ck=torch.load(pth,map_location='cpu')
            self.backbone.load_state_dict(ck['state_dict'],strict=False)
            print("✅ pre-trained backbone loaded")

        nn.init.constant_(self.cls[-1].bias, -np.log(3))   # p=0.25

    def forward(self,img,clin):
        x = self.backbone(img)
        x = F.adaptive_avg_pool3d(x,1).flatten(1)   # (B,512)
        c = self.cli(clin)                          # (B,32)
        return self.cls(torch.cat([x,c],1))

# ───────── helper ─────────
def to_dev(batch, dev):
    img = batch["vol"].to(dev)

    def _tensor(v):
        if torch.is_tensor(v): return v.float()
        v=np.asarray(v,dtype=float); return torch.tensor(v,dtype=torch.float32)

    cli = torch.stack([_tensor(batch[k]).view(-1) for k in
                       ("tumor","smoke","betel","drink")],1).to(dev)
    lbl = _tensor(batch["label"]).view(-1,1).to(dev)
    return img, cli, lbl

def train_epoch(dl, net, opt, dev):
    net.train(); tot=0
    for b in dl:
        img,cli,y = to_dev(b,dev)
        opt.zero_grad()
        loss = mixed_loss(net(img,cli),y)
        loss.backward(); opt.step(); tot+=loss.item()
    return tot/len(dl)

@torch.no_grad()
def evaluate(dl, net, dev):
    net.eval(); logits, labels = [],[]
    for b in dl:
        img,cli,y = to_dev(b,dev)
        o = net(img,cli).squeeze(1)
        logits += o.cpu().tolist(); labels += y.cpu().squeeze(1).tolist()

    logits = np.array(logits); labels = np.array(labels)
    preds  = (logits>=0).astype(int)           # logit 0 ≈ prob 0.5
    acc = (preds==labels).mean()
    f1  = f1_score(labels,preds,zero_division=0)
    auc = roc_auc_score(labels,logits) if len(np.unique(labels))>1 else None
    cm  = confusion_matrix(labels,preds)
    return acc,f1,auc,cm

# ───────── main ─────────
if __name__=="__main__":
    tr_ds, te_ds = prepare(DATA_DIR, LABEL_CSV)
    tr_dl = DataLoader(tr_ds, batch_size=2, shuffle=True)
    te_dl = DataLoader(te_ds, batch_size=2, shuffle=False)
    dev   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = Net(PRETRAIN_W).to(dev)

    # ───── 冻結 backbone 前 5 epoch ─────
    head = list(net.cli.parameters())+list(net.cls.parameters())
    base = [p for p in net.backbone.parameters()]
    for p in base: p.requires_grad=False

    opt = optim.AdamW([
        {"params": head, "lr":1e-3},
        {"params": base, "lr":1e-5}
    ], weight_decay=1e-4)

    EPOCHS = 100; best_f1=0; patience, bad=10,0
    for ep in range(1,EPOCHS+1):
        if ep==6:                                # un-freeze
            for p in base: p.requires_grad=True
            opt.param_groups[1]["lr"]=1e-4

        loss = train_epoch(tr_dl,net,opt,dev)
        logging.info(f"Epoch {ep}/{EPOCHS} loss {loss:.4f}")

        if ep%10==0:
            acc,f1,auc,cm = evaluate(te_dl,net,dev)
            logging.info(f"eval acc={acc:.2f} f1={f1:.2f} "
                         f"auc={auc if auc else 'NA'}\n{cm}")

            # early-stop
            if f1>best_f1: best_f1,fname=f1,f"best_{f1:.2f}.pth"; bad=0
            else:          bad+=1
            if bad>patience:
                logging.info("Early-stopping"); break

    torch.save(net.state_dict(),"model_best_final.pth")
