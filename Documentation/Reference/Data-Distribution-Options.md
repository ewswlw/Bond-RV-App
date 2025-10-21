# Data Distribution Strategy - Options Analysis

**Document Version**: 1.0  
**Date**: October 21, 2025  
**Status**: Reference Document

---

## The Problem

- Raw Excel files (~17MB compressed) and Parquet outputs (~3MB) are too large for Git
- Multiple computers need access to both raw data and processed parquet files
- Need to maintain version control for code while handling large data files

---

## Option 1: Git LFS (Git Large File Storage) ⭐

### Pros:
- ✅ Everything in one repo (code + data)
- ✅ Version control for data files
- ✅ Simple `git pull` workflow
- ✅ GitHub supports it natively

### Cons:
- ⚠️ GitHub LFS has storage/bandwidth limits (1GB free, then paid)
- ⚠️ Requires Git LFS installation on each machine

### Implementation:
```bash
# One-time setup in repo
git lfs install
git lfs track "*.xlsx"
git lfs track "*.parquet"
git add .gitattributes
git commit -m "Add Git LFS tracking"
git push

# On other computers
git lfs install
git clone https://github.com/ewswlw/Bond-RV-App.git
# LFS files download automatically
```

### Cost:
- Free tier: 1GB storage, 1GB/month bandwidth
- Paid: $5/month for 50GB storage, 50GB bandwidth

---

## Option 2: Separate Data Repository

### Pros:
- ✅ Clean separation of code and data
- ✅ Can use Git LFS just for data repo
- ✅ Code repo stays lightweight

### Cons:
- ⚠️ Two repos to manage
- ⚠️ Need to sync both repos

### Structure:
```
Bond-RV-App/              # Code repo (current)
Bond-RV-App-Data/         # Data repo (new)
  ├── raw/
  │   └── Universe Historical/*.xlsx
  └── processed/
      └── parquet/*.parquet
```

### Workflow:
```bash
# Clone both repos
git clone https://github.com/ewswlw/Bond-RV-App.git
git clone https://github.com/ewswlw/Bond-RV-App-Data.git

# Symlink or configure path
cd Bond-RV-App
python pipeline.py -i "../Bond-RV-App-Data/raw/Universe Historical/" -m append
```

---

## Option 3: Cloud Storage (S3/Google Drive/Dropbox) + Git ⭐ SELECTED

### Pros:
- ✅ Unlimited storage (relatively cheap)
- ✅ Fast sync across machines
- ✅ Git repo stays lightweight
- ✅ Can use existing cloud storage

### Cons:
- ⚠️ Requires cloud storage account
- ⚠️ Manual sync or scripting needed
- ⚠️ Not version controlled

### Implementation:
```
# Store data in cloud
Dropbox: ~/Dropbox/Bond-RV-App-Data/
  └── Universe Historical/*.xlsx

# Clone repo and run pipeline
git clone https://github.com/ewswlw/Bond-RV-App.git
cd Bond-RV-App
python bond_pipeline/pipeline.py -i "~/Dropbox/Bond-RV-App-Data/Universe Historical/" -m override
```

### Workflow:
```bash
git clone https://github.com/ewswlw/Bond-RV-App.git
cd Bond-RV-App
# Point pipeline to Dropbox folder
python bond_pipeline/pipeline.py -i "~/Dropbox/Bond-RV-App-Data/Universe Historical/" -m override
```

---

## Option 4: Hybrid - Git LFS for Parquet Only, Cloud for Raw

### Pros:
- ✅ Parquet files (small, frequently used) in Git LFS
- ✅ Raw Excel files (large, rarely change) in cloud storage
- ✅ Best of both worlds

### Cons:
- ⚠️ Slightly more complex setup

### Structure:
```
Bond-RV-App/ (Git + LFS)
  ├── bond_pipeline/
  ├── bond_data/
  │   └── parquet/*.parquet  # In Git LFS
  └── scripts/
      └── download_raw_data.sh

Cloud Storage:
  └── Universe Historical/*.xlsx
```

---

## Option 5: Network Share / NAS

### Pros:
- ✅ Simple if you have internal network
- ✅ Fast access
- ✅ No cloud costs

### Cons:
- ⚠️ Only works within same network
- ⚠️ Not suitable for remote work

---

## Decision Matrix

| Scenario | Best Option |
|----------|-------------|
| Small team, simple workflow | **Option 1** (Git LFS only) |
| Cost-conscious, large files | **Option 3** (Cloud storage) |
| Need version control for data | **Option 1** or **Option 2** |
| Already using S3/cloud | **Option 4** (Hybrid) |
| Internal network only | **Option 5** (NAS) |

---

## Selected Approach: Option 3 (Dropbox + Git)

### Rationale:
1. **Simplicity**: No Git LFS setup required
2. **Cost**: Use existing Dropbox account
3. **Flexibility**: Easy to share with team
4. **Clean separation**: Code in Git, data in Dropbox
5. **Rebuild philosophy**: Always regenerate parquet files from source

### Implementation:
See `Documentation/Workflows/Dropbox-Workflow.md` for detailed instructions.

---

## Future Considerations

If data volume grows significantly (>100GB) or version control for data becomes critical, consider:
- Migrating to **Option 4 (Hybrid)** with S3 for raw data
- Using **DVC (Data Version Control)** for data versioning
- Implementing incremental backups with versioning

